from abc import ABC, abstractmethod
from enum import auto, Flag
from face2anime.gen_utils import coalesce
from face2anime.misc import FeaturesCalculator, gram_matrix
from face2anime.torch_utils import vectorize_upper_diag
from fastai.vision.all import *
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, weight_norm
from typing import Callable, List, Type


__all__ = ['ConcatPool2d', 'ConditionalBatchNorm2d', 'MiniBatchStdDev', 'CondConvLayer', 'TransformsLayer',
           'ParamRemover', 'FeatureStatType', 'PretrainedVGGSource', 'ParentNetSource', 'FeaturesStats', 
           'DownsamplingOperation2d', 'AvgPoolHalfDownsamplingOp2d', 'ConcatPoolHalfDownsamplingOp2d', 
           'ConvHalfDownsamplingOp2d', 'ZeroDownsamplingOp2d','UpsamplingOperation2d',  
           'PixelShuffleUpsamplingOp2d', 'InterpConvUpsamplingOp2d', 'CondInterpConvUpsamplingOp2d',  
           'ConvX2UpsamplingOp2d', 'CondConvX2UpsamplingOp2d', 'ParamRemoverUpsamplingOp2d', 'MiniResBlock', 
           'ResBlockUp', 'RescaledResBlockUp', 'DenseBlockUp', 'CondResBlockUp', 'ResBlockDown', 
           'RescaledResBlockDown', 'PseudoDenseBlockDown', 'DenseBlockDown']


class ConcatPool2d(nn.Module):
    "Layer that concats `AvgPool2d` and `MaxPool2d`"
    def __init__(self, ks, stride=None, padding=0):
        super().__init__()
        self.ap = nn.AvgPool2d(ks, stride, padding)
        self.mp = nn.MaxPool2d(ks, stride, padding)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class ConditionalBatchNorm2d(nn.Module):
    """BN layer whose gain (gamma) and bias (beta) params also depend on an external condition vector."""
    def __init__(self, n_ftrs:int, cond_sz:int, gain_init:Callable=None, bias_init:Callable=None):
        super().__init__()
        self.n_ftrs = n_ftrs
        # Don't learn beta and gamma inside self.bn (fix to irrelevance: beta=1, gamma=0)
        self.bn = nn.BatchNorm2d(n_ftrs, affine=False)
        self.gain = nn.Linear(cond_sz, n_ftrs, bias=False)
        self.bias = nn.Linear(cond_sz, n_ftrs, bias=False)        
        if gain_init is None: gain_init = nn.init.zeros_
        if bias_init is None: bias_init = nn.init.zeros_
        init_default(self.gain, gain_init)
        init_default(self.bias, bias_init)

    def forward(self, x, cond):
        out = self.bn(x)
        gamma = 1 + self.gain(cond)
        beta = self.bias(cond)
        out = gamma.view(-1, self.n_ftrs, 1, 1) * out + beta.view(-1, self.n_ftrs, 1, 1)
        return out


class ConditionalInstanceNorm2d(nn.Module):
    """IN layer whose gain (gamma) and bias (beta) params also depend on an external condition vector."""
    def __init__(self, n_ftrs:int, cond_sz:int, gain_init=None, bias_init=None):
        super().__init__()
        self.n_ftrs = n_ftrs
        # Don't learn beta and gamma inside self.inn (fix to irrelevance: beta=1, gamma=0)
        self.inn = nn.InstanceNorm2d(n_ftrs, affine=False)
        self.gain = nn.Linear(cond_sz, n_ftrs, bias=False)
        self.bias = nn.Linear(cond_sz, n_ftrs, bias=False)        
        if gain_init is None: gain_init = nn.init.zeros_
        if bias_init is None: bias_init = nn.init.zeros_
        init_default(self.gain, gain_init)
        init_default(self.bias, bias_init)

    def forward(self, x, cond):
        out = self.inn(x)
        gamma = 1 + self.gain(cond)
        beta = self.bias(cond)
        out = gamma.view(-1, self.n_ftrs, 1, 1) * out + beta.view(-1, self.n_ftrs, 1, 1)
        return out


class MiniBatchStdDev(nn.Module):
    """Layer that appends to every element of a batch a new ftr map containing the std of its group."""
    def __init__(self, group_sz=4, unbiased_std=False):
        super().__init__()
        self.group_sz = group_sz
        self.unbiased_std = unbiased_std
        
    def forward(self, x):
        bs, n_ch, h, w = x.shape
        # We assume bs is divisible by self.group_sz
        x_groups = x.view(-1, self.group_sz, n_ch, h, w)
        stds_by_chw = x_groups.std(dim=1, unbiased=self.unbiased_std)
        mean_std = stds_by_chw.mean(dim=[1, 2, 3], keepdim=True)
        new_ftr_map = mean_std.unsqueeze(-1).repeat(1, self.group_sz, 1, h, w).view(bs, 1, h, w)
        return torch.cat([x, new_ftr_map], axis=1)


class TransformsLayer(nn.Module):
    "Applies a chain of transforms to the input tensor"
    def __init__(self, tfms:List[RandTransform]):
        super().__init__()
        self.tfms = tfms
        
    def forward(self, x):
        # Needed to comply with type restrictions of fastai augmentations
        if not isinstance(x, TensorImage):
            x = TensorImage(x)  
        for tfm in self.tfms: 
            x = tfm(x, split_idx=0)
        return x


class ParamRemover(nn.Module):
    "Forwards the input args minus the last one to a wrapped module."
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, *args):
        return self.module(*args[:-1])


class FeatureStatType(Flag):
    NONE = 0
    MEAN = auto()
    STD = auto()
    CORRELATIONS = auto()


class FeaturesStatsSource(ABC):
    @abstractmethod
    def get_ftrs(self, x) -> List[nn.Module]:
        pass

    @property
    def set_parent(self, parent:nn.Module):
        pass


class PretrainedVGGSource(FeaturesStatsSource):
    def __init__(self, ftrs_calc=None, input_norm_tf=None, device=None):
        if ftrs_calc is None:
            layers_idxs = [6, 11, 20]
            ftrs_calc = FeaturesCalculator(layers_idxs, [],
                                           input_norm_tf=input_norm_tf,
                                           device=device)
        self.ftrs_calc = ftrs_calc

    def get_ftrs(self, x):
        return self.ftrs_calc.calc_style(x)


class ParentNetSource(FeaturesStatsSource):
    def __init__(self, layer_types=None):
        if layer_types is None: layer_types = nn.Conv2d
        self.layer_types = layer_types

    def set_parent(self, parent:nn.Module):
        self.parent = parent
        layers = [m for m in parent.modules() if isinstance(m, self.layer_types)]
        self.hooks = hook_outputs(layers)

    def get_ftrs(self, x):
        self.parent(x)
        return [h.stored for h in self.hooks]

    def __del__(self):
        if hasattr(self, 'hooks'): self.hooks.remove()


class FeaturesStats(nn.Module):
    "Returns statistics from intermediate feature maps obtained when forwarding input through a certain network."
    def __init__(self, in_sz, in_ch, out_ftrs=None, input_norm_tf=None, device=None,
                 ftrs_stats=FeatureStatType.MEAN, ftrs_stats_source:FeaturesStatsSource=None,
                 max_corr_in_ftrs=256):
        super().__init__()
        assert ftrs_stats not in (None, FeatureStatType.NONE)
        self.include_mean = FeatureStatType.MEAN in ftrs_stats
        self.include_std = FeatureStatType.STD in ftrs_stats
        self.include_correlations = FeatureStatType.CORRELATIONS in ftrs_stats
        if ftrs_stats_source is None:
            ftrs_stats_source = PretrainedVGGSource(input_norm_tf=input_norm_tf,
                                                    device=device)
        self.ftrs_stats_source = ftrs_stats_source
        self.linear = None

        if (out_ftrs is not None) or self.include_correlations:
            with torch.no_grad():
                test_out = self.ftrs_stats_source.get_ftrs(torch.rand(1, in_ch, in_sz, in_sz, device=device))
            n_total_ftrs = sum([ftrs.shape[1] for ftrs in test_out])          

        if out_ftrs is not None:
            lin_in_ftrs = 0
            if self.include_mean: lin_in_ftrs += n_total_ftrs 
            if self.include_std: lin_in_ftrs += n_total_ftrs
            if self.include_correlations: lin_in_ftrs += n_total_ftrs
            self.linear = spectral_norm(nn.Linear(lin_in_ftrs, out_ftrs))

        if self.include_correlations:
            self.corr_ftrs_layers_idxs = [i for i, ftrs in enumerate(test_out) 
                                          if ftrs.shape[1] <= max_corr_in_ftrs]
            corr_lin_in_ftrs = sum([((test_out[i].shape[1]**2 - test_out[i].shape[1]) // 2) 
                                    for i in self.corr_ftrs_layers_idxs]) 
            #corr_lin_in_ftrs = (n_total_ftrs**2 - n_total_ftrs) // 2
            self.correlations_linear = spectral_norm(nn.Linear(corr_lin_in_ftrs, n_total_ftrs))
          
    def forward(self, x):
        ftrs_by_layer = self.ftrs_stats_source.get_ftrs(x)
        ftrs_stats_by_layer = []
        if self.include_mean:
            means = [ftrs.mean(axis=(2, 3)) for ftrs in ftrs_by_layer]
            ftrs_stats_by_layer.extend(means)
        if self.include_std:
            stds = [ftrs.std(axis=(2, 3)) for ftrs in ftrs_by_layer]
            ftrs_stats_by_layer.extend(stds)
        if self.include_correlations:
            correlations = [vectorize_upper_diag(gram_matrix(ftrs_by_layer[i]), offset=1) 
                            for i in self.corr_ftrs_layers_idxs]
            correlations_reduced = self.correlations_linear(torch.cat(correlations, axis=1))
            ftrs_stats_by_layer.append(correlations_reduced)
        out = torch.cat(ftrs_stats_by_layer, axis=1)
        if self.linear is not None: 
            out = self.linear(out)
        return out


class CondConvLayer(nn.Module):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and conditional `norm_type` layers."
    @delegates(nn.Conv2d)
    def __init__(self, ni, nf, cond_sz, ks=3, stride=1, padding=None, bias=None, ndim=2, 
                 norm_type=NormType.Batch, bn_1st=True, act_cls=defaults.activation, 
                 transpose=False, init='auto', xtra=None, xtra_begin=None, bias_std=0.01, 
                 **kwargs):
        super().__init__()
        if padding is None: padding = ((ks-1)//2 if not transpose else 0)
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None: bias = not (bn or inn)
        conv_func = nn.ConvTranspose2d if transpose else nn.Conv2d
        conv = conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs)
        act = None if act_cls is None else act_cls()
        init_linear(conv, act, init=init, bias_std=bias_std)
        if   norm_type==NormType.Weight:   conv = weight_norm(conv)
        elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
        self.layers = nn.ModuleList([conv])
        act_bn = []
        if act is not None: act_bn.append(act)
        if bn: act_bn.append(ConditionalBatchNorm2d(nf, cond_sz))
        if inn: act_bn.append(ConditionalInstanceNorm2d(nf, cond_sz))
        if bn_1st: act_bn.reverse()
        self.layers += act_bn
        if xtra_begin: self.layers.insert(0, xtra_begin)
        if xtra: self.layers.append(xtra)
    
    def forward(self, x, cond):
        for l in self.layers:
            if isinstance(l, (ConditionalBatchNorm2d, ConditionalInstanceNorm2d)):
                x = l(x, cond)
            else:
                x = l(x)
        return x

    
class DownsamplingOperation2d(ABC):
    @abstractmethod
    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        "Must return a layer that increases the size of the last 2d of the input"
        pass

    
class AvgPoolHalfDownsamplingOp2d(DownsamplingOperation2d):
    "Returns a module that downsamples the input spatially using an AvgPool layer."
    def __init__(self, conv_ks=3, act_cls=None, norm_type=None):
        self.conv_ks = conv_ks
        self.act_cls = act_cls
        self.norm_type = norm_type

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        layers = [nn.AvgPool2d(2)]
        if out_ftrs != in_ftrs:
            layers.append(ConvLayer(in_ftrs, out_ftrs, ks=self.conv_ks, act_cls=self.act_cls,
                                    norm_type=self.norm_type, **op_kwargs))
        return nn.Sequential(*layers)

    
class ConcatPoolHalfDownsamplingOp2d(DownsamplingOperation2d):
    "Returns a module that downsamples the input spatially using a ConcatPool layer."
    def __init__(self, conv_ks=3, act_cls=None, norm_type=None, always_add_conv=False):
        self.conv_ks = conv_ks
        self.act_cls = act_cls
        self.norm_type = norm_type
        self.always_add_conv = always_add_conv

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        layers = [ConcatPool2d(2)]
        if (out_ftrs != (in_ftrs*2)) or self.always_add_conv:
            layers.append(ConvLayer(in_ftrs*2, out_ftrs, ks=self.conv_ks, act_cls=self.act_cls,
                                    norm_type=self.norm_type, **op_kwargs))
        return nn.Sequential(*layers)
    

class ConvHalfDownsamplingOp2d(DownsamplingOperation2d):
    "Returns a module that downsamples the input spatially using a strided convolution."
    def __init__(self, ks=4, padding=1, act_cls:Type[nn.Module]=None, norm_type=None,
                 bn_1st=True):
        self.ks=ks
        self.act_cls = act_cls
        self.padding = padding
        self.norm_type = norm_type
        self.bn_1st = bn_1st

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        assert (in_ftrs is not None) and (out_ftrs is not None), \
            "in_ftrs and out_ftrs must both be valued for this UpsamplingOperation"
        if 'act_cls' not in op_kwargs:
            op_kwargs['act_cls'] = self.act_cls
        if 'norm_type' not in op_kwargs:
            op_kwargs['norm_type'] = self.norm_type
        if 'padding' not in op_kwargs:
            op_kwargs['padding'] = self.padding
#         conv = ConvLayer(in_ftrs, out_ftrs, self.ks, 2, bias=False, bn_1st=self.bn_1st,
#                          **op_kwargs)
        conv = ConvLayer(in_ftrs, out_ftrs, self.ks, 2, bn_1st=self.bn_1st,
                         **op_kwargs)
        return conv


class ZeroDownsamplingOp2d(DownsamplingOperation2d):
    "Returns a module whose output is always zero."
    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        return Lambda(lambda t: 0)


class UpsamplingOperation2d(ABC):
    @abstractmethod
    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        "Must return a layer that increases the size of the last 2d of the input"
        pass

    
class PixelShuffleUpsamplingOp2d(UpsamplingOperation2d):
    "Returns a module that upsamples the input spatially using pixel shuffle."
    def __init__(self, scale_factor=2, blur=False, n_extra_convs=0, act_cls:Type[nn.Module]=None, norm_type=None):
        self.scale_factor = scale_factor
        self.blur = blur
        self.n_extra_convs = n_extra_convs
        self.act_cls = act_cls
        self.norm_type = norm_type
    
    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs):
        if 'act_cls' not in op_kwargs:
            op_kwargs['act_cls'] = self.act_cls
        if 'blur' not in op_kwargs:
            op_kwargs['blur'] = self.blur
        if 'norm_type' not in op_kwargs:
            op_kwargs['norm_type'] = self.norm_type
        layer = PixelShuffle_ICNR(in_ftrs, out_ftrs, self.scale_factor, **op_kwargs)
        if self.n_extra_convs > 0:
            extra_convs = [ConvLayer(out_ftrs, out_ftrs, norm_type=self.norm_type)
                           for _ in range(self.n_extra_convs)]
            layer = nn.Sequential(layer, *extra_convs)
        return layer

    
class InterpConvUpsamplingOp2d(UpsamplingOperation2d):
    "Returns a module that upsamples the input using interpolation followed by a neutral conv."
    def __init__(self, scale_factor=2, mode='nearest', ks=3, act_cls:Type[nn.Module]=None,
                 norm_type=NormType.Batch, bn_1st=True):
        self.scale_factor = scale_factor
        self.mode = mode
        self.act_cls = act_cls
        self.ks = ks
        self.norm_type = norm_type
        self.bn_1st = bn_1st

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        if 'act_cls' not in op_kwargs:
            op_kwargs['act_cls'] = self.act_cls
        if 'norm_type' not in op_kwargs:
            op_kwargs['norm_type'] = self.norm_type
        return nn.Sequential(nn.Upsample(scale_factor=self.scale_factor, mode=self.mode),
                             ConvLayer(in_ftrs, out_ftrs, self.ks, bn_1st=self.bn_1st, 
                                       **op_kwargs))


class CondInterpConvUpsamplingOp2d(UpsamplingOperation2d):
    """Returns a module that upsamples the input using interpolation followed by a neutral conv.
    
    The input passed to the module should contain two parameters, with the last one being the 
    condition vector, which is used as the second param of a conditional BN/IN layer included 
    after the convolution"""
    def __init__(self, cond_sz, scale_factor=2, mode='nearest', ks=3, act_cls:Type[nn.Module]=None,
                 norm_type=NormType.Batch, bn_1st=True):
        self.cond_sz = cond_sz
        self.scale_factor = scale_factor
        self.mode = mode
        self.act_cls = act_cls
        self.ks = ks
        self.norm_type = norm_type
        self.bn_1st = bn_1st

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        if 'act_cls' not in op_kwargs:
            op_kwargs['act_cls'] = self.act_cls
        if 'norm_type' not in op_kwargs:
            op_kwargs['norm_type'] = self.norm_type
        upsample = nn.Upsample(scale_factor=self.scale_factor, mode=self.mode)
        return CondConvLayer(in_ftrs, out_ftrs, self.cond_sz, self.ks, bn_1st=self.bn_1st, 
                             xtra_begin=upsample, **op_kwargs)


class ConvX2UpsamplingOp2d(UpsamplingOperation2d):
    "Returns a module that upsamples the input spatially using a strided transpose convolution."
    def __init__(self, ks=4, act_cls:Type[nn.Module]=None, padding=1, output_padding=0,
                 norm_type=NormType.Batch, bn_1st=True):
        #self.apply_sn = apply_sn
        self.ks=ks
        self.act_cls = act_cls
        self.padding = padding
        self.output_padding = output_padding
        self.norm_type = norm_type
        self.bn_1st=bn_1st

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        assert (in_ftrs is not None) and (out_ftrs is not None), \
            "in_ftrs and out_ftrs must both be valued for this UpsamplingOperation"
        if 'act_cls' not in op_kwargs:
            op_kwargs['act_cls'] = self.act_cls
        if 'norm_type' not in op_kwargs:
            op_kwargs['norm_type'] = self.norm_type
        conv = ConvLayer(in_ftrs, out_ftrs, self.ks, 2, self.padding, transpose=True, 
                         bias=False, output_padding=self.output_padding, bn_1st=self.bn_1st,
                         **op_kwargs)
#         if self.apply_sn and (op_kwargs.get('norm_type') != NormType.Spectral): 
#             add_sn(conv)
        return conv


class CondConvX2UpsamplingOp2d(UpsamplingOperation2d):
    """Returns a module that upsamples the input spatially using a strided transpose convolution.
    
    The input passed to the module should contain two parameters, with the last one being the 
    condition vector, which is used as the second param of a conditional BN/IN layer included 
    after the convolution"""    
    def __init__(self, cond_sz, ks=4, act_cls:Type[nn.Module]=None, padding=1, output_padding=0,
                 norm_type=NormType.Batch, bn_1st=True):
        self.cond_sz = cond_sz
        self.ks=ks
        self.act_cls = act_cls
        self.padding = padding
        self.output_padding = output_padding
        self.norm_type = norm_type
        self.bn_1st=bn_1st

    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        assert (in_ftrs is not None) and (out_ftrs is not None), \
            "in_ftrs and out_ftrs must both be valued for this UpsamplingOperation"
        if 'act_cls' not in op_kwargs:
            op_kwargs['act_cls'] = self.act_cls
        if 'norm_type' not in op_kwargs:
            op_kwargs['norm_type'] = self.norm_type
        conv = CondConvLayer(in_ftrs, out_ftrs, self.cond_sz, self.ks, 2, self.padding, 
                             transpose=True, bias=False, output_padding=self.output_padding, 
                             bn_1st=self.bn_1st, **op_kwargs)
        return conv


class ParamRemoverUpsamplingOp2d(UpsamplingOperation2d):
    "Returns a module that forwards the input args minus the last one to the wrapped up op."
    def __init__(self, wrapped_up_op:UpsamplingOperation2d):
        self.wrapped_up_op = wrapped_up_op
        
    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        return ParamRemover(self.wrapped_up_op.get_layer(in_ftrs, out_ftrs, **op_kwargs))


class MiniResBlock(nn.Module):
    "Residual block with just one convolution in main path"
    def __init__(self, in_ftrs, out_ftrs, norm_type=None, act_cls=nn.ReLU, **conv_kwargs):
        super().__init__()
        norm_type = (NormType.BatchZero if norm_type==NormType.Batch else
                     NormType.InstanceZero if norm_type==NormType.Instance 
                     else norm_type)
        self.convpath = ConvLayer(in_ftrs, out_ftrs, norm_type=norm_type, act_cls=act_cls,
                                  **conv_kwargs)
        self.act = act_cls()
        
    def forward(self, x): return self.act(self.convpath(x) + x)


class ResBlockUp(nn.Module):
    "Upsampling block with an identity connection that should include some simple upsampling operation."
    def __init__(self, in_ftrs, out_ftrs, up_op:UpsamplingOperation2d, 
                 id_up_op:UpsamplingOperation2d, n_extra_convs=1, 
                 upsample_first=True, norm_type=NormType.Batch, 
                 act_cls=nn.ReLU, bn_1st=True, hook=None, **up_op_kwargs):
        super().__init__()
        self.hook = hook        

        up_layer = up_op.get_layer(in_ftrs, out_ftrs, **up_op_kwargs)
        extra_convs_ftrs = out_ftrs if upsample_first else in_ftrs
        inner_path_ls = ([ConvLayer(extra_convs_ftrs, extra_convs_ftrs, bn_1st=bn_1st, norm_type=norm_type,
                                    #norm_type=norm_type if ((i<n_extra_convs-1) or not upsample_first) else norm2,
                                    act_cls=act_cls if ((i<n_extra_convs-1) or (not upsample_first) or (not bn_1st)) else None)
                         for i in range(n_extra_convs)])
        inner_path_ls.insert(0 if upsample_first else len(inner_path_ls),
                            up_layer)
        self.inner_path = nn.Sequential(*inner_path_ls)
        
        self.id_path = id_up_op.get_layer(in_ftrs, out_ftrs)

        if hook is not None: 
            self.hook_adapter = ConvLayer(out_ftrs*2, out_ftrs, bn_1st=bn_1st, 
                                          norm_type=norm_type, 
                                          act_cls=act_cls if not bn_1st else None)
        
        self.act = defaults.activation(inplace=True) if act_cls is None else act_cls()
        
    def forward(self, x): 
        out = self.inner_path(x) + self.id_path(x)
        if self.hook is not None:
            out = torch.cat((out, self.hook.stored), axis=1)
            out = self.hook_adapter(out)
        return self.act(out)
    

class RescaledResBlockUp(ResBlockUp):
    "Like ResBlockUp, but divides the output by half."
    def forward(self, x): return self.act((self.inner_path(x) + self.id_path(x)) / 2)
    

class DenseBlockUp(ResBlockUp):
    "Upsampling dense block with an identity connection that should include some simple upsampling operation."
    def __init__(self, in_ftrs, out_ftrs, up_op:UpsamplingOperation2d, 
                 id_up_op:UpsamplingOperation2d, n_extra_convs=1, 
                 upsample_first=True, norm_type=NormType.Batch, 
                 act_cls=nn.ReLU, bn_1st=True, **up_op_kwargs):
        super().__init__(in_ftrs, out_ftrs, up_op, id_up_op, n_extra_convs=n_extra_convs, 
                         upsample_first=upsample_first, norm_type=norm_type, 
                         act_cls=act_cls, bn_1st=bn_1st, **up_op_kwargs)
        
        self.final_conv = ConvLayer(out_ftrs*2, out_ftrs, bn_1st=bn_1st, norm_type=norm_type,
                                    act_cls=act_cls)       
        
    def forward(self, x): 
        return self.final_conv(self.act(torch.cat([self.inner_path(x), self.id_path(x)], axis=1)))


class CondResBlockUp(nn.Module):
    """Upsampling block with an identity connection that should include some simple upsampling operation.
    
    The input passed to the module should contain two parameters, with the last one being the condition 
    vector, which is used as the second param of the conditional BN/IN layers included in the block.
    """
    def __init__(self, in_ftrs, out_ftrs, cond_sz, up_op:UpsamplingOperation2d, 
                 id_up_op:UpsamplingOperation2d, n_extra_convs=1, 
                 upsample_first=True, norm_type=NormType.Batch, 
                 act_cls=nn.ReLU, bn_1st=True, **up_op_kwargs):
        super().__init__()
        
        up_layer = up_op.get_layer(in_ftrs, out_ftrs, **up_op_kwargs)
        extra_convs_ftrs = out_ftrs if upsample_first else in_ftrs
        inner_path_ls = ([CondConvLayer(extra_convs_ftrs, extra_convs_ftrs, cond_sz, bn_1st=bn_1st, norm_type=norm_type,
                                    act_cls=act_cls if ((i<n_extra_convs-1) or (not upsample_first) or (not bn_1st)) else None)
                         for i in range(n_extra_convs)])
        inner_path_ls.insert(0 if upsample_first else len(inner_path_ls),
                            up_layer)
        self.inner_path = nn.ModuleList(inner_path_ls)
        
        self.id_path = id_up_op.get_layer(in_ftrs, out_ftrs)
        
        self.act = defaults.activation(inplace=True) if act_cls is None else act_cls()
        
    def forward(self, x, cond): 
        inner_out = x
        for l in self.inner_path:
            inner_out = l(inner_out, cond)
        return self.act(inner_out + self.id_path(x, cond))


class ResBlockDown(nn.Module):
    "Downsampling block with an identity connection that should include some simple downsampling operation."
    def __init__(self, in_ftrs, out_ftrs, down_op:DownsamplingOperation2d, 
                 id_down_op:DownsamplingOperation2d, n_extra_convs=1,
                 downsample_first=False, norm_type=NormType.Batch, 
                 act_cls=partial(nn.LeakyReLU, negative_slope=0.2),
                 bn_1st=True, main_act_cls=None, **down_op_kwargs):
        super().__init__()
        
#         norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
#                  NormType.InstanceZero if norm_type==NormType.Instance 
#                  else norm_type)
        down_layer = down_op.get_layer(in_ftrs, out_ftrs, **down_op_kwargs)
        extra_convs_ftrs = out_ftrs if downsample_first else in_ftrs
        inner_path_ls = ([ConvLayer(extra_convs_ftrs, extra_convs_ftrs, bn_1st=bn_1st, norm_type=norm_type,
                                    #norm_type=norm_type if ((i<n_extra_convs-1) or not downsample_first) else norm2,
                                    act_cls=act_cls if ((i<n_extra_convs-1) or (not downsample_first) or (not bn_1st)) else None)
                         for i in range(n_extra_convs)])
        inner_path_ls.insert(0 if downsample_first else len(inner_path_ls),
                            down_layer)
        self.inner_path = nn.Sequential(*inner_path_ls)
        
        self.id_path = id_down_op.get_layer(in_ftrs, out_ftrs, bias=False)
        
        main_act_cls = coalesce(main_act_cls, act_cls, partial(defaults.activation, inplace=True))
        self.act = main_act_cls()
        
    def forward(self, x): return self.act(self.inner_path(x) + self.id_path(x))
    

class RescaledResBlockDown(ResBlockDown):
    "Like ResBlockDown, but divides the output by half."
    def forward(self, x): return self.act((self.inner_path(x) + self.id_path(x)) / 2)
    
    
class PseudoDenseBlockDown(ResBlockDown):
    """Downsampling dense block with an identity connection that should include some simple downsampling operation.
    
    Its dowsampling operations reduce the number of features to out_ftrs//2 before the concatenation in order to
    enforce the output to match `out_ftrs`."""
    def __init__(self, in_ftrs, out_ftrs, down_op:DownsamplingOperation2d, 
                 id_down_op:DownsamplingOperation2d, n_extra_convs=1,
                 downsample_first=False, norm_type=NormType.Batch, 
                 act_cls=partial(nn.LeakyReLU, negative_slope=0.2),
                 bn_1st=True, **down_op_kwargs):
        super().__init__(in_ftrs, out_ftrs//2, down_op, id_down_op, n_extra_convs=n_extra_convs,
                         downsample_first=downsample_first, norm_type=norm_type, 
                         act_cls=act_cls, bn_1st=bn_1st, **down_op_kwargs)      
        
    def forward(self, x): 
        return self.act(torch.cat([self.inner_path(x), self.id_path(x)], axis=1)) 
    
    
class DenseBlockDown(ResBlockDown):
    """Downsampling dense block with an identity connection that should include some simple downsampling operation.
    
    It includes an extra neutral ConvLayer that reduces the number of features after the concatenation
    to match `out_ftrs`."""
    def __init__(self, in_ftrs, out_ftrs, down_op:DownsamplingOperation2d, 
                 id_down_op:DownsamplingOperation2d, n_extra_convs=1,
                 downsample_first=False, norm_type=NormType.Batch, 
                 act_cls=partial(nn.LeakyReLU, negative_slope=0.2),
                 bn_1st=True, **down_op_kwargs):
        super().__init__(in_ftrs, out_ftrs, down_op, id_down_op, n_extra_convs=n_extra_convs,
                         downsample_first=downsample_first, norm_type=norm_type, 
                         act_cls=act_cls, bn_1st=bn_1st, **down_op_kwargs)
        
        self.final_conv = ConvLayer(out_ftrs*2, out_ftrs, bn_1st=bn_1st, norm_type=norm_type,
                                    act_cls=act_cls)
        
    def forward(self, x): 
        return self.final_conv(self.act(torch.cat([self.inner_path(x), self.id_path(x)], axis=1)))
