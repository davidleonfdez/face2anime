from abc import ABC, abstractmethod
from fastai.vision.all import *
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm, weight_norm
from typing import Callable, List, Type


__all__ = ['ConcatPool2d', 'ConditionalBatchNorm2d', 'MiniBatchStdDev', 'CondConvLayer', 'TransformsLayer',
           'ParamRemover', 'DownsamplingOperation2d', 'AvgPoolHalfDownsamplingOp2d', 
           'ConcatPoolHalfDownsamplingOp2d', 'ConvHalfDownsamplingOp2d',  'UpsamplingOperation2d', 
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
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, *args):
        return self.module(*args[:-1])


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


class UpsamplingOperation2d(ABC):
    @abstractmethod
    def get_layer(self, in_ftrs:int=None, out_ftrs:int=None, **op_kwargs) -> nn.Module:
        "Must return a layer that increases the size of the last 2d of the input"
        pass

    
class PixelShuffleUpsamplingOp2d(UpsamplingOperation2d):
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
    def __init__(self, wrapped_up_op):
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
    def __init__(self, in_ftrs, out_ftrs, up_op:UpsamplingOperation2d, 
                 id_up_op:UpsamplingOperation2d, n_extra_convs=1, 
                 upsample_first=True, norm_type=NormType.Batch, 
                 act_cls=nn.ReLU, bn_1st=True, **up_op_kwargs):
        super().__init__()
        
#         norm2 = (NormType.BatchZero if norm_type==NormType.Batch else
#                  NormType.InstanceZero if norm_type==NormType.Instance 
#                  else norm_type)
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
        
        self.act = defaults.activation(inplace=True) if act_cls is None else act_cls()
        
    def forward(self, x): return self.act(self.inner_path(x) + self.id_path(x))
    

class RescaledResBlockUp(ResBlockUp):
    def forward(self, x): return self.act((self.inner_path(x) + self.id_path(x)) / 2)
    

class DenseBlockUp(ResBlockUp):
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
    def __init__(self, in_ftrs, out_ftrs, down_op:DownsamplingOperation2d, 
                 id_down_op:DownsamplingOperation2d, n_extra_convs=1,
                 downsample_first=False, norm_type=NormType.Batch, 
                 act_cls=partial(nn.LeakyReLU, negative_slope=0.2),
                 bn_1st=True, **down_op_kwargs):
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
        
        self.act = defaults.activation(inplace=True) if act_cls is None else act_cls()
        
    def forward(self, x): return self.act(self.inner_path(x) + self.id_path(x))
    

class RescaledResBlockDown(ResBlockDown):
    def forward(self, x): return self.act((self.inner_path(x) + self.id_path(x)) / 2)
    
    
class PseudoDenseBlockDown(ResBlockDown):
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
