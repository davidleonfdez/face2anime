from abc import ABC, abstractmethod
from fastai.vision.all import *
from fastai.vision.gan import *
import torch.nn as nn
import torch.nn.functional as F

from layers import (CondResBlockUp, DownsamplingOperation2d, MiniBatchStdDev, ResBlockDown, 
                    ResBlockUp, UpsamplingOperation2d)
from torch_utils import add_sn


__all__ = ['custom_generator', 'res_generator', 'NoiseSplitStrategy', 'NoiseSplitEqualLeave1stOutStrategy', 
           'NoiseSplitDontSplitStrategy', 'CondResGenerator', 'SkipGenerator', 'res_critic']


def custom_generator(out_size, n_channels, up_op:UpsamplingOperation2d, in_sz=100, 
                     n_features=64, n_extra_layers=0, sn=True, **kwargs):
    "A basic generator from `in_sz` to images `n_channels` x `out_size` x `out_size`."
    cur_size, cur_ftrs = 4, n_features//2
    while cur_size < out_size:  cur_size *= 2; cur_ftrs *= 2
    layers = [AddChannels(2), ConvLayer(in_sz, cur_ftrs, 4, 1, transpose=True, **kwargs)]
    cur_size = 4
    while cur_size < out_size // 2:
        layers.append(up_op.get_layer(cur_ftrs, cur_ftrs//2, **kwargs))
        cur_ftrs //= 2; cur_size *= 2
    layers += [ConvLayer(cur_ftrs, cur_ftrs, 3, 1, 1, transpose=True, **kwargs) for _ in range(n_extra_layers)]
    layers += [up_op.get_layer(cur_ftrs, n_channels, norm_type=None, act_cls=None), nn.Tanh()]
    generator = nn.Sequential(*layers)
    if sn: add_sn(generator)
    return generator


def res_generator(out_sz, n_ch, up_op:UpsamplingOperation2d, id_up_op:UpsamplingOperation2d,
                  in_sz=100, n_features=64, n_extra_res_blocks=1, n_extra_convs_by_res_block=1,
                  sn=True, bn_1st=True, upblock_cls=ResBlockUp, **kwargs):
    cur_sz, cur_ftrs = 4, n_features//2
    while cur_sz < out_sz:  cur_sz *= 2; cur_ftrs *= 2
    layers = [AddChannels(2), 
              ConvLayer(in_sz, cur_ftrs, 4, 1, transpose=True, bn_1st=bn_1st, **kwargs)]
    cur_sz = 4
    while cur_sz < out_sz // 2:
        layers.append(upblock_cls(cur_ftrs, cur_ftrs//2, up_op, id_up_op, 
                                  n_extra_convs=n_extra_convs_by_res_block,
                                  bn_1st=bn_1st, **kwargs))
        cur_ftrs //= 2; cur_sz *= 2
    layers += [ResBlock(1, cur_ftrs, cur_ftrs, bn_1st=bn_1st, **kwargs) 
               for _ in range(n_extra_res_blocks)]
    layers += [up_op.get_layer(cur_ftrs, n_ch, norm_type=None, act_cls=None), nn.Tanh()]
    generator = nn.Sequential(*layers)
    if sn: add_sn(generator)
    return generator


class NoiseSplitStrategy(ABC):    
    @abstractmethod
    def calc_cond_sz(self, noise_sz, n_splits):
        pass
    
    @abstractmethod
    def split_noise(self, noise, n_splits):
        pass


class NoiseSplitEqualLeave1stOutStrategy(NoiseSplitStrategy):     
    def calc_cond_sz(self, noise_sz, n_splits):
        # Divide by `n_splits+1` to next leave first chunk out of conditions
        return noise_sz // (n_splits + 1)
        
    def split_noise(self, noise, n_splits):
        noise_sz = noise.shape[1]
        cond_sz = self.calc_cond_sz(noise_sz, n_splits)
        return noise.split(cond_sz, 1)[1:]
    
    
class NoiseSplitDontSplitStrategy(NoiseSplitStrategy):
    def calc_cond_sz(self, noise_sz, n_splits):
        return noise_sz
        
    def split_noise(self, noise, n_splits):
        return [noise] * n_splits


class CondResGenerator(nn.Module):
    init_sz = 4
    
    def __init__(self, out_sz, n_ch, up_op:UpsamplingOperation2d, id_up_op:UpsamplingOperation2d, 
                 noise_split_strategy, in_sz=100, n_features=64, n_extra_res_blocks=1, 
                 n_extra_convs_by_res_block=1, sn=True, bn_1st=True, upblock_cls=CondResBlockUp,
                 **kwargs):
        super().__init__()
        self.noise_split_strategy = noise_split_strategy
        cur_sz, cur_ftrs = self.init_sz, n_features//2
        while cur_sz < out_sz:  cur_sz *= 2; cur_ftrs *= 2
        self.initial_layers = nn.Sequential(
            AddChannels(2), 
            ConvLayer(in_sz, cur_ftrs, 4, 1, transpose=True, bn_1st=bn_1st, **kwargs))
        cur_sz = self.init_sz
        n_splits = self.calc_n_upblocks(out_sz)
        self.cond_sz = noise_split_strategy.calc_cond_sz(in_sz, n_splits)
        self.up_layers = nn.ModuleList([])
        while cur_sz < out_sz // 2:
            self.up_layers.append(upblock_cls(cur_ftrs, cur_ftrs//2, self.cond_sz, 
                                              up_op, id_up_op, bn_1st=bn_1st, 
                                              n_extra_convs=n_extra_convs_by_res_block,
                                              **kwargs))
            cur_ftrs //= 2; cur_sz *= 2
            
        self.extra_blocks = nn.Sequential(
            *[ResBlock(1, cur_ftrs, cur_ftrs, bn_1st=bn_1st, **kwargs) 
              for _ in range(n_extra_res_blocks)])
        self.final_conv = up_op.get_layer(cur_ftrs, n_ch, norm_type=None, act_cls=None)
        self.act = nn.Tanh()
        if sn: 
            add_sn(self.initial_layers)
            for up_l in self.up_layers: add_sn(up_l)
            add_sn(self.extra_blocks)
            add_sn(self.final_conv)
    
    @classmethod
    def calc_n_upblocks(cls, out_sz):
        return int(math.log2(out_sz // (2 * cls.init_sz)))
    
    def forward(self, z):
        n_splits = len(self.up_layers)
        z_splits = self.noise_split_strategy.split_noise(z, n_splits)
        x = self.initial_layers(z)
        for zi, up_layer in zip(z_splits, self.up_layers):
            x = up_layer(x, zi)
        x = self.extra_blocks(x)
        return self.act(self.final_conv(x, None))

    
class SkipGenerator(nn.Module):
    def __init__(self, out_sz, n_ch, up_op:UpsamplingOperation2d, id_up_op:UpsamplingOperation2d,
                 in_sz=100, n_features=64, n_extra_res_blocks=1, n_extra_convs_by_res_block=1,
                 sn=True, bn_1st=True, upblock_cls=ResBlockUp, upsample_skips_mode='nearest',
                 skip2rgb_ks=3, skip_act_cls=nn.Tanh, **kwargs):
        super().__init__()
        cur_sz, cur_ftrs = 4, n_features//2
        while cur_sz < out_sz:  cur_sz *= 2; cur_ftrs *= 2
        self.initial_layers = nn.Sequential(
            AddChannels(2), 
            ConvLayer(in_sz, cur_ftrs, 4, 1, transpose=True, bn_1st=bn_1st, **kwargs))
        cur_sz = 4
        self.up_layers = nn.ModuleList([])
        self.skips_torgb = nn.ModuleList([])
        self.upsample_skips_mode = upsample_skips_mode
        while cur_sz < out_sz // 2:
            self.up_layers.append(upblock_cls(cur_ftrs, cur_ftrs//2, up_op, id_up_op, 
                                              n_extra_convs=n_extra_convs_by_res_block,
                                              bn_1st=bn_1st, **kwargs))
            self.skips_torgb.append(ConvLayer(cur_ftrs//2, n_ch, ks=skip2rgb_ks, norm_type=None,
                                              act_cls=skip_act_cls, bias=False))
            cur_ftrs //= 2; cur_sz *= 2
        self.extra_blocks = nn.Sequential(
            *[ResBlock(1, cur_ftrs, cur_ftrs, bn_1st=bn_1st, **kwargs) 
              for _ in range(n_extra_res_blocks)])
        self.last_up = up_op.get_layer(cur_ftrs, n_ch, norm_type=None, act_cls=None)
        self.act = nn.Tanh()
        if sn: 
            add_sn(nn.Sequential(self.initial_layers, self.up_layers, self.skips_torgb,
                                 self.extra_blocks, self.last_up))
    
    def forward(self, x):
        x = self.initial_layers(x)
        out = None
        for up_layer, skip_torgb in zip(self.up_layers, self.skips_torgb):
            x = up_layer(x)
            skip_x = skip_torgb(x)
            out = skip_x if out is None else out + skip_x
            out = F.interpolate(out, scale_factor=2, mode=self.upsample_skips_mode)
        x = self.extra_blocks(x)
        x = self.last_up(x)
        return self.act(out + x)
    

def res_critic(in_size, n_channels, down_op, id_down_op, n_features=64, n_extra_res_blocks=1, 
               norm_type=NormType.Batch, n_extra_convs_by_res_block=0, sn=True, bn_1st=True,
               downblock_cls=ResBlockDown, flatten_full=False, include_minibatch_std=False,
               **kwargs):
    "A basic critic for images `n_channels` x `in_size` x `in_size`."
    layers = [down_op.get_layer(n_channels, n_features, norm_type=None, **kwargs)]
    cur_size, cur_ftrs = in_size//2, n_features
    layers += [ResBlock(1, cur_ftrs, cur_ftrs, norm_type=norm_type, bn_1st=bn_1st, **kwargs) 
               for _ in range(n_extra_res_blocks)]
    while cur_size > 4:
        layers.append(downblock_cls(cur_ftrs, cur_ftrs*2, down_op, id_down_op,
                                    n_extra_convs=n_extra_convs_by_res_block,
                                    norm_type=norm_type, bn_1st=bn_1st, **kwargs))
        cur_ftrs *= 2 ; cur_size //= 2
    init = kwargs.get('init', nn.init.kaiming_normal_)
    if include_minibatch_std: 
        # it may not make sense when using BN, although it, unlike BN, calculates a different
        # stdev for any spatial position.
        layers.append(MiniBatchStdDev())
        cur_ftrs += 1
    #layers += [init_default(nn.Conv2d(cur_ftrs, 1, 4, padding=0, bias=False), init), Flatten()]    
    layers += [init_default(nn.Conv2d(cur_ftrs, 1, 4, padding=0), init), Flatten(full=flatten_full)]
    critic =  nn.Sequential(*layers)
    if sn: add_sn(critic)
    return critic
