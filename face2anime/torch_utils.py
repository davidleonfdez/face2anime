import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn.utils.spectral_norm import SpectralNorm


__all__ = ['is_conv', 'has_sn_hook', 'every_conv_has_sn', 'get_mean_weights', 
           'add_sn']


conv_types = [nn.Conv1d, nn.Conv2d, nn.Conv3d,
              nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]


def is_conv(m:nn.Module):
    return any(isinstance(m, conv_type) for conv_type in conv_types)


def has_sn_hook(m:nn.Module):
    return any(isinstance(h, SpectralNorm) 
               for h in m._forward_pre_hooks.values())


def every_conv_has_sn(module:nn.Module):
    for m in module.modules():
        if is_conv(m) and not has_sn_hook(m):
            return False
    return True


def get_mean_weights(m:nn.Module, layer_types):
    all_modules_dict = dict(m.named_modules())
    result = {}
    for param_name, param in m.named_parameters():
        module = all_modules_dict[param_name.rsplit('.', 1)[0]]
        if isinstance(module, layer_types):
            result[param_name] = param.data.abs().mean()
    return result


def add_sn(m:nn.Module):
    idxs_to_edit = []
    for i, l in enumerate(m.children()):
        if isinstance(l, nn.Conv2d) or isinstance(l, nn.ConvTranspose2d):
            idxs_to_edit.append(i)
        add_sn(l)
    for i in idxs_to_edit:
        m[i] = spectral_norm(m[i])
