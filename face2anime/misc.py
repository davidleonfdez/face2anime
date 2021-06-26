from face2anime.transforms import normalize_imagenet_tf
from fastai.vision.all import hook_outputs
import torch
import torch.nn as nn
from torchvision.models.vgg import vgg19
from typing import List, Tuple


__all__ = ['FeaturesCalculator']


class FeaturesCalculator:
    "Helps calculate intermediate feature maps of a vgg network."
    def __init__(self, vgg_style_layers_idx:List[int], vgg_content_layers_idx:List[int],
                 vgg:nn.Module=None, input_norm_tf=None, device:torch.device=None):
        self.vgg = vgg19(pretrained=True) if vgg is None else vgg
        self.vgg.eval()
        if device is not None: self.vgg.to(device)
        modules_to_hook = [self.vgg.features[idx] for idx in (*vgg_style_layers_idx, *vgg_content_layers_idx)]
        self.hooks = hook_outputs(modules_to_hook, detach=False)
        self.style_ftrs_hooks = self.hooks[:len(vgg_style_layers_idx)]
        self.content_ftrs_hooks = self.hooks[len(vgg_style_layers_idx):]
        self.input_norm_tf = input_norm_tf
    
    def _get_hooks_out(self, hooks):
        return [h.stored for h in hooks]
    
    def _adjust_normalization(self, img_t):
        should_normalize_input = ((self.input_norm_tf is None)
                                  or (not torch.allclose(self.input_norm_tf.mean, normalize_imagenet_tf.mean))
                                  or (not torch.allclose(self.input_norm_tf.std, normalize_imagenet_tf.std)))
        if should_normalize_input:
            if self.input_norm_tf is not None:
                img_t = self.input_norm_tf.decode(img_t)
            img_t = normalize_imagenet_tf(img_t)
        return img_t
    
    def _forward(self, img_t:torch.Tensor):
        img_t = self._adjust_normalization(img_t)
        self.vgg(img_t)
    
    def calc_style(self, img_t:torch.Tensor) -> List[torch.Tensor]:
        self._forward(img_t)
        return self._get_hooks_out(self.style_ftrs_hooks)
    
    def calc_content(self, img_t:torch.Tensor) -> List[torch.Tensor]:
        self._forward(img_t)
        return self._get_hooks_out(self.content_ftrs_hooks)
    
    def calc_style_and_content(self, img_t:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self._forward(img_t)
        style_ftrs = self._get_hooks_out(self.style_ftrs_hooks)
        content_ftrs = self._get_hooks_out(self.content_ftrs_hooks)
        return style_ftrs, content_ftrs
    
    def __del__(self):
        self.hooks.remove()
