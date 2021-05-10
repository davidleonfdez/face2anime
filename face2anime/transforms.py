from fastai.vision.all import *
from fastai.vision.augment import _draw_mask, AffineCoordTfm, HSVTfm
import math
import torch
from torch import zeros_like as t0, ones_like as t1


__all__ = ['Translate', 'DifferentiableHue', 'ADATransforms', 'AdaptiveAugmentsCallback']


def translate_mat(x, p=0.5, draw=None, batch=False):
    "Return a random translation matrix"
    #def _def_draw(x): return torch.distributions.Uniform(-0.125, 0.125).sample([x.size(0), 2])
    def _def_draw(x): return x.new_empty((x.size(0), 2)).uniform_(-0.125, 0.125)
    mask = _draw_mask(x, _def_draw, draw=draw, p=p, batch=batch)
    bias_x = mask[:, 0]
    bias_y = mask[:, 1]
    return affine_mat(t1(bias_x), t0(bias_x), bias_x,
                      t0(bias_x), t1(bias_x), bias_y)


class Translate(AffineCoordTfm):
    "Randomly translate a batch of images with a probability `p`"
    def __init__(self, p=0.5, draw=None, size=None, mode='bilinear', pad_mode=PadMode.Reflection, align_corners=True):#, batch=False):
        aff_fs = partial(translate_mat, p=p, draw=draw)#, batch=batch)
        super().__init__(aff_fs, size=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners, p=p)
        

class _DifferentiableHue():
    def __init__(self, max_hue=0.1, p=0.75, draw=None, batch=False): store_attr()

    def _def_draw(self, x):
        if not self.batch: res = x.new_empty(x.size(0)).uniform_(math.log(1-self.max_hue), -math.log(1-self.max_hue))
        else: res = x.new_zeros(x.size(0)) + random.uniform(math.log(1-self.max_hue), -math.log(1-self.max_hue))
        return torch.exp(res)

    def before_call(self, x):
        self.change = _draw_mask(x, self._def_draw, draw=self.draw, p=self.p, neutral=0., batch=self.batch)

    def __call__(self, x):
        h,s,v = x.unbind(1)
        h = h + self.change[:,None,None]
        h = h % 1.0
        return torch.stack((h, s, v),dim=1)
    

class DifferentiableHue(HSVTfm):
    "Apply change in hue of `max_hue` to batch of images with probability `p`."
    # It's a copy of fastai `Hue` modified not to use non differentiable method `Tensor.set_`
    # Ref: https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.functional.adjust_hue
    def __init__(self,max_hue=0.1, p=0.75, draw=None, batch=False):
        super().__init__(_DifferentiableHue(max_hue, p, draw, batch))


class ADATransforms():
    def __init__(self, p, pad_mode=PadMode.Reflection):
        self.flip = Flip(p=p, pad_mode=pad_mode)
        def draw_90_multiple(x): return (x.new_empty(x.size(0)).uniform_(1, 4) // 1) * 90
        self.rotate_90x = Rotate(p=p, draw=draw_90_multiple, pad_mode=pad_mode)
        # TODO: PadMode.Border could be better or less leaking????
        self.int_translation = Translate(p=p, pad_mode=pad_mode)
        self.iso_zoom = Zoom(p=p, pad_mode=pad_mode, 
                             draw=lambda x: 2 ** x.new_empty(x.size(0)).normal_(0, 0.2))
        p_rot = self._p_rot(p)
        self.pre_rotate = Rotate(p=p_rot, max_deg=180, pad_mode=pad_mode)
        # Here "StyleGAN 2 with limited data" applies anisotropic scaling but warp could be more interesting.
        # Anisotropic scaling would require an easy modification of fastai Zoom to use distinct strength for x and y
        # (new impl.)
        self.warp = Warp(p=p, pad_mode=pad_mode)
        self.post_rotate = Rotate(p=p_rot, max_deg=180, pad_mode=pad_mode)
        def draw_normal_0_0125(x): return x.new_empty((x.size(0), 2)).normal_(0, 0.125)
        self.frac_translation = Translate(p=p, draw=draw_normal_0_0125, pad_mode=pad_mode)
        self.brightness = Brightness(p=p, draw=lambda x: x.new_empty(x.size(0)).normal_(0.5, 0.1))
        self.contrast = Contrast(p=p, draw=lambda x: x.new_empty(x.size(0)).log_normal_(0, 0.5 * math.log(2)))
        self.hue = DifferentiableHue(p=p, max_hue=0.1)
        self.saturation = Saturation(p=p, draw=lambda x: x.new_empty(x.size(0)).log_normal_(0, math.log(2)))
        self.random_erasing = RandomErasing(p=p, sl=0., sh=0.2, min_aspect=0.3)
        # When calling manually, Brightness, Contrast, Hue, Saturation 
        # and RandomErasing need to pass split_idx=0.
        
    def _p_rot(self, p):
        return p * math.sqrt(1 - p)
        
    def to_array(self):
        return [
            self.flip, 
            self.rotate_90x, 
            self.int_translation, 
            self.iso_zoom, 
            self.pre_rotate,
            self.warp, 
            self.post_rotate,
            self.frac_translation, 
            self.brightness, 
            self.contrast,
            self.hue, 
            self.saturation,
            self.random_erasing
        ]
    
    def update_ps(self, p):
        # This is horribly implementation dependant, but not easy to avoid
        self.flip.aff_fs[0].keywords['p'] = p
        self.rotate_90x.aff_fs[0].keywords['p'] = p
        self.int_translation.aff_fs[0].keywords['p'] = p
        self.iso_zoom.aff_fs[0].keywords['p'] = p
        self.pre_rotate.aff_fs[0].keywords['p'] = self._p_rot(p)
        self.warp.coord_fs[0].p = p
        self.post_rotate.aff_fs[0].keywords['p'] = self._p_rot(p)
        self.frac_translation.aff_fs[0].keywords['p'] = p
        self.brightness.fs[0].p = p
        self.contrast.fs[0].p = p
        self.hue.fs[0].p = p
        self.saturation.fs[0].p = p
        self.random_erasing.p = p


class AdaptiveAugmentsCallback(Callback):
    def __init__(self, ada_transforms, crit_preds_tracker, update_cycle_len=5,
                 preds_above_0_overfit_threshold=0.6, bs=64,
                 n_imgs_to_reach_p_1=5e5):
        self.ada_transforms = ada_transforms
        self.preds_tracker = crit_preds_tracker
        self.update_cycle_len = update_cycle_len
        self.preds_above_0_overfit_threshold = preds_above_0_overfit_threshold
        self.n_batches_processed = 0
        self.p = 0
        self.ada_transforms.update_ps(self.p)
        
        # Mult by 2 because we only update in critic mode (half of iters)
        self.p_change = (bs * update_cycle_len * 2) / n_imgs_to_reach_p_1
        self.p_history = []
    
    def _update_p(self):
        overfitting = (self.preds_tracker.real_preds > 0).float().mean() > self.preds_above_0_overfit_threshold
        if overfitting: 
            self.p = min(self.p + self.p_change, 1.)
        else: 
            self.p = max(self.p - self.p_change, 0.)
#         self.p = (min(self.p + self.p_change, 1.) if overfitting
#                   else max(self.p - self.p_change, 0.))
        self.ada_transforms.update_ps(self.p)
        self.p_history.append(self.p)
        self.preds_tracker.reset()
    
    def after_batch(self):
        if self.gan_trainer.gen_mode: return
        self.n_batches_processed += 1
        if self.n_batches_processed % self.update_cycle_len != 0: return
        self._update_p()
