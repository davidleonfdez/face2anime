from face2anime.losses import CritPredsTracker
from face2anime.transforms import AdaptiveAugmentsCallback, ADATransforms
from fastai.vision.all import PadMode, setup_aug_tfms, TensorImage
from testing_fakes import DummyObject
import torch


def test_ada_transforms():
    pad_mode = PadMode.Border
    img_sz = (8, 8)
    ada_tfms = ADATransforms(1., img_sz, pad_mode=pad_mode)
    composed_tfms = setup_aug_tfms(ada_tfms.to_array())
    t = TensorImage(torch.rand(5, 3, *img_sz))

    p1_all_changed = True
    for tfm in composed_tfms:
        new_t = tfm(t.clone(), split_idx=0)
        if torch.allclose(t, new_t, atol=1e-4):
            p1_all_changed = False
            break
        t = new_t
    
    ada_tfms.update_ps(0.)
    t = TensorImage(torch.rand(5, 3, *img_sz))
    p0_none_changed = True
    for tfm in composed_tfms:
        new_t = tfm(t.clone(), split_idx=0)
        if not torch.allclose(t, new_t, atol=1e-4):
            p0_none_changed = False
            break
        t = new_t

    pad_mode_ok = True
    for tfm in ada_tfms.to_array():
        if getattr(tfm, 'pad_mode', pad_mode) != pad_mode:
            pad_mode_ok = False
            break
        
    assert p1_all_changed
    assert p0_none_changed
    assert pad_mode_ok


class FakeADATransforms():
    def __init__(self, p): self.p = p
    def update_ps(self, p): self.p = p


def test_adaptive_augments_cb():
    ada_tfms = FakeADATransforms(0.5)
    crit_preds_tracker = CritPredsTracker(reduce_batch=False)
    bs = 4
    ada_cb = AdaptiveAugmentsCallback(ada_tfms, crit_preds_tracker, update_cycle_len=2,
                                      preds_above_0_overfit_threshold=0.5, bs=bs,
                                      n_imgs_to_reach_p_1=64)
    initial_p = ada_cb.p
    # monkey patch expected gan_trainer attrs
    ada_cb.gan_trainer = DummyObject()
    ada_cb.gan_trainer.gen_mode = False
    # 2 * 2 (update_cycle_len) * 4 (bs) / 64
    expected_p_change = 0.25
    
    def _sim_one_batch(real_preds):
        crit_preds_tracker(real_preds, torch.rand(bs))
        ada_cb.after_batch()
    
    _sim_one_batch(torch.Tensor([2.7, 0.3, 1.3, -0.2]))
    p_after_one_batch = ada_cb.p
    _sim_one_batch(torch.Tensor([2.7, 0.3, -1.3, -0.2]))
    p_after_two_batches = ada_cb.p
    _sim_one_batch(torch.Tensor([-0.7, -0.3, 0.5, 0.2]))
    _sim_one_batch(torch.Tensor([0.1, -0.3, 0.2, 0.2]))
    p_after_four_batches = ada_cb.p
    _sim_one_batch(torch.Tensor([-0.7, -0.3, 0.5, 0.2]))
    _sim_one_batch(torch.Tensor([2.1, -0.3, -0.2, 4.2]))
    p_after_six_batches = ada_cb.p
    
    assert initial_p == 0
    assert p_after_one_batch == 0
    assert p_after_two_batches == expected_p_change
    assert p_after_four_batches == 2 * expected_p_change
    assert p_after_six_batches == expected_p_change
