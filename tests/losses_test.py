from face2anime.losses import (ContentLossCallback, CritPredsTracker, CycleConsistencyLoss, 
                               CycleGANLoss, IdentityLoss, LossWrapper)
from face2anime.networks import CycleCritic, CycleGenerator
from fastai.vision.all import Lambda
from fastai.vision.gan import GANLoss, GANModule, gan_loss_from_func
import math
from testing_fakes import DummyObject
import torch


class FakeFeaturesCalculator:
    def __init__(self, *args, **kwargs): pass
    def calc_content(self, t): return t.mean(axis=1)


def test_content_loss_callback():
    ftrs_calc=FakeFeaturesCalculator()
    cb = ContentLossCallback(ftrs_calc=FakeFeaturesCalculator())
    cb.x = torch.Tensor([[1., 2, 3]])
    cb.pred = torch.Tensor([[3, 3.5, 4]])
    cb.learn = DummyObject()
    cb.learn.loss_grad = 0.
    cb.learn.loss_func = DummyObject()
    cb.gan_trainer = DummyObject()
    cb.gan_trainer.gen_mode = False
    cb.after_loss()
    loss_crit_mode = cb.learn.loss_grad
    cb.gan_trainer.gen_mode = True
    cb.after_loss()
    loss_gen_mode = cb.learn.loss_grad

    assert math.isclose(loss_crit_mode, 0.)
    assert math.isclose(loss_gen_mode, 2.25)


def test_cycle_gan_loss():
    g_inner_loss_func = lambda *args: 0
    c_inner_loss_func = lambda out, target: (out-target).abs().mean()
    g_loss_func, c_loss_func = gan_loss_from_func(g_inner_loss_func, c_inner_loss_func)
    g1 = Lambda(lambda t: t*2)
    g2 = Lambda(lambda t: t/2)
    generator = CycleGenerator(g1, g2)
    c1 = Lambda(lambda t: t.mean(axis=1))
    c2 = Lambda(lambda t: t.mean(axis=1)/2)
    critic = CycleCritic(c1, c2)
    cycle_loss = CycleGANLoss(GANLoss(g_loss_func,
                                      c_loss_func,
                                      GANModule(generator, critic)))
    fake_a2b = torch.Tensor([[0.5, 0.5], [0.1, 0.1]])
    fake_b2a = torch.Tensor([[0.6, 0.6], [0.4, 0.4]])
    real_pred = torch.Tensor([0.8, 0.9, 0.8, 0.7])
    in_a = torch.Tensor([[0.1, 0.2], [0.2, 0.2]])
    in_b = torch.Tensor([[1., 1], [0.4, 0.4]])
    
    actual_g_loss = cycle_loss.generator((fake_a2b, fake_b2a), torch.rand(2, 2), torch.rand(2, 2))    
    actual_c_loss = cycle_loss.critic(real_pred, in_a, in_b)
     
    # expected_fake_preds = ([0.5, 0.1, 0.3, 0.2])
    expected_g_loss =       (0.5 + 0.9 + 0.7 + 0.8) / 4
    # expected_g_out = ([[0.2, 0.4], [0.4, 0.4]], [[0.5, 0.5], [0.2, 0.2]])
    # expected_fake_preds = ([0.3, 0.4, 0.25, 0.1])
    expected_c_loss = (1.05 / 4 + 0.2) / 2
    
    assert math.isclose(actual_g_loss, expected_g_loss, abs_tol=1e-5)
    assert math.isclose(actual_c_loss, expected_c_loss, abs_tol=1e-5)


def test_cycle_consistency_loss():
    g_a2b = Lambda(lambda t: t*2)
    g_b2a = Lambda(lambda t: t/4)
    in_a = torch.Tensor([[0.5, 0.4], [0.3, 0.3]])
    in_b = torch.Tensor([[1., 1.], [0.8, 0.6]])
    out_b = g_a2b(in_a)
    out_a = g_b2a(in_b)
    cycle_cons_loss = CycleConsistencyLoss(g_a2b, g_b2a)
    
    actual_loss = cycle_cons_loss(in_a, in_b, out_b, out_a)
    # expected in_a_rec = [[0.25, 0.2], [0.15, 0.15]]
    # expected in_b_rec = [[0.5, 0.5], [0.4, 0.3]]
    expected_loss = (0.75 + 1.7) / 4
    
    assert math.isclose(actual_loss, expected_loss, abs_tol=1e-5)
    

def test_identity_loss():
    g_a2b = Lambda(lambda t: t*2)
    g_b2a = Lambda(lambda t: t/4)
    id_loss = IdentityLoss(g_a2b, g_b2a)
    in_a = torch.Tensor([[1, 0.6], [0.5, 0.3]])
    in_b = torch.Tensor([[0., 1.], [0.5, 0.5]])
    
    actual_loss = id_loss(in_a, in_b)
    expected_loss = (0.75 + 0.45 + 0.375 + 0.225 + 0 + 1 + 0.5 + 0.5) / 4
    
    assert math.isclose(actual_loss, expected_loss, abs_tol=1e-5)


class FakeLossArgsInterceptor:
    def __init__(self):
        self.call_args = []
        self.call_kwargs = []

    def __call__(self, *args, **kwargs):
        self.call_args.append(args)
        self.call_kwargs.append(kwargs)


def test_loss_wrapper():
    loss_call_args, loss_call_kwargs = [], []
    def _fake_loss(a, b, c=2):
        loss_call_args.append((a, b))
        loss_call_kwargs.append(c)
        return a + b * c
    interceptors = [FakeLossArgsInterceptor(), FakeLossArgsInterceptor()]
    loss_wrapper = LossWrapper(_fake_loss, loss_args_interceptors=interceptors)

    args1, kwargs1 = (1, 3), {}
    args2, kwargs2 = (-2, 2), {'c': 10}
    l1 = loss_wrapper(*args1, **kwargs1)
    l2 = loss_wrapper(*args2, **kwargs2)

    assert loss_call_args == [args1, args2]
    assert loss_call_kwargs == [2, kwargs2['c']]
    assert [interceptor.call_args == [args1, args2] for interceptor in interceptors]
    assert [interceptor.call_kwargs == [kwargs1, kwargs2] for interceptor in interceptors]
    assert l1 == _fake_loss(*args1, **kwargs1)
    assert l2 == _fake_loss(*args2, **kwargs2)


def test_crit_preds_tracker():
    cpt = CritPredsTracker()
    cpt_means = CritPredsTracker(reduce_batch=True)
    real_preds1 = torch.Tensor([0.5, 0.75, 1])
    fake_preds1 = torch.Tensor([0.5, 0.25, 0])
    real_preds2 = torch.Tensor([1., 1, 1])
    fake_preds2 = torch.Tensor([0.3, 0.2, 0.1])
    device = real_preds1.device
    
    cpt(real_preds1, fake_preds1)
    cpt(real_preds2, fake_preds2)
    cpt_means(real_preds1, fake_preds1)
    cpt_means(real_preds2, fake_preds2)
    
    cpt_copy = CritPredsTracker()
    cpt_copy.load_from_df(cpt.to_df(), device)
    
    assert torch.equal(cpt.real_preds, torch.Tensor([0.5, 0.75, 1, 1, 1, 1]))
    assert torch.equal(cpt.fake_preds, torch.Tensor([0.5, 0.25, 0, 0.3, 0.2, 0.1]))
    assert torch.allclose(cpt_means.real_preds, torch.Tensor([0.75, 1]))
    assert torch.allclose(cpt_means.fake_preds, torch.Tensor([0.25, 0.2]))
    assert torch.equal(cpt.real_preds, cpt_copy.real_preds)
    assert torch.equal(cpt.fake_preds, cpt_copy.fake_preds)
