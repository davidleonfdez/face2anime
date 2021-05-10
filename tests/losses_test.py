from face2anime.losses import CritPredsTracker, LossWrapper
import torch


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