from face2anime.misc import FeaturesCalculator
from fastai.vision.all import *
from fastai.vision.gan import *
import pandas as pd
import torch
from typing import Callable


__all__ = ['random_epsilon_gp_sampler', 'GANGPCallback', 'R1GANGPCallback', 
           'repelling_reg_term', 'RepellingRegCallback', 'ContentLossCallback',
           'CycleGANLoss', 'CycleConsistencyLoss', 'CycleConsistencyLossCallback', 
           'IdentityLoss', 'IdentityLossCallback', 'LossWrapper', 'CritPredsTracker',
           'MultiCritPredsTracker']


def random_epsilon_gp_sampler(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    # A different random value of epsilon for any element of a batch
    epsilon_vec = torch.rand(real.shape[0], 1, 1, 1, dtype=torch.float, device=real.device, requires_grad=False)
    return epsilon_vec.expand_as(real)


class GANGPCallback(Callback):
    def __init__(self, plambda=10., epsilon_sampler=None, center_val=1): 
        self.plambda = plambda
        if epsilon_sampler is None: epsilon_sampler = random_epsilon_gp_sampler
        self.epsilon_sampler = epsilon_sampler
        self.center_val = center_val
        
    def _gradient_penalty(self, real, fake, plambda, epsilon_sampler):
        epsilon = epsilon_sampler(real, fake)
        x_hat = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
        x_hat_pred = self.model.critic(x_hat).mean()

        grads = torch.autograd.grad(outputs=x_hat_pred, inputs=x_hat, create_graph=True)[0]
        return plambda * ((grads.norm() - self.center_val)**2)    
        
    def after_loss(self):
        if not self.gan_trainer.gen_mode:
            # In critic mode, GANTrainer swaps x and y; so, here x is original y (real target)
            real = self.x
            assert not self.y.requires_grad
            # Cast to TensorImage to enable product compatibility
            fake = TensorImage(self.model.generator(self.y))
            # Updated to fastai version 2.2.7: backward isn't called on learn.loss anymore, 
            # but on learn.loss_grad
            self.learn.loss_grad += self._gradient_penalty(real, fake, self.plambda, self.epsilon_sampler)


class R1GANGPCallback(Callback):
    def __init__(self, weight=10., critic=None): 
        self.weight = weight
        self.critic = critic
        
    def _gradient_penalty(self, real, weight):
        if not isinstance(real, (list, tuple)):
            # Single target case
            # Wrap with a list to handle it the same way as multiple targets case
            real = [real]
        x = [real_t.detach().requires_grad_(True) for real_t in real]
        critic = self.critic or self.model.critic
        preds = critic(*x).mean()

        grads = torch.autograd.grad(outputs=preds, inputs=x, create_graph=True)[0]
        #return weight * (grads.norm()**2)  
        # (flat+dot product) seems more efficient than norm**2
        flat_grads = grads.view(-1)
        return weight * flat_grads.dot(flat_grads)
        
    def after_loss(self):
        if not self.gan_trainer.gen_mode:
            # In critic mode, GANTrainer swaps x and y; so, here x is original y (real target)
            real = self.x
            # Updated to fastai version 2.2.7: backward isn't called on learn.loss anymore, 
            # but on learn.loss_grad
            self.learn.loss_grad += self._gradient_penalty(real, self.weight)


def repelling_reg_term(ftr_map, weight):
    bs = ftr_map.shape[0]
    flat_ftrs = ftr_map.view(bs, -1)
    norms = flat_ftrs.norm(dim=1).unsqueeze(1)
    # Cosine similarity between ftrs of any element of batch
    # cos_sims[i, j] = cosine similarity between ftrs of item `i` and item `j` of current batch
    cos_sims = torch.mm(flat_ftrs, flat_ftrs.t()) / torch.mm(norms, norms.t())
    # Substract bs to discard the diagonal, which is full of 1's
    return weight * (cos_sims.square().sum() - bs) / (bs * (bs - 1))


class RepellingRegCallback(Callback):
    "Increases the G loss every iteration with a repelling regularization term."
    def __init__(self, module, weight=1.):
        self.hook = hook_output(module, detach=False)
        self.weight = weight
        self.history = []
    
    def after_loss(self):
        if not self.training: return
        if self.gan_trainer.gen_mode:
            ftr_map = self.hook.stored
            assert ftr_map.requires_grad
            reg_term = repelling_reg_term(ftr_map, self.weight)
            self.history.append(reg_term.detach().cpu())
            self.learn.loss_grad += reg_term


class ContentLossCallback(Callback):
    def __init__(self, weight=1., ftrs_calc=None, device=None, content_loss_func=None):
        self.weight = weight
        self.content_loss_func = (nn.MSELoss(reduction='mean') if content_loss_func is None
                                  else content_loss_func)
        if ftrs_calc is None:
            vgg_content_layers_idx = [22]
            ftrs_calc = FeaturesCalculator([], vgg_content_layers_idx, device=device)
        self.ftrs_calc = ftrs_calc
        
    def after_loss(self):
        if self.gan_trainer.gen_mode:
            input_content_ftrs = self.ftrs_calc.calc_content(self.x)[0]
            output_content_ftrs = self.ftrs_calc.calc_content(self.pred)[0]
            loss_val = self.weight * self.content_loss_func(output_content_ftrs, input_content_ftrs)
            # Store result inside learn.loss_func to make it visible to metrics display
            self.learn.loss_func.content_loss = loss_val
            self.learn.loss_grad += loss_val


class CycleGANLoss(GANModule):
    "Wrapper around GANLoss that handles inputs, target and G outputs containing two items."
    def __init__(self, gan_loss:GANLoss):
        super().__init__()
        self.gen_loss_func = gan_loss.gen_loss_func
        self.crit_loss_func = gan_loss.crit_loss_func
        self.gan_model = gan_loss.gan_model
        
    def generator(self, output, target_b, target_a):
        "Evaluate `out_b, out_a` with the critic then uses `self.gen_loss_func`"
        out_b, out_a = output
        fake_pred = self.gan_model.critic(out_b, out_a)
        self.gen_loss = self.gen_loss_func(fake_pred, output, (target_b, target_a))
        return self.gen_loss

    def critic(self, real_pred, in_a, in_b):
        "Create some `fake_pred` with the generator from `in_a, in_b` and compare them to `real_pred` in `self.crit_loss_func`."
        fake_b, fake_a = self.gan_model.generator(in_a, in_b)
        fake_b.requires_grad_(False)
        fake_a.requires_grad_(False)
        fake_pred = self.gan_model.critic(fake_b, fake_a)
        self.crit_loss = self.crit_loss_func(real_pred, fake_pred)
        return self.crit_loss


class CycleConsistencyLoss:
    def __init__(self, g_a2b, g_b2a, loss_func=None):
        self.g_a2b = g_a2b
        self.g_b2a = g_b2a
        self.loss_func = nn.L1Loss() if loss_func is None else loss_func
    
    def __call__(self, in_a, in_b, out_b, out_a):
        in_a_rec = self.g_b2a(out_b)
        in_b_rec = self.g_a2b(out_a)
        in_a_rec_loss = self.loss_func(in_a, in_a_rec)
        in_b_rec_loss = self.loss_func(in_b, in_b_rec)
        return in_a_rec_loss + in_b_rec_loss
        

class CycleConsistencyLossCallback(Callback):
    def __init__(self, g_a2b, g_b2a, loss_func=None, weight=1.):
        self.loss = CycleConsistencyLoss(g_a2b, g_b2a, loss_func=loss_func)
        self.weight = weight
    
    def after_loss(self):
        if self.gan_trainer.gen_mode:
            in_a, in_b = self.x
            out_b, out_a = self.pred
            loss_val = self.weight * self.loss(in_a, in_b, out_b, out_a)
            self.learn.loss_func.cycle_loss = loss_val
            self.learn.loss_grad += loss_val


class IdentityLoss:
    def __init__(self, g_a2b, g_b2a, loss_func=None):
        self.g_a2b = g_a2b
        self.g_b2a = g_b2a
        self.loss_func = nn.L1Loss() if loss_func is None else loss_func
        
    def __call__(self, in_a, in_b):
        id_a = self.g_b2a(in_a)
        id_b = self.g_a2b(in_b)
        return self.loss_func(in_a, id_a) + self.loss_func(in_b, id_b)

    
class IdentityLossCallback(Callback):
    def __init__(self, g_a2b, g_b2a, loss_func=None, weight=1.):
        self.loss = IdentityLoss(g_a2b, g_b2a, loss_func=loss_func)
        self.weight = weight
    
    def after_loss(self):
        if self.gan_trainer.gen_mode:
            in_a, in_b = self.x
            loss_val = self.weight * self.loss(in_a, in_b)
            self.learn.loss_func.identity_loss = loss_val
            self.learn.loss_grad += loss_val


class LossWrapper():
    def __init__(self, orig_loss, loss_args_interceptors=None):
        self.orig_loss = orig_loss
        self.loss_args_interceptors = loss_args_interceptors or []

    def __call__(self, *args, **kwargs):
        for interceptor in self.loss_args_interceptors:
            interceptor(*args, **kwargs)
        return self.orig_loss(*args, **kwargs)


class CritPredsTracker():
    def __init__(self, reduce_batch=False):
        self.real_preds = None
        self.fake_preds = None
        self.reduce_batch = reduce_batch

    def __call__(self, real_pred, fake_pred):
        new_real_pred = (real_pred.mean(dim=0, keepdim=True) if self.reduce_batch else real_pred).detach()
        new_fake_pred = (fake_pred.mean(dim=0, keepdim=True) if self.reduce_batch else fake_pred).detach()
        self.real_preds = (new_real_pred.clone() if self.real_preds is None 
                           else torch.cat([self.real_preds, new_real_pred]))
        self.fake_preds = (new_fake_pred.clone() if self.fake_preds is None 
                           else torch.cat([self.fake_preds, new_fake_pred]))
        
    def load_from_df(self, df, device):
        self.real_preds = torch.Tensor(df['RealPreds']).to(device)
        self.fake_preds = torch.Tensor(df['FakePreds']).to(device)
        
    def to_df(self):
        return pd.DataFrame(dict(RealPreds=self.real_preds.cpu(),
                                 FakePreds=self.fake_preds.cpu()))
    
    def reset(self):
        self.real_preds = None
        self.fake_preds = None


class MultiCritPredsTracker():
    def __init__(self, n=2, reduce_batch=False, group_preds:Callable=None):
        assert n >= 2
        self.trackers = [CritPredsTracker(reduce_batch=reduce_batch) for _ in range(n)]
        self.group_preds = ifnone(group_preds, lambda t: torch.chunk(t, n))

    def __call__(self, real_pred, fake_pred):
        real_pred_groups = self.group_preds(real_pred) 
        fake_pred_groups = self.group_preds(fake_pred)

        for real_pred, fake_pred, tracker in zip(real_pred_groups, fake_pred_groups, self.trackers):
            tracker(real_pred, fake_pred)
        
    def load_from_dfs(self, dfs, device):
        for df, tracker in zip (dfs, self.trackers):
            tracker.load_from_df(df, device)
        
    def to_dfs(self):
        return [tracker.to_df() for tracker in self.trackers]
    
    def reset(self):
        for tracker in self.trackers: tracker.reset()
