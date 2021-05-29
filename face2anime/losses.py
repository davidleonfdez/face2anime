from face2anime.misc import FeaturesCalculator
from fastai.vision.all import *
import pandas as pd
import torch


__all__ = ['random_epsilon_gp_sampler', 'GANGPCallback', 'R1GANGPCallback', 
           'repelling_reg_term', 'RepellingRegCallback', 'ContentLossCallback',
           'LossWrapper', 'CritPredsTracker']


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
        x = real.detach().requires_grad_(True)
        critic = self.critic or self.model.critic
        preds = critic(x).mean()

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
    # Cosine similarity between ftrs of any batch
    # cos_sims[i, j] = cosine similarity between ftrs of batch `i` and ftrs of batch `j`
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
