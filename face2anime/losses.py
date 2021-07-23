from face2anime.misc import FeaturesCalculator
from face2anime.networks import Img2ImgGenerator
from fastai.vision.all import *
from fastai.vision.gan import *
import pandas as pd
import torch
from typing import Callable, List, Tuple, Union


__all__ = ['random_epsilon_gp_sampler', 'GANGPCallback', 'R1GANGPCallback', 
           'repelling_reg_term', 'RepellingRegCallback', 'ContentLossCallback',
           'CycleGANLoss', 'CycleConsistencyLoss', 'CycleConsistencyLossCallback', 
           'IdentityLoss', 'IdentityLossCallback', 'CrossIdentityLoss', 
           'CrossIdentityLossCallback', 'LossWrapper', 'CritPredsTracker',
           'MultiCritPredsTracker']


def random_epsilon_gp_sampler(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    "Creates a tensor containing a batch of random epsilon values to be used by WGAN-GP loss."
    # A different random value of epsilon for any element of a batch
    epsilon_vec = torch.rand(real.shape[0], 1, 1, 1, dtype=torch.float, device=real.device, requires_grad=False)
    return epsilon_vec.expand_as(real)


class GANGPCallback(Callback):
    """Adds a GP to the loss of the critic of the GANLearner this callback is attached to.
    
    It implements the original gradient penalty as defined in "Improved Training of 
    Wasserstein GANs" (https://arxiv.org/abs/1704.00028).
    Args:
        plambda (float): weight of the GP
        epsilon_sampler (Callable): must return a tensor containing a batch of random epsilon 
            values. Epsilon is the weight that multiplies a real in the interpolation between 
            a real and a fake image. By default (if None), random_epsilon_gp_sampler is taken.
        center_val: value of the norm of the gradients that makes the GP minimum (zero).
    """
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
    """Adds a R1 GP to the loss of critic of the GANLearner this callback is attached to.
    
    It implements the R1 gradient penalty defined in "Which Training Methods for GANs do 
    actually Converge?" (https://arxiv.org/abs/1801.04406).
    Args
        weight: float that multiplies the gradient penalty before adding it to the loss.
            When there's a single real target (critic only receives one input tensor), it
            must always be a float.
            When there are multiple real targets (critic receives a tuple of tensors):
                -If weight is a tuple, it should have a length equal to the length of the
                 input of the critic.
                -If weight is a float, the same weight will be used to multiply the 
                 norm of the gradients of the preds w.r.t. every input.
        critic: module used to obtain the obtain the preds later used to calculate the
            gradients w.r.t. the input[s].
    """
    def __init__(self, weight:Union[float, Tuple[float, ...]]=10., critic:nn.Module=None): 
        self.weight = weight
        self.critic = critic
        
    def _gradient_penalty(self, real, weight):
        if not isinstance(real, (list, tuple)):
            # Single target case
            # Wrap with a list to handle it the same way as multiple targets case
            real = [real]
        if not isinstance(weight, (list, tuple)): weight = [weight]
        x = [real_t.detach().requires_grad_(True) for real_t in real]
        critic = self.critic or self.model.critic
        preds = critic(*x).mean()
        grads_by_input = torch.autograd.grad(outputs=preds, inputs=x, create_graph=True)
        
        # If there are multiple real targets (len(x) > 1) and only 1 weight is given, we'll
        # use the same weight for every input
        if len(x) > len(weight): weight = weight * len(x)
        result = None
        
        for grads, wi in zip (grads_by_input, weight):
            # gp = wi * (grads.norm()**2)  
            # (flat+dot product) seems more efficient than norm**2
            flat_grads = grads.view(-1)
            gp = wi * flat_grads.dot(flat_grads)
            result = gp if result is None else result + gp
            
        return result
        
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
    def __init__(self, module:nn.Module, weight=1.):
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
    "Adds a content loss term to the G loss of the GANLearner this callback is attached to."
    def __init__(self, weight=1., ftrs_calc:FeaturesCalculator=None, device=None, 
                 content_loss_func:Callable=None):
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
            if self.training: self.learn.loss_grad += loss_val


class CycleGANLoss(GANModule):
    "Wrapper around `GANLoss` that handles inputs, targets and G outputs containing two items."
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
    def __init__(self, g_a2b:nn.Module, g_b2a:nn.Module, loss_func:Callable=None):
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
    "Adds a cycle consistency loss term to the G loss of the GANLearner this cb is attached to."
    def __init__(self, g_a2b:nn.Module, g_b2a:nn.Module, loss_func:Callable=None, 
                 weight=1.):
        self.loss = CycleConsistencyLoss(g_a2b, g_b2a, loss_func=loss_func)
        self.weight = weight
    
    def after_loss(self):
        if self.gan_trainer.gen_mode:
            in_a, in_b = self.x
            out_b, out_a = self.pred
            loss_val = self.weight * self.loss(in_a, in_b, out_b, out_a)
            self.learn.loss_func.cycle_loss = loss_val
            if self.training: self.learn.loss_grad += loss_val


class IdentityLoss:
    def __init__(self, g_a2b:nn.Module, g_b2a:nn.Module, loss_func:Callable=None):
        self.g_a2b = g_a2b
        self.g_b2a = g_b2a
        self.loss_func = nn.L1Loss() if loss_func is None else loss_func
        
    def __call__(self, in_a, in_b):
        id_a = self.g_b2a(in_a)
        id_b = self.g_a2b(in_b)
        return self.loss_func(in_a, id_a) + self.loss_func(in_b, id_b)

 
class IdentityLossCallback(Callback):
    """Adds an identity loss term to the G loss of the GANLearner this callback is attached to.
    
    It incentivizes every generator to produce and output equal to the input when the input
    belongs to its target domain. For example, the generator A->B is incentivized to produce an 
    output equal to the input when it receives and input from domain B.
    """
    def __init__(self, g_a2b, g_b2a, loss_func=None, weight=1.):
        self.loss = IdentityLoss(g_a2b, g_b2a, loss_func=loss_func)
        self.weight = weight
    
    def after_loss(self):
        if self.gan_trainer.gen_mode:
            in_a, in_b = self.x
            loss_val = self.weight * self.loss(in_a, in_b)
            self.learn.loss_func.identity_loss = loss_val
            if self.training: self.learn.loss_grad += loss_val


class CrossIdentityLoss:
    def __init__(self, encoder_a:nn.Module, encoder_b:nn.Module, decoder_a:nn.Module, 
                 decoder_b:nn.Module, loss_func:Callable=None):
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.decoder_a = decoder_a
        self.decoder_b = decoder_b
        self.loss_func = nn.L1Loss() if loss_func is None else loss_func
        
    def __call__(self, in_a, in_b):
        id_a = self.decoder_a(self.encoder_a(in_a))
        id_b = self.decoder_b(self.encoder_b(in_b))
        return self.loss_func(in_a, id_a) + self.loss_func(in_b, id_b)


class CrossIdentityLossCallback(Callback):
    """Adds an identity loss term to the G loss of the GANLearner this callback is attached to.
    
    It incentivizes the combination of the encoder of one generator and the decoder of the
    other to produce an output equal to the input when it receives an input from the target
    domain of the generator that the decoder belongs to.
    For example, the generator A->B has an encoder that is expected to produce a latent
    representation l_i of an image a_i from domain A. If we assume the latent representation is 
    shared between generators and holds some domain independent information (usually called 
    "content"), we would expect the decoder of the generator B->A to be able to produce an 
    output from domain A which is equal to a_i when given l_i as input.
    """
    def __init__(self, g_a2b:Img2ImgGenerator, g_b2a:Img2ImgGenerator, loss_func=None, 
                 weight=1.):
        encoder_a = g_a2b.get_encoder()
        decoder_b = g_a2b.get_decoder()
        encoder_b = g_b2a.get_encoder()
        decoder_a = g_b2a.get_decoder()
        self.loss = CrossIdentityLoss(encoder_a, encoder_b, decoder_a, decoder_b, 
                                      loss_func=loss_func)
        self.weight = weight
    
    def after_loss(self):
        if self.gan_trainer.gen_mode:
            in_a, in_b = self.x
            # It could be optimized to pass latents too (and save half forward time)
            # using hooks, but it would require careful callbacks ordering, given that other
            # callback could forward G with non real images before this cb is executed.
            loss_val = self.weight * self.loss(in_a, in_b)
            self.learn.loss_func.cross_identity_loss = loss_val
            if self.training: self.learn.loss_grad += loss_val


class LossWrapper():
    "Callable class that wraps `orig_loss` and forwards its call args to `loss_args_interceptors`"
    def __init__(self, orig_loss, loss_args_interceptors:List[Callable]=None):
        self.orig_loss = orig_loss
        self.loss_args_interceptors = ifnone(loss_args_interceptors, [])

    def __call__(self, *args, **kwargs):
        for interceptor in self.loss_args_interceptors:
            interceptor(*args, **kwargs)
        return self.orig_loss(*args, **kwargs)


class CritPredsTracker():
    """Interceptor of critic loss (GANLoss.critic) that stores the history of real and fake predictions.
    
    It's meant to be used with `LossWrapper` as an item of `loss_args_interceptors`.
    Args:
        reduce_batch: if True, it only stores the mean for each batch of predictions."""
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
    """Interceptor of multiple critic loss (normally CycleGANLoss.critic) that stores the history of predictions.
    
    It stores separately the predictions for each critic and for reals and fakes.
    Args:
        reduce_batch: if True, it only stores the mean for each batch of predictions.
        group_preds: method that splits the predictions by critic. Must accept an input that 
            has exactly the same type and structure than the output of the global critic.
            By default, if None, it assumes the input is a tensor result of concatenating
            the predictions of each critic.
    """
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
