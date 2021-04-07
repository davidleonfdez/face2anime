from fastai.vision.all import *
import torch


__all__ = ['random_epsilon_gp_sampler', 'GANGPCallback', 'R1GANGPCallback']


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
    def __init__(self, weight=10.): 
        self.weight = weight
        
    def _gradient_penalty(self, real, weight):
        x = real.detach().requires_grad_(True)
        preds = self.model.critic(x).mean()

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
