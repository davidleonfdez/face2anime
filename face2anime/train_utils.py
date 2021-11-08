from abc import ABC, abstractmethod
from fastai.vision.all import *
import gc
import torch
from typing import Callable, List


__all__ = ['EMAAverager', 'EMACallback', 'add_ema_to_gan_learner', 'custom_save_model',
           'custom_load_model', 'EpochFilterAll', 'EpochFilterMultipleOfN', 
           'EpochFilterAfterN', 'ComposedEpochFilter', 'UpdateEMAPreSaveAction', 
           'SaveCheckpointsCallback', 'clean_mem']


class EMAAverager():
    """Callable class that calculates the EMA of a parameter.
    
    It can be used as the `avg_fn` parameter of `torch.optim.swa_utils.AveragedModel`
    Args:
        decay (float): weight of averaged value. The new value of the parameter is
          multiplied by 1 - decay.
    """
    def __init__(self, decay=0.999):
        self.decay = decay
        
    def __call__(self, averaged_model_parameter, model_parameter, num_averaged):
        return self.decay * averaged_model_parameter + (1 - self.decay) * model_parameter


def _default_forward_batch(model, batch, device):
    input = batch
    if isinstance(input, (list, tuple)):
        input = input[0]
    if device is not None:
        input = input.to(device)
    model(input)


class FullyAveragedModel(torch.optim.swa_utils.AveragedModel):
    """Extension of AveragedModel that also averages the buffers.
    
    To update both the parameters and the buffers, the method `update_all` should be 
    called instead of `update_parameters`."""
    def _update_buffers(self, model):
        for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
            device = b_swa.device
            b_model_ = b_model.detach().to(device)
            if self.n_averaged == 0:
                b_swa.detach().copy_(b_model_)
            else:
                b_swa.detach().copy_(self.avg_fn(b_swa.detach(), b_model_,
                                                 self.n_averaged.to(device)))                

    def update_all(self, model):
        # Buffers must be updated first, because this method relies on n_averaged,
        # which is updated by super().update_parameters()
        self._update_buffers(model)
        self.update_parameters(model)


@torch.no_grad()
def _update_bn(loader, model, device=None, forward_batch:Callable=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
        forward_batch: method that chooses how to extract the input from every 
            element of :attr:`loader`, transfers it to :attr:`device` and
            finally makes a forward pass on :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> _update_bn(loader, model)
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    if forward_batch is None: forward_batch = _default_forward_batch
    for batch in loader:
        forward_batch(model, batch, device)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class EMACallback(Callback):
    """Updates the averaged weights of the generator of a GAN after every opt step.

    It's meant to be used only with a GANLearner; i.e., an instance of this callback
    is assumed to be attached to a GANLearner.
    Args:
        ema_model: AveragedModel that wraps the averaged generator module.
        orig_model: active (not averaged) generator module, the one that's included
            in learner.model and updated by the optimizer.
        dl: dataloader needed to iterate over all data and make forward passes over the 
            ema_model in order to update the running statistic of BN layers.
        update_buffers: if True, not only parameters, but also buffers, of ema_model are 
            averaged and updated, 
        forward_batch (Callable): Method with params (model, batch, device) that chooses 
            how to extract the input from every element of `dl`, transfers it to the proper 
            device and finally makes a forward pass on the model (here `ema_model`). 
            It's needed for updating the running statistics of BN layers.
    """
    def __init__(self, ema_model:FullyAveragedModel, orig_model:nn.Module, dl, 
                 update_buffers=True, forward_batch=None): 
        self.ema_model = ema_model
        self.orig_model = orig_model
        self.dl = dl
        self.update_buffers = update_buffers
        self.update_bn_pending = False
        self.forward_batch = forward_batch
        
    def after_step(self):
        if self.gan_trainer.gen_mode:
            update_method = (self.ema_model.update_all if self.update_buffers
                             else self.ema_model.update_parameters)
            update_method(self.orig_model)
            self.update_bn_pending = True
            
    def after_fit(self):
        self.update_bn()

    def update_bn(self):
        if not self.update_bn_pending: return
        #torch.optim.swa_utils.update_bn(self.dl, self.ema_model)
        _update_bn(self.dl, self.ema_model, forward_batch=self.forward_batch)
        self.update_bn_pending = False        


def add_ema_to_gan_learner(gan_learner, dblock, decay=0.999, update_bn_dl_bs=64,
                           forward_batch=None):
    """"Creates and setups everything needed to update an alternative EMA generator.

    It stores the EMA generator in `ema_model` attribute of `gan_learner`.
    Args:
        gan_learner (GANLearner): the learner to add the EMA generator to.
        dblock (DataBlock): needed to create dataloaders that are independent of those 
            of `gan_learner`, used after fit to update BN running stats of the EMA G.
        decay: weight that multiplies averaged parameter every update.
        update_bn_dl_bs: batch size used to update BN running stats.
        forward_batch (Callable): Method with params (model, batch, device) that chooses 
            how to extract the input from every element of the dataloader, transfers it 
            to the proper device and finally makes a forward pass on the ema model. 
            It's needed for updating the running statistics of BN layers.
    """
    generator = gan_learner.model.generator
    ema_avg_fn = EMAAverager(decay=decay)
    gan_learner.ema_model = FullyAveragedModel(generator, avg_fn=ema_avg_fn)
    ds_path = gan_learner.dls.path
    clean_dls = dblock.dataloaders(ds_path, path=ds_path, bs=update_bn_dl_bs)
    gan_learner.ema_model.eval().to(clean_dls.device)
    gan_learner.add_cb(EMACallback(gan_learner.ema_model, generator, clean_dls.train, 
                                   forward_batch=forward_batch))


def custom_save_model(learner, filename, base_path='.'):
    """Saves the model and optimizer state of the learner.
    
    The path of the generated file is base_path/learner.model_dir/filename
    with ".pth" extension. If the learner has an EMA G model attached too,
    a similar file with the suffix "_ema" is generated too.
    """
    if isinstance(base_path, str): base_path = Path(base_path)
    if not isinstance(base_path, Path): raise Exception('Invalid base_path')
    file = join_path_file(filename, base_path/learner.model_dir, ext='.pth')
    save_model(file, learner.model, learner.opt)
    if getattr(learner, 'ema_model', None) is not None:
        _save_ema_model(learner, base_path, filename)
    

def custom_load_model(learner, filename, with_opt=True, device=None, 
                      base_path='./models', 
                      with_ema=False, **kwargs):
    """Loads the model and optimizer state of the learner.
    
    The file is expected to be placed in `base_path/filename` with ".pth"
    extension. `kwargs` are forwarded to fastai's `load_model` method.
    """
    if isinstance(base_path, str): base_path = Path(base_path)
    if not isinstance(base_path, Path): raise Exception('Invalid base_path')
    if device is None and hasattr(learner.dls, 'device'): device = learner.dls.device
    if learner.opt is None: learner.create_opt()
    #file = join_path_file(filename, base_path/learner.model_dir, ext='.pth')
    file = base_path/f'{filename}.pth'
    load_model(file, learner.model, learner.opt, with_opt=with_opt, device=device, **kwargs)
    if with_ema:
        _load_ema_model(learner, base_path, filename)

    
def _load_ema_model(learner, base_path, filename, device=None):
    ema_filename = base_path/f'{filename}_ema.pth'
    load_model(ema_filename, learner.ema_model, None, with_opt=False, device=device)
    #state_dict = torch.load(ema_filename)
    #learner.ema_model.load_state_dict(state_dict)


def _save_ema_model(learner, base_path, filename):
    file = join_path_file(filename+'_ema', base_path/learner.model_dir, ext='.pth')
    save_model(file, learner.ema_model, None, with_opt=False)
    #torch.save(file, learner.ema_model.state_dict())    


class EpochFilter(ABC):
    "A child class must define a criteria to filter epochs depending on its index"
    @abstractmethod
    def contains(self, epoch:int):
        pass


class EpochFilterAll(EpochFilter):
    def contains(self, epoch: int):
        return True


class EpochFilterMultipleOfN(EpochFilter):
    def __init__(self, n):
        self.n = n

    def contains(self, epoch: int):
        return (epoch % self.n) == 0


class EpochFilterAfterN(EpochFilter):
    def __init__(self, n):
        self.n = n

    def contains(self, epoch: int):
        return epoch > self.n


class ComposedEpochFilter(EpochFilter):
    def __init__(self, filters:List[EpochFilter]):
        self.filters = filters

    def contains(self, epoch: int):
        return all(filter.contains(epoch) for filter in self.filters)


class PreSaveAction(ABC):
    "Child classes must implement a piece of code to be executed before saving a model at a given epoch."
    @abstractmethod
    def before_save(self, epoch):
        pass


class UpdateEMAPreSaveAction(PreSaveAction):
    def __init__(self, learn, epoch_filter=None):
        self.ema_cb = first(learn.cbs, f=lambda cb: isinstance(cb, EMACallback))
        self.epoch_filter = epoch_filter if epoch_filter is not None else EpochFilterAll()
    
    def before_save(self, epoch):
        if self.ema_cb is None: return
        if self.epoch_filter.contains(epoch):
            self.ema_cb.update_bn()


class SaveCheckpointsCallback(Callback):
    "Callback that saves the model at the end of each epoch."
    def __init__(self, fn_prefix:str, base_path=Path('.'), initial_epoch=1,
                 save_cycle_len=1, pre_save_actions:List[PreSaveAction]=None):
        self.fn_prefix = fn_prefix
        self.base_path = base_path
        self.epoch = initial_epoch
        self.save_cycle_len = save_cycle_len
        self.pre_save_actions = pre_save_actions if pre_save_actions is not None else []
        
    def after_epoch(self):
        if (self.epoch % self.save_cycle_len) == 0:
            for action in self.pre_save_actions: action.before_save(self.epoch)
            fn = f'{self.fn_prefix}_{self.epoch}ep'
            custom_save_model(self.learn, fn, base_path=self.base_path)
        self.epoch += 1


def clean_mem():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
