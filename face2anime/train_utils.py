from fastai.vision.all import *
import torch


__all__ = ['EMAAverager', 'EMACallback', 'add_ema_to_gan_learner', 'custom_save_model',
           'custom_load_model']


class EMAAverager():
    def __init__(self, decay=0.999):
        self.decay = decay
        
    def __call__(self, averaged_model_parameter, model_parameter, num_averaged):
        return self.decay * averaged_model_parameter + (1 - self.decay) * model_parameter
    

class EMACallback(Callback):
    def __init__(self, ema_model, orig_model, dl, update_buffers=True): 
        self.ema_model = ema_model
        self.orig_model = orig_model
        self.dl = dl
        self.update_buffers = update_buffers
        self.update_bn_pending = False
        
    def after_step(self):
        if self.gan_trainer.gen_mode:
            update_method = (self.ema_model.update_all if self.update_buffers
                             else self.ema_model.update_parameters)
            update_method(self.orig_model)
            self.update_bn_pending = True
            
    def after_fit(self):
        if not self.update_bn_pending: return
        torch.optim.swa_utils.update_bn(self.dl, self.ema_model)
        self.update_bn_pending = False


class FullyAveragedModel(torch.optim.swa_utils.AveragedModel):
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

        
def add_ema_to_gan_learner(gan_learner, dblock, ds_path, decay=0.999, update_bn_dl_bs=64):
    generator = gan_learner.model.generator
    ema_avg_fn = EMAAverager(decay=decay)
    gan_learner.ema_model = FullyAveragedModel(generator, avg_fn=ema_avg_fn)
    clean_dls = dblock.dataloaders(ds_path, path=ds_path, bs=update_bn_dl_bs)
    gan_learner.ema_model.eval().to(clean_dls.device)
    gan_learner.add_cb(EMACallback(gan_learner.ema_model, generator, clean_dls.train))


def custom_save_model(learner, filename, base_path='.'):
    if isinstance(base_path, str): base_path = Path(base_path)
    if not isinstance(base_path, Path): raise Exception('Invalid base_path')
    file = join_path_file(filename, base_path/learner.model_dir, ext='.pth')
    save_model(file, learner.model, learner.opt)
    if getattr(learner, 'ema_model', None) is not None:
        _save_ema_model(learner, base_path, filename)
    

def custom_load_model(learner, filename, with_opt=True, device=None, 
                      base_path='./models', 
                      with_ema=False, **kwargs):
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
