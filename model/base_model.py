# --coding:utf-8--
import os
import torch
from collections import OrderedDict
from torch.optim import lr_scheduler

class BaseModel:
    """TODO: Base function for ProjModel."""
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda'
        self.img_ID = []
        self.idx_list = []
        self.model_name = []
        self.optimizer_name = []
        self.loss_name = []
        self.visual_name = []


    def setup_model(self, opt):
        if opt.phase == 'train' and opt.resume_epoch >= 0:
            self.load_model(opt.resume_epoch)
        elif opt.phase == 'test':
            self.load_model(opt.load_epoch)
        self.print_model()


    def print_model(self):
        print('-----------------------Networks initialized-----------------------')
        for name in self.model_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print('[Network %s] Total number of parameters: %.3f M' % (name, num_params / 1e6))
        print('------------------------------------------------------------------')


    def load_model(self, epoch):
        for name in self.model_name:
            if isinstance(name, str):
                load_filename = '{:0>3d}_net_{}.pth'.format(epoch, name)
                load_path = os.path.join(self.opt.checkpoint_root, self.opt.name, 'model', load_filename)
                model = getattr(self, 'net_' + name)
                if isinstance(model, torch.nn.DataParallel):
                    model = model.module
                model.to(self.device)
                model.load_state_dict(torch.load(load_path))


    def save_model(self, epoch, aff_tag=None):
        for name in self.model_name:
            if isinstance(name, str):
                if aff_tag is not None:
                    save_filename = '%s_net_%s.pth' % (str(epoch).join('_') + aff_tag, name)
                else:
                    save_filename = '%03d_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.opt.checkpoint_root, self.opt.name, 'model', save_filename)
                model = getattr(self, 'net_' + name)
                torch.save(model.module.state_dict(), save_path)


    def train(self):
        """Make models train mode during training time."""
        for name in self.model_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()


    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_name:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()


    @staticmethod
    def get_current_scheduler(optimizer, opt):
        """TODO: Return the current learning rate scheduler."""
        if opt.lr_policy == 'linear':
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.03, total_iters=opt.num_epoch)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.decay_epoch, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
        else:
            raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler


    def update_learning_rate(self, opt):
        """TODO: Update the learning rate after optimize training parameters."""
        self.schedulers = [self.get_current_scheduler(getattr(self, 'optimizer_' + name), opt) for name in self.optimizer_name]
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()


    def get_current_loss(self):
        """Return training losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_name:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name)) # float(...) works for both scalar tensor and float number
        return errors_ret


    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with tensorboard, and save these images to the disk"""
        visual_ret = OrderedDict()
        for name in self.visual_name:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret


    # def calculate_accuracy(self):
    #     """Print and save accuracy of the trained results"""
    #     pred = getattr(self, self.output_names + '_output').detach().cpu().numpy()
    #     label = getattr(self, 'image_label').detach().cpu().numpy()
    #     accuracy = accuracy_score(label, np.argmax(pred, axis=1), normalize=False)
    #     return accuracy


    def get_image_path(self):
        """Return image paths."""
        keys = list(self.__dict__.keys())
        img_path_dict = dict()
        for i in range(len(keys)):
            if 'path' in keys[i]:
                img_path_dict[keys[i]] = self.__dict__[keys[i]]
        return img_path_dict


    def get_image_ID(self):
        """Return image paths."""
        return self.img_ID


    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
