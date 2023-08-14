# --coding:utf-8--
import os
from brain_reg_seg.utils.utils import mkdirs
from torch.utils.tensorboard import SummaryWriter

class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.visual_dir = os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'visuals')
        mkdirs([self.visual_dir])
        self.writer = SummaryWriter(log_dir=self.visual_dir)
        self.writer.add_text('Experiment name:', text_string=cfg.name)
        # self.writer.add_graph(net, input_to_model=torch.rand(9, 1, 128, 128, 96))
        self.writer.flush()


    def display_current_results(self, visuals, count):
        for label, img in visuals.items():
            img_grid = img[0, 0, ..., 48] # display the medium slice of the results(seg and reg)
            self.writer.add_image(label, img_grid, dataformats='HW')
        self.writer.flush()


    def display_current_loss(self, losses, count, aff_tag=''):
        for label, loss in losses.items():
            self.writer.add_scalar(label + aff_tag, loss, count)
        self.writer.flush()


    def print_current_loss(self, losses, epoch):
        if self.cfg.loss_verbose:
            message = ''
            for k, v in losses.items():
                message += '%s: %.3f ' % (k, v)
            print('\n' + str(epoch) + ' ' + message)  # print the message

            # save the loss to log.txt
            cfg_dir = os.path.join(self.cfg.checkpoint_root, self.cfg.name,
                                   '{}'.format(self.cfg.reg_folder if self.cfg.model == 'reg' else self.cfg.seg_folder), 'log')

            cfg_name = os.path.join(cfg_dir, 'log.txt')
            with open(cfg_name, 'a') as c_file:
                c_file.write(str(epoch) + ' ' + message + '\n')