# --coding:utf-8--
import os

from utils.utils import mkdirs
from torch.utils.tensorboard import SummaryWriter

class Visualizer:
    def __init__(self, cfg):
        self.visual_dir = os.path.join(cfg.checkpoint_root, cfg.name, 'visuals')
        mkdirs([self.visual_dir])
        self.writer = SummaryWriter(log_dir=self.visual_dir)
        self.writer.add_text('Experiment name:', text_string=cfg.name)
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