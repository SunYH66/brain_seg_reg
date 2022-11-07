# --coding:utf-8--
import torch
import random
from model import network
from model.base_model import BaseModel
from data import mask_sample, random_sample
from utils.loss import FocalLoss, ThreeDiceLoss, gradient_loss, to_one_hot
from monai.losses.dice import DiceLoss


class ProjModel(BaseModel):
    """TODO: Build up IBIS segmentation and registration model."""
    def __init__(self, opt):
        super(ProjModel, self).__init__(opt)
        self.model_name = ['seg']
        self.optimizer_name = ['seg']
        self.loss_name = ['seg_1', 'seg_2', 'seg_3', 'total']
        self.visual_name = ['seg_output']
        self.path_name = ['img_06_path', 'img_12_path', 'img_24_path']

        # define networks
        if opt.phase == 'train' or opt.phase == 'val':
            self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.crop_size)
            self.net_reg = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.ori_size)
        elif opt.phase == 'test':
            self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.ori_size)
            self.net_reg = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.ori_size)

        # define Spatial Transformer (spa_tra)
        self.spa_tra = network.SpatialTransformer(opt.ori_size)
        self.spa_tra.to('cuda')

        # define loss
        self.loss_seg_criterion_dice = DiceLoss(to_onehot_y=True)
        self.loss_seg_criterion_focal = FocalLoss(class_num=4)

        # define optimizer
        self.optimizer_seg = torch.optim.Adam(self.net_seg.parameters(), lr=opt.lr)

    def set_input(self, input_data):
        self.img_06 = input_data['img_06'].to('cuda').float()
        self.img_12 = input_data['img_12'].to('cuda').float()
        self.img_24 = input_data['img_24'].to('cuda').float()
        self.seg_06 = input_data['seg_06'].to('cuda').float()
        self.seg_12 = input_data['seg_12'].to('cuda').float()
        self.seg_24 = input_data['seg_24'].to('cuda').float()
        self.img_06_path = input_data['img_06_path']
        self.img_12_path = input_data['img_12_path']
        self.img_24_path = input_data['img_24_path']
        self.warped_ori_12_06 = input_data['warped_ori_12_06'].to('cuda').float()
        self.warped_ori_24_06 = input_data['warped_ori_24_06'].to('cuda').float()
        self.warped_ori_06_12 = input_data['warped_ori_06_12'].to('cuda').float()
        self.warped_ori_24_12 = input_data['warped_ori_24_12'].to('cuda').float()
        self.warped_ori_06_24 = input_data['warped_ori_06_24'].to('cuda').float()
        self.warped_ori_12_24 = input_data['warped_ori_12_24'].to('cuda').float()

    def forward(self, opt):

        # implement segmentation
        self.seg_output = self.net_seg(torch.cat((self.img_06, self.img_12, self.img_24), dim=0))
        self.seg_warped_12_06 = self.net_seg(self.warped_ori_12_06)
        self.seg_warped_24_06 = self.net_seg(self.warped_ori_24_06)
        self.seg_warped_06_12 = self.net_seg(self.warped_ori_06_12)
        self.seg_warped_24_12 = self.net_seg(self.warped_ori_24_12)
        self.seg_warped_06_24 = self.net_seg(self.warped_ori_06_24)
        self.seg_warped_12_24 = self.net_seg(self.warped_ori_12_24)

    def optimize_train_parameters(self, count, opt):
        if count % opt.update_frequency == 0:
            # forward()
            self.forward(opt)

            # Calculate segmentation loss (target should be one-channel)
            target = torch.cat((self.seg_06, self.seg_12, self.seg_24), dim=0)
            self.loss_seg_1 = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                              10 * self.loss_seg_criterion_focal(self.seg_output, target)
            target = torch.cat((self.seg_warped_12_06.argmax(1).unsqueeze(1), self.seg_warped_06_12.argmax(1).unsqueeze(1), self.seg_warped_06_24.argmax(1).unsqueeze(1)), dim=0)
            self.loss_seg_2 = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, target)
            target = torch.cat((self.seg_warped_24_06.argmax(1).unsqueeze(1), self.seg_warped_24_12.argmax(1).unsqueeze(1), self.seg_warped_12_24.argmax(1).unsqueeze(1)), dim=0)
            self.loss_seg_3 = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, target)
            self.loss_total = self.loss_seg_1 + opt.trade_off_seg * self.loss_seg_2 + opt.trade_off_seg * self.loss_seg_3

            # Transfer results to one-channel results
            self.seg_output = self.seg_output.argmax(1).unsqueeze(1)

            # Backward
            self.optimizer_seg.zero_grad()
            self.loss_total.backward()
            self.optimizer_seg.step()


    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            # forward()
            self.forward(opt)

            # Calculate segmentation loss (target should be one-channel)
            target = torch.cat((self.seg_06, self.seg_12, self.seg_24), dim=0)
            self.loss_seg_1 = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                              10 * self.loss_seg_criterion_focal(self.seg_output, target)
            target = torch.cat((self.seg_warped_12_06.argmax(1).unsqueeze(1), self.seg_warped_06_12.argmax(1).unsqueeze(1), self.seg_warped_06_24.argmax(1).unsqueeze(1)), dim=0)
            self.loss_seg_2 = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, target)
            target = torch.cat((self.seg_warped_24_06.argmax(1).unsqueeze(1), self.seg_warped_24_12.argmax(1).unsqueeze(1), self.seg_warped_12_24.argmax(1).unsqueeze(1)), dim=0)
            self.loss_seg_3 = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, target)
            self.loss_total = self.loss_seg_1 + opt.trade_off_seg * self.loss_seg_2 + opt.trade_off_seg * self.loss_seg_3

            # Transfer results to one-channel results
            self.seg_output = self.seg_output.argmax(1).unsqueeze(1)
