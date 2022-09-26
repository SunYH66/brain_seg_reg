# --coding:utf-8--
import torch
from model import network
from model.base_model import BaseModel
from monai.losses.dice import DiceLoss
from utils.loss import FocalLoss

class SegModel(BaseModel):
    """TODO: Build up IBIS segmentation model."""
    def __init__(self, opt):
        super(SegModel, self).__init__(opt)
        self.model_name = ['seg']
        self.optimizer_name = ['seg']
        self.loss_name = ['seg']
        self.visual_name = ['seg_output']

        # define networks
        self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.crop_size)


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


    def forward(self, opt):
        # implement segmentation
        self.seg_output = self.net_seg(self.img_24)


    def optimize_train_parameters(self, epoch, opt):
        # forward()
        self.forward(opt)

        # Calculate segmentation loss
        self.loss_seg = self.loss_seg_criterion_dice.forward(self.seg_output, self.seg_24) + \
                        10 * self.loss_seg_criterion_focal(self.seg_output, self.seg_24)

        self.seg_output = self.seg_output.argmax(1).unsqueeze(1)

        self.optimizer_seg.zero_grad()
        self.loss_seg.backward()
        self.optimizer_seg.step()


    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            # forward()
            self.forward(opt)

            # Calculate segmentation loss
            self.loss_seg = self.loss_seg_criterion_dice.forward(self.seg_output, self.seg_24) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, self.seg_24)

            self.seg_output = self.seg_output.argmax(1).unsqueeze(1)