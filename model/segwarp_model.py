# --coding:utf-8--
import torch
from brain_reg_seg.model import network
from brain_reg_seg.model.base_model import BaseModel
from monai.losses.dice import DiceLoss
from brain_reg_seg.utils.loss import FocalLoss

class SegWarpModel(BaseModel):
    """TODO: Build up IBIS segmentation model."""
    def __init__(self, opt):
        super(SegWarpModel, self).__init__(opt)
        self.model_name = ['seg']
        self.optimizer_name = ['seg']
        self.loss_name = ['seg']
        self.visual_name = ['seg_output']
        self.path_name = ['img_path']

        # define networks
        self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.seg_train_mode, opt.crop_size)


        # define loss
        self.loss_seg_criterion_dice = DiceLoss(to_onehot_y=True)
        self.loss_seg_criterion_focal = FocalLoss(class_num=4)


        # define optimizer
        self.optimizer_seg = torch.optim.Adam(self.net_seg.parameters(), lr=opt.lr)


    def set_input(self, input_data):
        self.img = input_data['img'].to('cuda').float()
        self.seg = input_data['seg'].to('cuda').float()
        self.GT = self.seg
        self.img_path = input_data['img_path']


    def forward(self, opt):
        # implement segmentation
        self.seg_output = self.net_seg(self.img)


    def optimize_train_parameters(self, epoch, opt):
        # forward()
        self.forward(opt)

        # Calculate segmentation loss
        self.loss_seg = self.loss_seg_criterion_dice.forward(self.seg_output, self.seg) + \
                        10 * self.loss_seg_criterion_focal(self.seg_output, self.seg)

        self.seg_output = self.seg_output.argmax(1).unsqueeze(1)

        self.optimizer_seg.zero_grad()
        self.loss_seg.backward()
        self.optimizer_seg.step()


    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            # forward()
            self.forward(opt)

            # Calculate segmentation loss
            self.loss_seg = self.loss_seg_criterion_dice.forward(self.seg_output, self.seg) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, self.seg)

            self.seg_output = self.seg_output.argmax(1).unsqueeze(1)