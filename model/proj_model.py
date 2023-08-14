# --coding:utf-8--
import torch
from brain_reg_seg.model import network
from brain_reg_seg.model.base_model import BaseModel
from brain_reg_seg.utils.loss import FocalLoss
from monai.losses.dice import DiceLoss
from monai.inferers import SlidingWindowInferer

class ProjModel(BaseModel):
    """TODO: Build up IBIS segmentation and registration model."""
    def __init__(self, opt):
        super(ProjModel, self).__init__(opt)
        self.model_name = ['seg']
        self.optimizer_name = ['seg']
        self.loss_name = ['seg_main', 'seg_consist1', 'seg_consist2', 'total'] #if opt.seg_train_mode != 'fusion_seg' else ['seg_main', 'seg_help1', 'seg_help2', 'seg_consist1', 'seg_consist2', 'total']
        self.visual_name = ['seg_output']
        self.path_name = ['img_path_main', 'warped_ori_path_help_1', 'warped_ori_path_help_2']

        # define networks
        # 06 month for feature fusion
        if '{:02d}mo'.format(self.mo_list[0]) == '06mo':
            if opt.phase == 'train' or opt.phase == 'val':
                self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, 'fusion_only', opt.crop_size, opt.seg_fusion)
            elif opt.phase == 'test':
                self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, 'fusion_only', opt.ori_size, opt.seg_fusion)
        else:
            if opt.phase == 'train' or opt.phase == 'val':
                self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, 'single', opt.crop_size)
            elif opt.phase == 'test':
                self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, 'single', opt.ori_size)

        # define loss
        self.loss_seg_criterion_dice = DiceLoss(to_onehot_y=True)
        self.loss_seg_criterion_focal = FocalLoss(class_num=4)

        # define optimizer
        self.optimizer_seg = torch.optim.Adam(self.net_seg.parameters(), lr=opt.lr)


    def set_input(self, input_data):
        self.img_main = input_data['img_main'].to('cuda').float()
        self.seg_main = input_data['seg_main'].to('cuda').float()
        self.GT = self.seg_main

        self.warped_ori_help_1 = input_data['warped_ori_help_1'].to('cuda').float()
        self.warped_seg_help_1 = input_data['warped_seg_help_1'].to('cuda').float()

        self.warped_ori_help_2 = input_data['warped_ori_help_2'].to('cuda').float()
        self.warped_seg_help_2 = input_data['warped_seg_help_2'].to('cuda').float()

        self.img_path_main = input_data['img_path_main']
        self.warped_ori_path_help_1 = input_data['warped_ori_path_help_1']
        self.warped_ori_path_help_2 = input_data['warped_ori_path_help_2']


    def forward_train(self, opt):
        # implement segmentation
        self.seg_output = self.net_seg(torch.cat((self.img_main, self.warped_ori_help_1, self.warped_ori_help_2), dim=1))
        # self.seg_output = torch.cat(([self.seg_output[i] for i in range(len(self.seg_output))]), dim=0)


    def forward_val(self, opt):
        # define sliding window inference object
        inferer = SlidingWindowInferer(opt.crop_size, overlap=0.25)
        self.seg_output = inferer(torch.cat((self.img_main, self.warped_ori_help_1, self.warped_ori_help_2), dim=1), self.net_seg)
        # self.seg_output = torch.cat(([self.seg_output[i] for i in range(len(self.seg_output))]), dim=0)


    def optimize_train_parameters(self, count, opt):
        # forward()
        self.forward_train(opt)

        # Calculate segmentation loss (target should be one-channel)
        self.loss_seg_main = self.loss_seg_criterion_dice.forward(self.seg_output, self.seg_main) + \
                             10 * self.loss_seg_criterion_focal(self.seg_output, self.seg_main)

        self.loss_seg_consist1 = self.loss_seg_criterion_dice.forward(self.seg_output, self.warped_seg_help_1) + \
                              10 * self.loss_seg_criterion_focal(self.seg_output, self.warped_seg_help_1)
        self.loss_seg_consist2 = self.loss_seg_criterion_dice.forward(self.seg_output, self.warped_seg_help_2) + \
                              10 * self.loss_seg_criterion_focal(self.seg_output, self.warped_seg_help_2)

        self.loss_total = self.loss_seg_main + opt.seg_trade_off * self.loss_seg_consist1 + opt.seg_trade_off * self.loss_seg_consist2


        if opt.seg_train_mode == 'fusion_seg':
            self.loss_seg_help1 = self.loss_seg_criterion_dice.forward(self.seg_output[opt.batch_size:2*opt.batch_size, ...], self.warped_seg_help_1) + \
                                  10 * self.loss_seg_criterion_focal(self.seg_output[opt.batch_size:2*opt.batch_size, ...], self.warped_seg_help_1)
            self.loss_seg_help2 = self.loss_seg_criterion_dice.forward(self.seg_output[2*opt.batch_size:3*opt.batch_size, ...], self.warped_seg_help_2) + \
                                  10 * self.loss_seg_criterion_focal(self.seg_output[2*opt.batch_size:3*opt.batch_size, ...], self.warped_seg_help_2)
            self.loss_seg_consist1 = self.loss_seg_criterion_dice.forward(self.seg_output[0:opt.batch_size, ...],
                                                                          self.seg_output[opt.batch_size:2*opt.batch_size, ...].argmax(1).unsqueeze(1))
            self.loss_seg_consist2 = self.loss_seg_criterion_dice.forward(self.seg_output[0:opt.batch_size, ...],
                                                                          self.seg_output[2*opt.batch_size:3*opt.batch_size, ...].argmax(1).unsqueeze(1))

            self.loss_total = self.loss_total + self.loss_seg_help1 + self.loss_seg_help2 +\
                              opt.seg_trade_off * self.loss_seg_consist1 + opt.seg_trade_off * self.loss_seg_consist2

        # Transfer results to one-channel results
        self.seg_output = self.seg_output.argmax(1).unsqueeze(1)

        # Backward
        self.optimizer_seg.zero_grad()

        self.loss_total.backward()

        self.optimizer_seg.step()


    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            # forward()
            self.forward_val(opt)

            # Calculate segmentation loss (target should be one-channel)
            self.loss_seg_main = self.loss_seg_criterion_dice.forward(self.seg_output[0:opt.batch_size, ...], self.seg_main) + \
                                 10 * self.loss_seg_criterion_focal(self.seg_output[0:opt.batch_size, ...], self.seg_main)
            self.loss_seg_consist1 = self.loss_seg_criterion_dice.forward(self.seg_output[0:opt.batch_size, ...],
                                                                          self.warped_seg_help_1) + \
                                     10 * self.loss_seg_criterion_focal(self.seg_output[0:opt.batch_size, ...],
                                                                        self.warped_seg_help_1)
            self.loss_seg_consist2 = self.loss_seg_criterion_dice.forward(self.seg_output[0:opt.batch_size, ...],
                                                                          self.warped_seg_help_2) + \
                                     10 * self.loss_seg_criterion_focal(self.seg_output[0:opt.batch_size, ...],
                                                                        self.warped_seg_help_2)

            self.loss_total = self.loss_seg_main + opt.seg_trade_off * self.loss_seg_consist1 + opt.seg_trade_off * self.loss_seg_consist2

            if opt.seg_train_mode == 'fusion_seg':
                self.loss_seg_help1 = self.loss_seg_criterion_dice.forward(self.seg_output[opt.batch_size:2 * opt.batch_size, ...], self.warped_seg_help_1) + \
                                      10 * self.loss_seg_criterion_focal(self.seg_output[opt.batch_size:2 * opt.batch_size, ...], self.warped_seg_help_1)
                self.loss_seg_help2 = self.loss_seg_criterion_dice.forward(self.seg_output[2 * opt.batch_size:3 * opt.batch_size, ...], self.warped_seg_help_2) + \
                                      10 * self.loss_seg_criterion_focal(self.seg_output[2 * opt.batch_size:3 * opt.batch_size, ...], self.warped_seg_help_2)
                self.loss_seg_consist1 = self.loss_seg_criterion_dice.forward(self.seg_output[0:opt.batch_size, ...],
                                                                              self.seg_output[opt.batch_size:2*opt.batch_size, ...].argmax(1).unsqueeze(1))
                self.loss_seg_consist2 = self.loss_seg_criterion_dice.forward(self.seg_output[0:opt.batch_size, ...],
                                                                              self.seg_output[2*opt.batch_size:3*opt.batch_size, ...].argmax(1).unsqueeze(1))

                self.loss_total = self.loss_total + self.loss_seg_help1 + self.loss_seg_help2 +\
                                  opt.seg_trade_off * self.loss_seg_consist1 + opt.seg_trade_off * self.loss_seg_consist2

            # Transfer results to one-channel results
            self.seg_output = self.seg_output.argmax(1).unsqueeze(1)

