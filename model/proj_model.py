# --coding:utf-8--
import torch
from model import network
from model.base_model import BaseModel
from utils.loss import FocalLoss, ThreeDiceLoss, gradient_loss
from monai.losses.dice import DiceLoss


class ProjModel(BaseModel):
    """TODO: Build up IBIS segmentation and registration model."""
    def __init__(self, opt):
        super(ProjModel, self).__init__(opt)
        self.model_name = ['seg', 'reg']
        self.optimizer_name = ['seg', 'reg']
        self.loss_name = ['seg', 'reg', 'grid', 'total']
        self.visual_name = ['seg_output', 'warped_seg', 'warped_ori']

        # define networks
        if opt.phase == 'train' or opt.phase == 'val':
            self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.crop_size)
            self.net_reg = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.crop_size)
        elif opt.phase == 'test':
            self.net_seg = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.ori_size)
            self.net_reg = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.ori_size)

        # define Spatial Transformer (spa_tra)
        self.spa_tra = network.SpatialTransformer(opt.crop_size)
        self.spa_tra.to('cuda')

        # define loss
        self.loss_seg_criterion_dice = DiceLoss(to_onehot_y=True)
        self.loss_seg_criterion_focal = FocalLoss(class_num=4)
        self.loss_reg_criterion = ThreeDiceLoss()
        self.loss_grid_criterion = gradient_loss()

        # define optimizer
        self.optimizer_seg = torch.optim.Adam(self.net_seg.parameters(), lr=opt.lr)
        self.optimizer_reg = torch.optim.Adam(self.net_reg.parameters(), lr=opt.lr)

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

    def forward_seg_reg(self, opt):
        # implement segmentation
        self.seg_output = self.net_seg(torch.cat((self.img_06, self.img_12,
                                                  self.img_24), dim=0))

        # implement registration
        self.fix_seg_06 = self.seg_output[0: opt.batch_size, ...].float()#.argmax(1).unsqueeze(1).float()
        self.mov_seg_12 = self.seg_output[opt.batch_size: 2 * opt.batch_size, ...].float()#.argmax(1).unsqueeze(1).float()
        self.mov_seg_24 = self.seg_output[2 * opt.batch_size: 3 * opt.batch_size, ...].float()#.argmax(1).unsqueeze(1).float()
        self.fix_seg = torch.cat((self.fix_seg_06, self.fix_seg_06), dim=0)
        self.mov_seg = torch.cat((self.mov_seg_12, self.mov_seg_24), dim=0)
        self.mov_ori = torch.cat((self.img_12, self.img_24), dim=0)


        self.warped_seg, self.flow = self.net_reg(torch.cat((self.mov_seg, self.fix_seg), dim=1))
        self.warped_ori = self.spa_tra(self.mov_ori, self.flow)

    def forward_seg(self, opt):
        # implement segmentation
        self.seg_output = self.net_seg(torch.cat((self.img_06, self.img_12,
                                                  self.img_24), dim=0))


    def optimize_train_parameters(self, count, opt):
        if count % opt.update_frequency == 0:
            # forward()
            self.forward_seg_reg(opt)

            # Calculate segmentation loss
            target = torch.cat((self.seg_06, self.seg_12, self.seg_24), dim=0)
            self.loss_seg = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, target)

            # Calculate registration loss
            self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                    self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                    self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))

            # Calculate DVFs loss
            self.loss_grid = opt.trade_off * self.loss_grid_criterion(self.flow)

            # Calculate total loss
            self.loss_total = self.loss_seg + self.loss_reg + self.loss_grid

            # transfer results to one-channel results
            self.seg_output = self.seg_output.argmax(1).unsqueeze(1)
            self.warped_seg = self.warped_seg.argmax(1).unsqueeze(1)

            self.set_requires_grad(self.net_reg, True)
            self.optimizer_seg.zero_grad()
            self.optimizer_reg.zero_grad()
            self.loss_total.backward()
            self.optimizer_seg.step()
            self.optimizer_reg.step()

            del self.fix_seg_06, self.mov_seg_12, self.mov_seg_24,\
                self.fix_seg, self.mov_seg, self.mov_ori

        else:
            self.forward_seg(opt)

            # Calculate segmentation loss
            target = torch.cat((self.seg_06, self.seg_12, self.seg_24), dim=0)
            self.loss_seg = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, target)

            # transfer results to one-channel results
            self.seg_output = self.seg_output.argmax(1).unsqueeze(1)

            self.set_requires_grad(self.net_reg, False)
            self.optimizer_seg.zero_grad()
            self.loss_seg.backward()
            self.optimizer_seg.step()

        # self.forward_seg_reg(opt)
        # # Calculate segmentation loss
        # target = torch.cat((self.seg_06, self.seg_12, self.seg_24), dim=0)
        # self.loss_seg = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
        #                 10 * self.loss_seg_criterion_focal(self.seg_output, target)
        #
        # # Calculate registration loss
        # self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
        #                                         self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
        #                                         self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))
        #
        # # Calculate DVFs loss
        # self.loss_grid = opt.trade_off * self.loss_grid_criterion(self.flow)
        #
        # # Calculate total loss
        # self.loss_total = self.loss_seg + self.loss_reg + self.loss_grid
        # print('Before backpropagation seg parameter:', next(self.net_seg.parameters())[0, 0, 0, 0, 0])
        # print('Before backpropagation reg parameter:', next(self.net_reg.parameters())[0, 0, 0, 0, 0])
        # self.optimizer_seg.zero_grad()
        # self.optimizer_reg.zero_grad()
        # self.loss_reg.backward()
        # self.optimizer_seg.step()
        # self.optimizer_reg.step()
        # print('After backpropagation seg parameter:', next(self.net_seg.parameters())[0, 0, 0, 0, 0])
        # print('After backpropagation reg parameter:', next(self.net_reg.parameters())[0, 0, 0, 0, 0])
        # print('aaa')

    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            # forward()
            self.forward_seg_reg(opt)

            # Calculate segmentation loss
            target = torch.cat((self.seg_06, self.seg_12, self.seg_24), dim=0)
            self.loss_seg = self.loss_seg_criterion_dice.forward(self.seg_output, target) + \
                            10 * self.loss_seg_criterion_focal(self.seg_output, target)

            # Calculate registration loss
            self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                    self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                    self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))

            # Calculate DVFs loss
            self.loss_grid = opt.trade_off * self.loss_grid_criterion(self.flow)

            # Calculate total loss
            self.loss_total = self.loss_seg + self.loss_reg + self.loss_grid

            # transfer results to one-channel results
            self.seg_output = self.seg_output.argmax(1).unsqueeze(1)
            self.warped_seg = self.warped_seg.argmax(1).unsqueeze(1)

            del self.fix_seg_06, self.mov_seg_12, self.mov_seg_24,\
                self.fix_seg, self.mov_seg, self.mov_ori