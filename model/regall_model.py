# --coding:utf-8--
import torch
from model import network
from model.base_model import BaseModel
from utils.loss import ThreeDiceLoss, gradient_loss, to_one_hot
# from monai.losses.dice import DiceLoss

class RegModel(BaseModel):
    """TODO: Build up IBIS segment model."""
    def __init__(self, opt):
        super(RegModel, self).__init__(opt)
        self.model_name = ['reg_06', 'reg_12', 'reg_24']
        self.optimizer_name = ['reg_06', 'reg_12', 'reg_24']
        self.loss_name = ['reg', 'grid', 'total']
        self.visual_name = ['warped_seg', 'warped_ori', 'fix_seg', 'fix_ori']
        self.path_name = ['fix_path', 'mov_path_1', 'mov_path_2'] #['fix_path', 'mov_path_1', 'mov_path_2'] ['fix_path', 'mov_path_1']

        # define networks
        if opt.phase == 'train' or opt.phase == 'val':
            self.net_reg_06 = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.seg_train_mode, opt.crop_size)
            self.net_reg_12 = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.seg_train_mode, opt.crop_size)
            self.net_reg_24 = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.seg_train_mode, opt.crop_size)
        elif opt.phase == 'test':
            self.net_reg_06 = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.seg_train_mode, opt.ori_size)
            self.net_reg_12 = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.seg_train_mode, opt.ori_size)
            self.net_reg_24 = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.seg_train_mode, opt.ori_size)

        # define Spatial Transformer (spa_tra)
        self.spa_tra = network.SpatialTransformer(opt.crop_size)
        self.spa_tra.to('cuda')

        # define loss TODO: change month for different reg modes
        self.loss_reg_criterion = ThreeDiceLoss()
        self.loss_grid_criterion = gradient_loss()

        # define optimizer
        self.optimizer_reg_06 = torch.optim.Adam(self.net_reg_06.parameters(), lr=opt.lr)
        self.optimizer_reg_12 = torch.optim.Adam(self.net_reg_12.parameters(), lr=opt.lr)
        self.optimizer_reg_24 = torch.optim.Adam(self.net_reg_24.parameters(), lr=opt.lr)

    def set_input(self, input_data):
        self.img_06 = input_data['img_06'].to('cuda').float()
        self.img_12 = input_data['img_12'].to('cuda').float()
        self.img_24 = input_data['img_24'].to('cuda').float()

        self.seg_06 = input_data['seg_06'].to('cuda').float()
        self.seg_12 = input_data['seg_12'].to('cuda').float()
        self.seg_24 = input_data['seg_24'].to('cuda').float()

        self.warped_ori_12_06 = input_data['warped_ori_12_06'].to('cuda').float()
        self.warped_ori_24_06 = input_data['warped_ori_24_06'].to('cuda').float()
        self.warped_ori_06_12 = input_data['warped_ori_06_12'].to('cuda').float()
        self.warped_ori_24_12 = input_data['warped_ori_24_12'].to('cuda').float()
        self.warped_ori_06_24 = input_data['warped_ori_06_24'].to('cuda').float()
        self.warped_ori_12_24 = input_data['warped_ori_12_24'].to('cuda').float()

        self.warped_seg_12_06 = input_data['warped_seg_12_06'].to('cuda').float()
        self.warped_seg_24_06 = input_data['warped_seg_24_06'].to('cuda').float()
        self.warped_seg_06_12 = input_data['warped_seg_06_12'].to('cuda').float()
        self.warped_seg_24_12 = input_data['warped_seg_24_12'].to('cuda').float()
        self.warped_seg_06_24 = input_data['warped_seg_06_24'].to('cuda').float()
        self.warped_seg_12_24 = input_data['warped_seg_12_24'].to('cuda').float()

        self.img_path_06 = input_data['img_path_06']
        self.img_path_12 = input_data['img_path_12']
        self.img_path_24 = input_data['img_path_24']

    def forward_06(self):
        # implement registration
        self.fix_seg = torch.cat((to_one_hot(self.seg_06), to_one_hot(self.seg_06)), dim=0)
        self.mov_seg = torch.cat((to_one_hot(self.warped_seg_12_06), to_one_hot(self.warped_seg_24_06)), dim=0)
        self.fix_ori = torch.cat((self.img_06, self.img_06), dim=0)
        self.mov_ori = torch.cat((self.warped_ori_12_06, self.warped_ori_24_06), dim=0)

        self.fix_path = self.img_path_06
        self.mov_path_1 = self.img_path_12
        self.mov_path_2 = self.img_path_24

        self.warped_seg, self.flow = self.net_reg_06(torch.cat((self.mov_seg, self.fix_seg), dim=1))
        self.warped_ori = self.spa_tra(self.mov_ori, self.flow)

    def forward_12(self):
        # implement registration
        self.fix_seg = torch.cat((to_one_hot(self.seg_12), to_one_hot(self.seg_12)), dim=0)
        self.mov_seg = torch.cat((to_one_hot(self.warped_seg_06_12), to_one_hot(self.warped_seg_24_12)), dim=0)
        self.fix_ori = torch.cat((self.img_12, self.img_12), dim=0)
        self.mov_ori = torch.cat((self.warped_ori_06_12, self.warped_ori_24_12), dim=0)

        self.fix_path = self.img_path_12
        self.mov_path_1 = self.img_path_06
        self.mov_path_2 = self.img_path_24

        self.warped_seg, self.flow = self.net_reg_12(torch.cat((self.mov_seg, self.fix_seg), dim=1))
        self.warped_ori = self.spa_tra(self.mov_ori, self.flow)

    def forward_24(self):
        # implement registration
        self.fix_seg = torch.cat((to_one_hot(self.seg_24), to_one_hot(self.seg_24)), dim=0)
        self.mov_seg = torch.cat((to_one_hot(self.warped_seg_06_24), to_one_hot(self.warped_seg_12_24)), dim=0)
        self.fix_ori = torch.cat((self.img_24, self.img_24), dim=0)
        self.mov_ori = torch.cat((self.warped_ori_06_24, self.warped_ori_12_24), dim=0)

        self.fix_path = self.img_path_24
        self.mov_path_1 = self.img_path_06
        self.mov_path_2 = self.img_path_12

        self.warped_seg, self.flow = self.net_reg_24(torch.cat((self.mov_seg, self.fix_seg), dim=1))
        self.warped_ori = self.spa_tra(self.mov_ori, self.flow)


    def optimize_train_parameters(self, count, opt):
        #=============================================forward_06=============================================
        # forward()
        self.forward_06()

        # Calculate registration loss
        self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))

        # Calculate DVFs loss
        self.loss_grid = opt.reg_trade_off * self.loss_grid_criterion(self.flow)

        # Calculate total loss
        self.loss_total = self.loss_reg + self.loss_grid

        self.optimizer_reg_06.zero_grad()
        self.loss_total.backward()
        self.optimizer_reg_06.step()

        # =============================================forward_12=============================================
        # forward()
        self.forward_12()
        # Calculate registration loss
        self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))

        # Calculate DVFs loss
        self.loss_grid = opt.reg_trade_off * self.loss_grid_criterion(self.flow)

        # Calculate total loss
        self.loss_total = self.loss_reg + self.loss_grid

        self.optimizer_reg_12.zero_grad()
        self.loss_total.backward()
        self.optimizer_reg_12.step()

        # =============================================forward_24==============================================
        # forward()
        self.forward_24()
        # Calculate registration loss
        self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))

        # Calculate DVFs loss
        self.loss_grid = opt.reg_trade_off * self.loss_grid_criterion(self.flow)

        # Calculate total loss
        self.loss_total = self.loss_reg + self.loss_grid

        self.optimizer_reg_24.zero_grad()
        self.loss_total.backward()
        self.optimizer_reg_24.step()


    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            pass
            # self.forward_06()
            # # transfer results to one-channel results
            # self.fix_seg = self.fix_seg.argmax(1).unsqueeze(1)
            # self.warped_seg = self.warped_seg.argmax(1).unsqueeze(1)
            #
            # self.forward_12()
            # # transfer results to one-channel results
            # self.fix_seg = self.fix_seg.argmax(1).unsqueeze(1)
            # self.warped_seg = self.warped_seg.argmax(1).unsqueeze(1)
            #
            #
            # self.forward_24()
            # # transfer results to one-channel results
            # self.fix_seg = self.fix_seg.argmax(1).unsqueeze(1)
            # self.warped_seg = self.warped_seg.argmax(1).unsqueeze(1)
            #
            #
            # # Calculate registration loss
            # self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
            #                                         self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
            #                                         self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))
            #
            # # Calculate DVFs loss
            # self.loss_grid = opt.reg_trade_off * self.loss_grid_criterion(self.flow)
            #
            # # Calculate total loss
            # self.loss_total = self.loss_reg + self.loss_grid
