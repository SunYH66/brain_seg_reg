# --coding:utf-8--
import torch
from brain_reg_seg.model import network
from brain_reg_seg.model.base_model import BaseModel
from brain_reg_seg.utils.loss import DiceLoss, ThreeDiceLoss, gradient_loss, to_one_hot
# from monai.losses.dice import DiceLoss

class RegModel(BaseModel):
    """TODO: Build up IBIS segment model."""
    def __init__(self, opt):
        super(RegModel, self).__init__(opt)
        self.model_name = ['reg']
        self.optimizer_name = ['reg']
        self.loss_name = ['reg', 'grid', 'total']
        self.visual_name = ['warped_seg', 'warped_ori', 'fix_seg', 'fix_ori', 'flow']
        self.path_name = ['fix_path', 'mov_path_1', 'mov_path_2'] #['fix_path', 'mov_path_1', 'mov_path_2'] ['fix_path', 'mov_path_1']

        # define networks
        if opt.phase == 'train' or opt.phase == 'val':
            self.net_reg = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.seg_train_mode, opt.crop_size)
        elif opt.phase == 'test':
            self.net_reg = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.seg_train_mode, opt.ori_size)

        # define Spatial Transformer (spa_tra)
        self.spa_tra = network.SpatialTransformer(opt.crop_size)
        self.spa_tra.to('cuda')

        # define loss TODO: change month for different reg modes
        # self.loss_reg_criterion = DiceLoss()
        self.loss_reg_criterion = ThreeDiceLoss()
        self.loss_grid_criterion = gradient_loss()

        # define optimizer
        self.optimizer_reg = torch.optim.Adam(self.net_reg.parameters(), lr=opt.lr)

    def set_input(self, input_data):
        self.img_fix = input_data['img_fix'].to('cuda').float()
        self.img_mov_1 = input_data['img_mov_1'].to('cuda').float()
        self.img_mov_2 = input_data['img_mov_2'].to('cuda').float()
        self.seg_fix = input_data['seg_fix'].to('cuda').float()
        self.GT = self.seg_fix
        self.seg_mov_1 = input_data['seg_mov_1'].to('cuda').float()
        self.seg_mov_2 = input_data['seg_mov_2'].to('cuda').float()
        self.fix_path = input_data['img_path_fix']
        self.mov_path_1 = input_data['img_path_mov_1']
        self.mov_path_2 = input_data['img_path_mov_2']

    def forward(self):
        # implement registration
        self.fix_seg = torch.cat((to_one_hot(self.seg_fix), to_one_hot(self.seg_fix)), dim=0)
        self.mov_seg = torch.cat((to_one_hot(self.seg_mov_1), to_one_hot(self.seg_mov_2)), dim=0)
        self.fix_ori = torch.cat((self.img_fix, self.img_fix), dim=0)
        self.mov_ori = torch.cat((self.img_mov_1, self.img_mov_2), dim=0)

        self.warped_seg, self.flow = self.net_reg(torch.cat((self.mov_seg, self.fix_seg), dim=1))
        self.warped_ori = self.spa_tra(self.mov_ori, self.flow)

    def optimize_train_parameters(self, count, opt):
        # forward()
        self.forward()

        # Calculate registration loss
        self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))

        # Calculate DVFs loss
        self.loss_grid = opt.reg_trade_off * self.loss_grid_criterion(self.flow)

        # Calculate total loss
        self.loss_total = self.loss_reg + self.loss_grid

        self.optimizer_reg.zero_grad()
        self.loss_total.backward()
        self.optimizer_reg.step()

        # transfer results to one-channel results
        self.fix_seg = self.fix_seg.argmax(1).unsqueeze(1)
        self.warped_seg = self.warped_seg.argmax(1).unsqueeze(1)


    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            self.forward()

            # Calculate registration loss
            self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                    self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                    self.fix_seg[0: opt.batch_size, ...].argmax(1).unsqueeze(1))

            # Calculate DVFs loss
            self.loss_grid = opt.reg_trade_off * self.loss_grid_criterion(self.flow)

            # Calculate total loss
            self.loss_total = self.loss_reg + self.loss_grid

            # transfer results to one-channel results
            self.fix_seg = self.fix_seg.argmax(1).unsqueeze(1)
            self.warped_seg = self.warped_seg.argmax(1).unsqueeze(1)