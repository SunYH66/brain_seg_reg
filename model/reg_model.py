# --coding:utf-8--
import torch
from model import network
from model.base_model import BaseModel
from utils.loss import DiceLoss, ThreeDiceLoss, gradient_loss
# from monai.losses.dice import DiceLoss

class RegModel(BaseModel):
    """TODO: Build up IBIS segment model."""
    def __init__(self, opt):
        super(RegModel, self).__init__(opt)
        self.model_name = ['reg']
        self.loss_name = ['reg', 'grid', 'total']
        self.visual_name = ['warped_seg', 'warped_ori']

        # define networks
        if opt.phase == 'train' or opt.phase == 'val':
            self.net_reg = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.crop_size)
        elif opt.phase == 'test':
            self.net_reg = network.define_network(opt.input_nc, opt.output_nc, opt.reg_net_type, opt.ori_size)

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
        # implement registration
        self.fix_seg = torch.cat((self.seg_06, self.seg_06), dim=0)
        self.mov_seg = torch.cat((self.seg_12, self.seg_24), dim=0)
        self.mov_ori = torch.cat((self.img_12, self.img_24), dim=0)

        # self.warped_seg, self.flow = self.net_reg(torch.cat((self.seg_12, self.seg_06), dim=1)) # TODO: change month for different reg modes
        self.warped_seg, self.flow = self.net_reg(torch.cat((self.mov_seg, self.fix_seg), dim=1)) # TODO: change month for different reg modes
        self.warped_ori = self.spa_tra(self.mov_ori, self.flow) # TODO: change month for different reg modes


    def optimize_train_parameters(self, count, opt):
        # forward()
        if count % opt.update_frequency == 0:
            # forward()
            self.forward(opt)

            # Calculate registration loss
            # self.loss_reg = self.loss_reg_criterion.forward(self.warped_seg, self.seg_06)
            self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                    self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                    self.fix_seg[0: opt.batch_size, ...])

            # Calculate DVFs loss
            self.loss_grid = opt.trade_off * self.loss_grid_criterion(self.flow)

            # Calculate total loss
            self.loss_total = self.loss_reg + self.loss_grid

            self.optimizer_reg.zero_grad()
            self.loss_total.backward()
            self.optimizer_reg.step()


    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            self.forward(opt)

            # Calculate registration loss
            # self.loss_reg = self.loss_reg_criterion.forward(self.warped_seg, self.seg_06)
            self.loss_reg = self.loss_reg_criterion(self.warped_seg[0: opt.batch_size, ...],
                                                    self.warped_seg[opt.batch_size: 2 * opt.batch_size, ...],
                                                    self.fix_seg[0: opt.batch_size, ...])

            # Calculate DVFs loss
            self.loss_grid = opt.trade_off * self.loss_grid_criterion(self.flow)

            # Calculate total loss
            self.loss_total = self.loss_reg + self.loss_grid