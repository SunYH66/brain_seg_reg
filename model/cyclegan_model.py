# --coding:utf-8--
import torch
import itertools
from brain_reg_seg.model import network
from brain_reg_seg.model.base_model import BaseModel
from brain_reg_seg.utils.utils import ImagePool
from monai.inferers import SlidingWindowInferer

class CycleGANModel(BaseModel):
    """TODO: Build up IBIS segmentation model."""
    def __init__(self, opt):
        super(CycleGANModel, self).__init__(opt)
        self.model_name = ['GA', 'GB', 'DA', 'DB']
        self.optimizer_name = ['GA', 'GB', 'DA', 'DB']
        self.loss_name = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        visual_name_A = ['real_A', 'fake_B', 'rec_A', 'idt_B']
        visual_name_B = ['real_B', 'fake_A', 'rec_B', 'idt_A']
        self.visual_name = visual_name_A + visual_name_B

        self.path_name = ['img_path_main']

        # define networks
        self.net_GA = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.seg_train_mode, opt.crop_size)
        self.net_GB = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.seg_train_mode, opt.crop_size)

        self.net_DA = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.seg_train_mode, opt.crop_size, cycleG=False)
        self.net_DB = network.define_network(opt.input_nc, opt.output_nc, opt.seg_net_type, opt.seg_train_mode, opt.crop_size, cycleG=False)

        self.fake_A_pool = ImagePool(opt.pool_size)
        self.fake_B_pool = ImagePool(opt.pool_size)

        # define loss
        self.criterionGAN = network.CycleGAN.GANLoss('lsgan').to(self.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionConsist = torch.nn.MSELoss()

        # define optimizer
        self.optimizer_GA = torch.optim.Adam(self.net_GA.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_GB = torch.optim.Adam(self.net_GB.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_DA = torch.optim.Adam(self.net_DA.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_DB = torch.optim.Adam(self.net_DB.parameters(), lr=opt.lr, betas=(0.5, 0.999))


    def set_input(self, input_data):
        self.real_A = input_data['img_domainA'].to('cuda').float()
        self.real_B = input_data['img_domainB'].to('cuda').float()
        self.img_path_main = input_data['img_path']


    def forward_train(self, opt):
        self.fake_B = self.net_GA(self.real_A)
        self.rec_A = self.net_GB(self.fake_B)

        self.fake_A = self.net_GB(self.real_B)
        self.rec_B = self.net_GA(self.fake_A)

    def forward_val(self, opt):
        # define sliding window inference object
        inferer = SlidingWindowInferer(opt.crop_size, overlap=0.25)
        self.fake_B = inferer(self.real_A, self.net_GA)
        self.rec_A = inferer(self.fake_B, self.net_GB)

        self.fake_A = inferer(self.real_B, self.net_GB)
        self.rec_B = inferer(self.fake_A, self.net_GA)


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D


    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.net_DA, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.net_DB, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = 0.5
        lambda_A = 10.0
        lambda_B = 10.0
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.net_GA(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.net_GB(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.net_DA(self.fake_B), True)

        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.net_DB(self.fake_A), True)

        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()


    def optimize_train_parameters(self, epoch, opt):
        # forward
        self.forward_train(opt)
        # G_A and G_B
        self.optimizer_GA.zero_grad()
        self.optimizer_GB.zero_grad()
        self.backward_G()
        self.optimizer_GA.step()
        self.optimizer_GB.step()
        # D_A and D_B
        self.optimizer_DA.zero_grad()
        self.optimizer_DB.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_DA.step()
        self.optimizer_DB.step()


    def optimize_val_parameters(self, opt):
        with torch.no_grad():
            # forward
            self.forward_val(opt)
