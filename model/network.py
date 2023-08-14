# --coding:utf-8--
from brain_reg_seg.config.config import cfg
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .comparison.HyperDenseNet import HyperDenseNet
from .comparison.HyperTransformer import HyperTransformer
from .comparison.SkipDenseNet import SkipDenseNet
from .comparison.ResNetVAE import ResNetVAE
from .comparison.DenseVoxelNet import DenseVoxelNet
from .comparison import CycleGAN

# ========================================================================================================
# Basic functions for network
# ========================================================================================================
def init_weights(net, init_type='kaiming', init_gain=0.02):
    """
    Initialize the network weights.

    :param net:
    :param init_type:
    :param init_gain:
    :return:

    We use 'normal' in the by default, but xavier and kaiming might work better
    for some applications. Feel free to try yourself.
    """
    def init_func(m): # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1 or classname.find('BatchNorm1d') != -1: # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize the network with [%s]' % init_type)
    net.apply(init_func)


def init_net(net, init_type='kaiming', init_gain=0.02):
    """
    Initialize the network: 1. register CPU/GPU device (with multi-GPU support), 2. initialize the network weights
    :param net:
    :param init_type:
    :param init_gain:
    :return:
    """
    net.to('cuda')
    net = torch.nn.DataParallel(net)
    init_weights(net, init_type, init_gain)
    return net


def define_network(input_nc, output_nc, net_type, seg_train_mode, vol_size, seg_fusion=False, init_type='kaiming', init_gain=0.02, cycleG=True):
    """
    Create a network

    :param input_nc: the number of channels in input images
    :param output_nc: the number of channels in output images
    :param net_type: the architecture names: reg_net, seg_net
    :param seg_train_mode: the mode of seg_net training (seperate or joint)
    :param vol_size: the voxel size to be registered
    :param seg_fusion: true or false
    :param init_type: the name of initialization method
    :param init_gain: scaling factor for normal, xavier and orthogonal
    :param cycleG if True, define generator, else define discriminator
    :return: return a network

    Return a network
    """
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]
    if net_type == 'voxelmorph':
        net = RegNet(vol_size, nf_enc, nf_dec)
    elif seg_train_mode == 'fusion_only':
        net = FusionOnlySegNet(input_nc, output_nc, seg_fusion)
    elif seg_train_mode == 'fusion_seg':
        net = FusionSegSegNet(input_nc, output_nc, seg_fusion)
    elif seg_train_mode == 'single':
        net = SingleSegNet(input_nc, output_nc, seg_fusion)
    elif seg_train_mode == 'trans':
        net = HyperTransformer()
    elif seg_train_mode == 'attention':
        net = AttUNet()
    elif seg_train_mode == 'HyperDenseNet':
        net = HyperDenseNet()
    elif seg_train_mode == 'SkipDenseNet':
        net = SkipDenseNet()
    elif seg_train_mode == 'ResNetVAE':
        net = ResNetVAE()
    elif seg_train_mode == 'DenseVoxelNet':
        net = DenseVoxelNet()
    elif seg_train_mode == 'CycleGAN' and cycleG == True:
        net = CycleGAN.define_G(input_nc, output_nc, ngf=64, netG='resnet_9blocks')
    elif seg_train_mode == 'CycleGAN' and cycleG == False:
        net = CycleGAN.define_D(input_nc, ndf=64, netD='basic')
    else:
        raise NotImplementedError('Network model name [%s] is not recognized' % net_type)
    return init_net(net, init_type, init_gain)


# ========================================================================================================
# Classes for building up SEGMENTATION network
# ========================================================================================================
class BasicBlock(nn.Module):
    # TODO: basic convolutional block, conv -> batchnorm -> activate
    def __init__(self, in_channels, out_channels, kernel_size, padding, activate=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, bias=True)
        self.bn = nn.BatchNorm3d(out_channels)
        self.activate = nn.ReLU(inplace=True)
        self.en_activate = activate

    def forward(self, x):
        if self.en_activate:
            return self.activate(self.bn(self.conv(x)))

        else:
            return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    # TODO: basic residual block established by BasicBlock
    def __init__(self, in_channels, out_channels, kernel_size, padding, nums):
        """
        TODO: initial parameters for basic residual network
        :param in_channels: input channel numbers
        :param out_channels: output channel numbers
        :param kernel_size: convoluition kernel size
        :param padding: padding size
        :param nums: number of basic convolutional layer
        """
        super(ResidualBlock, self).__init__()

        layers = list()

        for _ in range(nums):
            if _ != nums - 1:
                layers.append(BasicBlock(in_channels, out_channels, kernel_size, padding, True))

            else:
                layers.append(BasicBlock(in_channels, out_channels, kernel_size, padding, False))

        self.do = nn.Sequential(*layers)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.do(x)
        return self.activate(output + x)


class InputTransition(nn.Module):
    # TODO: input transition convert image to feature space
    def __init__(self, in_channels, out_channels):
        """
        TODO: initial parameter for input transition <input size equals to output feature size>
        :param in_channels: input image channels
        :param out_channels: output feature channles
        """
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.activate1(self.bn1(self.conv1(x)))
        return out


class OutputTransition(nn.Module):
    # TODO: feature map convert to predict results
    def __init__(self, in_channels, out_channels, act='sigmoid'):
        """
        TODO: initial for output transition
        :param in_channels: input feature channels
        :param out_channels: output results channels
        :param act: final activate layer sigmoid or softmax
        """
        super(OutputTransition, self).__init__()
        assert act == 'sigmoid' or act =='softmax', \
            'final activate layer should be sigmoid or softmax, current activate is :{}'.format(act)
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1, 1))

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.act = act

    def forward(self, x):
        out = self.activate1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        if self.act == 'sigmoid':
            return self.sigmoid(out)
        elif self.act == 'softmax':
            return self.softmax(out)


class DownTransition(nn.Module):
    # TODO: fundamental down-sample layer <inchannel -> 2*inchannel>
    def __init__(self, in_channels, nums):
        """
        TODO: intial for down-sample
        :param in_channels: inpuit channels
        :param nums: number of reisidual block
        """
        super(DownTransition, self).__init__()

        out_channels = in_channels * 2
        self.down = nn.Conv3d(in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activate1 = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums)

    def forward(self, x):
        out = self.activate1(self.bn1(self.down(x)))
        out = self.residual(out)
        return out


class UpTransition(nn.Module):
    # TODO: fundamental up-sample layer (inchannels -> inchannels/2)
    def __init__(self, in_channels, out_channels, nums):
        """
        TODO: initial for up-sample
        :param in_channels: input channels
        :param out_channels: output channels
        :param nums: number of residual block
        """
        super(UpTransition, self).__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels//2, kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=1)
        self.bn = nn.BatchNorm3d(out_channels//2)
        self.activate = nn.ReLU(inplace=True)
        self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums)
        # self.basicblock = BasicBlock(out_channels, out_channels, 3, 1)

    def forward(self, x, skip_x):
        out = self.activate(self.bn(self.conv1(x)))
        out = torch.cat((out, skip_x), 1)
        out = self.residual(out)
        # out = self.basicblock(out)

        return out


class transformer_layer(nn.Module):
    def __init__(self, nhead=8, batch_first=True, nhidden=384):
        super(transformer_layer, self).__init__()
        self.transformer = nn.Transformer(nhead=nhead, d_model=nhidden, dropout=True, batch_first=batch_first)

    def forward(self, source, target):
        BS, CS, WS, HS, DS = source.shape
        BT, CT, WT, HT, DT = target.shape
        source = source.reshape(BS, CS, WS*HS*DS)
        target = target.reshape(BT, CT, WT*HT*DT)
        out = self.transformer(source, target)
        out = out.reshape(BT, CT, WT, HT, DT)
        return out


class SingleSegNet(nn.Module):
    # TODO: fundamental segmentation framework
    def __init__(self, in_channels, out_channels, fusion=False):
        super(SingleSegNet, self).__init__()
        self.fusion = fusion
        self.in_tr = InputTransition(in_channels, 16)
        self.down_32 = DownTransition(16, 1)
        self.down_64 = DownTransition(32, 1)
        self.down_128 = DownTransition(64, 2)
        self.down_256 = DownTransition(128, 2)

        self.up_256 = UpTransition(256, 256, 2)
        self.up_128 = UpTransition(256, 128, 2)
        self.up_64 = UpTransition(128, 64, 1)
        self.up_32 = UpTransition(64, 32, 1)

        self.out_tr = OutputTransition(32, out_channels, 'softmax')


    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        out = self.up_256(out256, out128)
        out = self.up_128(out, out64)
        out = self.up_64(out, out32)
        out = self.up_32(out, out16)

        out = self.out_tr(out)

        return [out]


class FusionOnlySegNet(nn.Module):
    # TODO: fundamental segmentation framework
    def __init__(self, in_channels, out_channels, fusion=False):
        super(FusionOnlySegNet, self).__init__()
        self.fusion = fusion
        self.in_tr_main = InputTransition(in_channels, 16)
        self.in_tr_help1 = InputTransition(in_channels, 16)
        self.in_tr_help2 = InputTransition(in_channels, 16)

        self.down_32_main = DownTransition(16, 1)
        self.down_64_main = DownTransition(32, 1)
        self.down_128_main = DownTransition(64, 2)
        self.down_256_main = DownTransition(128, 2)

        self.down_32_help1 = DownTransition(16, 1)
        self.down_64_help1 = DownTransition(32, 1)
        self.down_128_help1 = DownTransition(64, 2)
        self.down_256_help1 = DownTransition(128, 2)

        self.down_32_help2 = DownTransition(16, 1)
        self.down_64_help2 = DownTransition(32, 1)
        self.down_128_help2 = DownTransition(64, 2)
        self.down_256_help2 = DownTransition(128, 2)

        self.up_256 = UpTransition(256, 256, 2)
        self.up_128 = UpTransition(256, 128, 2)
        self.up_64 = UpTransition(128, 64, 1)
        self.up_32 = UpTransition(64, 32, 1)

        self.out_tr = OutputTransition(32, out_channels, 'softmax')

        if self.fusion:
            #############################
            # fusion
            #############################
            self.basicblock64 = BasicBlock(64 * 3, 64, kernel_size=3, padding=1)
            self.basicblock128 = BasicBlock(128 * 3, 128, kernel_size=3, padding=1)
            self.basicblock256 = BasicBlock(256 * 3, 256, kernel_size=3, padding=1)

            #############################
            # multiheadattention
            #############################
            # self.trans_layer = transformer_layer()
            # self.basicblock256 = BasicBlock(256 * 2, 256, kernel_size=3, padding=1)


    def forward(self, x):

        out16_main = self.in_tr_main(x[:,0:1, ...])
        out32_main = self.down_32_main(out16_main)
        out64_main = self.down_64_main(out32_main)
        out128_main = self.down_128_main(out64_main)
        out256_main = self.down_256_main(out128_main)

        out16_help1 = self.in_tr_help1(x[:,1:2, ...])
        out32_help1 = self.down_32_help1(out16_help1)
        out64_help1 = self.down_64_help1(out32_help1)
        out128_help1 = self.down_128_help1(out64_help1)
        out256_help1 = self.down_256_help1(out128_help1)

        out16_help2 = self.in_tr_help2(x[:,2:3, ...])
        out32_help2 = self.down_32_help2(out16_help2)
        out64_help2 = self.down_64_help2(out32_help2)
        out128_help2 = self.down_128_help2(out64_help2)
        out256_help2 = self.down_256_help2(out128_help2)

        #############################
        # 64 fusion
        #############################
        B64, C64, H64, W64, D64 = out64_main.shape
        out64_fusion = torch.empty((B64, 3 * C64, H64, W64, D64)).cuda()
        if self.fusion:
            for i in range(cfg.batch_size):
                out64_fusion[i:i+1, ...] = torch.cat((out64_main[i:i+1, ...],
                                                       out64_help1[i:i+1, ...],
                                                       out64_help2[i:i+1, ...]), dim=1)

            out64_fusion = self.basicblock64(out64_fusion)
        out128_main = self.down_128_main(out64_fusion if self.fusion else out64_main)
        #############################


        #############################
        # 128 fusion
        #############################
        B128, C128, H128, W128, D128 = out128_main.shape
        out128_fusion = torch.empty((B128, 3 * C128, H128, W128, D128)).cuda()
        if self.fusion:
            for i in range(cfg.batch_size):
                out128_fusion[i:i+1, ...] = torch.cat((out128_main[i:i+1, ...],
                                                       out128_help1[i:i+1, ...],
                                                       out128_help2[i:i+1, ...]), dim=1)

            out128_fusion = self.basicblock128(out128_fusion)
        out256_main = self.down_256_main(out128_fusion if self.fusion else out128_main)
        #############################


        #############################
        # 256 fusion
        #############################
        B256, C256, H256, W256, D256 = out256_main.shape
        out256_fusion = torch.empty((B256, 3 * C256, H256, W256, D256)).cuda()
        if self.fusion:
            for i in range(cfg.batch_size):
                out256_fusion[i:i+1, ...] = torch.cat((out256_main[i:i+1, ...],
                                                       out256_help1[i:i+1, ...],
                                                       out256_help2[i:i+1, ...]), dim=1)

            out256_fusion = self.basicblock256(out256_fusion)
        #############################

        """
        #############################
        # 256 Multiheadattention fusion
        #############################
        B256, C256, H256, W256, D256 = out256_main.shape

        # mha_256 = MultiheadAttention(H256 * W256 * D256, 8, batch_first=True).to('cuda')
        out256_fusion = torch.empty((B256, C256, H256, W256, D256)).cuda()
        if self.fusion:
            mha_256 = self.trans_layer(out256_main, out256_help2)
            # out256_att, _ = mha_256(
            #     out256_main.reshape(B256, C256, H256 * W256 * D256),
            #     out256_help1.reshape(B256, C256, H256 * W256 * D256),
            #     out256_help2.reshape(B256, C256, H256 * W256 * D256))
            # out256_att = out256_att.reshape(B256, C256, H256, W256, D256)

            out256_fusion = self.basicblock256(torch.cat((out256_main, mha_256), dim=1))
        ######################
        """
        out_main = self.up_256(out256_fusion if self.fusion else out256_main, out128_fusion if self.fusion else out128_main)#out128_fusion if self.fusion else out128_main
        out_main = self.up_128(out_main, out64_fusion if self.fusion else out64_main)#out64_fusion if self.fusion else out64_main
        out_main = self.up_64(out_main, out32_main)
        out_main = self.up_32(out_main, out16_main)

        out_main = self.out_tr(out_main)

        return [out_main]


class FusionSegSegNet(nn.Module):
    def __init__(self, in_channels, out_channels, fusion=False):
        super(FusionSegSegNet, self).__init__()
        self.fusion = fusion
        self.in_tr_main = InputTransition(in_channels, 16)
        self.down_32_main = DownTransition(16, 1)
        self.down_64_main = DownTransition(32, 1)
        self.down_128_main = DownTransition(64, 2)
        self.down_256_main = DownTransition(128, 2)

        self.in_tr_help1 = InputTransition(in_channels, 16)
        self.down_32_help1 = DownTransition(16, 1)
        self.down_64_help1 = DownTransition(32, 1)
        self.down_128_help1 = DownTransition(64, 2)
        self.down_256_help1 = DownTransition(128, 2)

        self.in_tr_help2 = InputTransition(in_channels, 16)
        self.down_32_help2 = DownTransition(16, 1)
        self.down_64_help2 = DownTransition(32, 1)
        self.down_128_help2 = DownTransition(64, 2)
        self.down_256_help2 = DownTransition(128, 2)

        self.up_256_main = UpTransition(256, 256, 2)
        self.up_128_main = UpTransition(256, 128, 2)
        self.up_64_main = UpTransition(128, 64, 1)
        self.up_32_main = UpTransition(64, 32, 1)

        self.up_256_help1 = UpTransition(256, 256, 2)
        self.up_128_help1 = UpTransition(256, 128, 2)
        self.up_64_help1 = UpTransition(128, 64, 1)
        self.up_32_help1 = UpTransition(64, 32, 1)

        self.up_256_help2 = UpTransition(256, 256, 2)
        self.up_128_help2 = UpTransition(256, 128, 2)
        self.up_64_help2 = UpTransition(128, 64, 1)
        self.up_32_help2 = UpTransition(64, 32, 1)

        self.out_tr_main = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_help1 = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_help2 = OutputTransition(32, out_channels, 'softmax')

        if self.fusion:
            #############################
            # fusion
            #############################
            # self.basicblock64 = BasicBlock(64 * 3, 64, kernel_size=3, padding=1)
            # self.basicblock128 = BasicBlock(128 * 3, 128, kernel_size=3, padding=1)
            self.basicblock256 = BasicBlock(256 * 3, 256, kernel_size=3, padding=1)

            #############################
            # multiheadattention
            #############################
            # self.basicblock256 = BasicBlock(256 * 2, 256, kernel_size=3, padding=1)

    def forward(self, img):
        # split the img into img_06, img_12, img_24
        img_06 = img[:,0:1, ...]
        img_12 = img[:,1:2, ...]
        img_24 = img[:,2:3, ...]

        # seg for 06 month
        out16_main = self.in_tr_main(img_06)
        out32_main = self.down_32_main(out16_main)
        out64_main = self.down_64_main(out32_main)
        out128_main = self.down_128_main(out64_main)
        out256_main = self.down_256_main(out128_main)

        # seg for 12 month
        out16_help1 = self.in_tr_help1(img_12)
        out32_help1 = self.down_32_help1(out16_help1)
        out64_help1 = self.down_64_help1(out32_help1)
        out128_help1 = self.down_128_help1(out64_help1)
        out256_help1 = self.down_256_help1(out128_help1)

        # seg for 24 month
        out16_help2 = self.in_tr_help2(img_24)
        out32_help2 = self.down_32_help2(out16_help2)
        out64_help2 = self.down_64_help2(out32_help2)
        out128_help2 = self.down_128_help2(out64_help2)
        out256_help2 = self.down_256_help2(out128_help2)

        #############################
        # 64 fusion
        #############################
        # B64, C64, H64, W64, D64 = out64_main.shape
        # out64_fusion = torch.empty((B64, 3 * C64, H64, W64, D64)).cuda()
        # if self.fusion:
        #     for i in range(cfg.batch_size):
        #         out64_fusion[i:i+1, ...] = torch.cat((out64_main[i:i+1, ...],
        #                                               out64_help1[i:i+1, ...],
        #                                               out64_help2[i:i+1, ...]), dim=1)
        #
        #     out64_fusion = self.basicblock64(out64_fusion)
        # out128_main = self.down_128_main(out64_fusion if self.fusion else out64_main)
        #############################

        #############################
        # 128 fusion
        #############################
        # B128, C128, H128, W128, D128 = out128_main.shape
        # out128_fusion = torch.empty((B128, 3 * C128, H128, W128, D128)).cuda()
        # if self.fusion:
        #     for i in range(cfg.batch_size):
        #         out128_fusion[i:i+1, ...] = torch.cat((out128_main[i:i+1, ...],
        #                                                out128_help1[i:i+1, ...],
        #                                                out128_help2[i:i+1, ...]), dim=1)
        #
        #     out128_fusion = self.basicblock128(out128_fusion)
        # out256_main = self.down_256_main(out128_fusion if self.fusion else out128_main)
        #############################

        #############################
        # 256 fusion
        #############################
        B256, C256, H256, W256, D256 = out256_main.shape
        out256_fusion = torch.empty((B256, 3 * C256, H256, W256, D256)).cuda()
        if self.fusion:
            for i in range(cfg.batch_size):
                out256_fusion[i:i+1, ...] = torch.cat((out256_main[i:i+1, ...],
                                                       out256_help1[i:i+1, ...],
                                                       out256_help2[i:i+1, ...]), dim=1)

            out256_fusion = self.basicblock256(out256_fusion)
        #############################

        out_main = self.up_256_main(out256_fusion if self.fusion else out256_main, out128_main)#out128_fusion if self.fusion else out128_main
        out_main = self.up_128_main(out_main, out64_main)#out64_fusion if self.fusion else out64_main
        out_main = self.up_64_main(out_main, out32_main)
        out_main = self.up_32_main(out_main, out16_main)
        #======================================

        out_help1 = self.up_256_help1(out256_help1, out128_help1)
        out_help1 = self.up_128_help1(out_help1, out64_help1)
        out_help1 = self.up_64_help1(out_help1, out32_help1)
        out_help1 = self.up_32_help1(out_help1, out16_help1)
        #======================================

        out_help2 = self.up_256_help2(out256_help2, out128_help2)
        out_help2 = self.up_128_help2(out_help2, out64_help2)
        out_help2 = self.up_64_help2(out_help2, out32_help2)
        out_help2 = self.up_32_help2(out_help2, out16_help2)
        #======================================

        out_main = self.out_tr_main(out_main)
        out_help1 = self.out_tr_help1(out_help1)
        out_help2 = self.out_tr_help2(out_help2)

        return [out_main, out_help1, out_help2]  #torch.cat((out_main, out_help1, out_help2), dim=0)


class SingleSegNet2(nn.Module):
    # TODO: fundamental segmentation framework
    def __init__(self, in_channels, out_channels, fusion=False):
        super(SingleSegNet2, self).__init__()
        self.fusion = fusion
        self.in_tr = InputTransition(in_channels, 16)
        self.down_32 = DownTransition(16, 1)
        self.down_64 = DownTransition(32, 1)
        self.down_128 = DownTransition(64, 2)
        self.down_256 = DownTransition(128, 2)

        self.up_256 = UpTransition(256, 256, 2)
        self.up_128 = UpTransition(256, 128, 2)
        self.up_64 = UpTransition(128, 64, 1)
        self.up_32 = UpTransition(64, 32, 1)

        self.out_tr_main  = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_help1 = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_help2 = OutputTransition(32, out_channels, 'softmax')

        if self.fusion:
            self.basicblock = BasicBlock(256 * 3, 256, kernel_size=3, padding=1)


    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        if self.fusion:
            # split into main, help1, help2
            out256_main = out256[0:cfg.batch_size, ...]
            out256_help1 = out256[cfg.batch_size:2 * cfg.batch_size, ...]
            out256_help2 = out256[2 * cfg.batch_size:3 * cfg.batch_size, ...]

            # build an empty tensor
            B, C, H, W, D = out256_main.shape
            out256_fusion = torch.empty((B, 3 * C, H, W, D)).cuda()

            # concate and conv the main, help1, help2 feature
            for i in range(cfg.batch_size):
                out256_fusion[i:i+1, ...] = torch.cat((out256_main[i:i+1, ...],
                                                       out256_help1[i:i+1, ...],
                                                       out256_help2[i:i+1, ...]), dim=1)
            out256_fusion = self.basicblock(out256_fusion)
            out256 = torch.cat((out256_fusion, out256_help1, out256_help2), dim=0)

        out = self.up_256(out256, out128)
        out = self.up_128(out, out64)
        out = self.up_64(out, out32)
        out = self.up_32(out, out16)

        out_main = out[0:cfg.batch_size, ...]
        out_help1 = out[cfg.batch_size:2*cfg.batch_size,...]
        out_help2 = out[2*cfg.batch_size:3*cfg.batch_size, ...]

        out_main = self.out_tr_main(out_main)
        out_help1 = self.out_tr_help1(out_help1)
        out_help2 = self.out_tr_help2(out_help2)

        return [out_main, out_help1, out_help2] #torch.cat((out_main, out_help1, out_help2), dim=0)


class SingleSegNet3(nn.Module):
    # TODO: fundamental segmentation framework
    def __init__(self, in_channels, out_channels, fusion=False):
        super(SingleSegNet3, self).__init__()
        self.fusion = fusion
        self.in_tr = InputTransition(in_channels, 16)
        self.down_32 = DownTransition(16, 1)
        self.down_64 = DownTransition(32, 1)
        self.down_128 = DownTransition(64, 2)
        self.down_256 = DownTransition(128, 2)

        self.up_256_main = UpTransition(256, 256, 2)
        self.up_128_main = UpTransition(256, 128, 2)
        self.up_64_main = UpTransition(128, 64, 1)
        self.up_32_main = UpTransition(64, 32, 1)

        self.up_256_help1 = UpTransition(256, 256, 2)
        self.up_128_help1 = UpTransition(256, 128, 2)
        self.up_64_help1 = UpTransition(128, 64, 1)
        self.up_32_help1 = UpTransition(64, 32, 1)

        self.up_256_help2 = UpTransition(256, 256, 2)
        self.up_128_help2 = UpTransition(256, 128, 2)
        self.up_64_help2 = UpTransition(128, 64, 1)
        self.up_32_help2 = UpTransition(64, 32, 1)

        self.out_tr_main  = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_help1 = OutputTransition(32, out_channels, 'softmax')
        self.out_tr_help2 = OutputTransition(32, out_channels, 'softmax')

        if self.fusion:
            self.basicblock = BasicBlock(256 * 3, 256, kernel_size=3, padding=1)

    def forward(self, x):

        out16 = self.in_tr(x)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        out256_main = out256[0:cfg.batch_size, ...]
        out256_help1 = out256[cfg.batch_size:2*cfg.batch_size,...]
        out256_help2 = out256[2*cfg.batch_size:3*cfg.batch_size, ...]

        B, C, H, W, D = out256_main.shape
        out256_fusion = torch.empty((B, 3 * C, H, W, D)).cuda()
        if self.fusion:
            for i in range(cfg.batch_size):
                out256_fusion[i:i+1, ...] = torch.cat((out256_main[i:i+1, ...],
                                                       out256_help1[i:i+1, ...],
                                                       out256_help2[i:i+1, ...]), dim=1)
            out256_fusion = self.basicblock(out256_fusion)

        out_main = self.up_256_main(out256_fusion if self.fusion else out256_main, out128[0:cfg.batch_size, ...])
        out_main = self.up_128_main(out_main, out64[0:cfg.batch_size, ...])
        out_main = self.up_64_main(out_main, out32[0:cfg.batch_size, ...])
        out_main = self.up_32_main(out_main, out16[0:cfg.batch_size, ...])

        out_help1 = self.up_256_help1(out256_help1, out128[cfg.batch_size:2*cfg.batch_size,...])
        out_help1 = self.up_128_help1(out_help1, out64[cfg.batch_size:2*cfg.batch_size,...])
        out_help1 = self.up_64_help1(out_help1, out32[cfg.batch_size:2*cfg.batch_size,...])
        out_help1 = self.up_32_help1(out_help1, out16[cfg.batch_size:2*cfg.batch_size,...])

        out_help2 = self.up_256_help2(out256_help2, out128[2*cfg.batch_size:3*cfg.batch_size, ...])
        out_help2 = self.up_128_help2(out_help2, out64[2*cfg.batch_size:3*cfg.batch_size, ...])
        out_help2 = self.up_64_help2(out_help2, out32[2*cfg.batch_size:3*cfg.batch_size, ...])
        out_help2 = self.up_32_help2(out_help2, out16[2*cfg.batch_size:3*cfg.batch_size, ...])

        out_main = self.out_tr_main(out_main)
        out_help1 = self.out_tr_help1(out_help1)
        out_help2 = self.out_tr_help2(out_help2)

        return [out_main, out_help1, out_help2]#torch.cat((out_main, out_help1, out_help2), dim=0)

# ========================================================================================================
# Classes for building up COMPARISON segmentation network
# ========================================================================================================
class conv_block_att(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_att, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(in_ch, out_ch, kernel_size=(3,3,3), stride=(1,1,1), padding=1, bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=(1,1,1), stride=(1,1,1), padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=(1,1,1), stride=(1,1,1), padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=(1,1,1), stride=(1,1,1), padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttUNet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=1, output_ch=4):
        super(AttUNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block_att(img_ch, filters[0])
        self.Conv2 = conv_block_att(filters[0], filters[1])
        self.Conv3 = conv_block_att(filters[1], filters[2])
        self.Conv4 = conv_block_att(filters[2], filters[3])
        self.Conv5 = conv_block_att(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block_att(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block_att(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block_att(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block_att(filters[1], filters[0])

        self.Conv = nn.Conv3d(filters[0], output_ch, kernel_size=(1,1,1), stride=(1,1,1), padding=0)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        out = self.softmax(out)

        return [out]

# ========================================================================================================
# Classes for building up REGISTRATION network
# ========================================================================================================
class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """

        out = self.main(x)
        out = self.activation(out)
        return out


class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding
                              layers
        """
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            # TODO: change between 2 and 8 (segmentation probibality maps)
            prev_nf = 8 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            # TODO: change between 2 and 8 (segmentation probibality maps)
            self.dec.append(conv_block(dim, dec_nf[4] + 8, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
             y = self.dec[i](y)
             y = self.upsample(y)
             y = torch.cat([y, x_enc[-(i+2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
             y = self.upsample(y)
             y = torch.cat([y, x_enc[0]], dim=1)
             y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
             y = self.vm2_conv(y)
        return y


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        self.grid = torch.stack(grids)  # y, x, z
        self.grid = torch.unsqueeze(self.grid, 0)  # add batch
        self.grid = self.grid.type(float).to('cuda') # TODO

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)


class RegNet(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for
    the 2018 implementation of voxelmorph.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(RegNet, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transform = SpatialTransformer(vol_size, mode='bilinear')

    def forward(self, x):# src, tgt # x
        """
        Pass input x through forward once
            :param x input data of moving img and fix img concatenated by torch.cat(dim=1)
        """
        x_1 = self.unet_model(x)
        flow = self.flow(x_1)
        warped = self.spatial_transform(x[:, 0:int(x.size()[1]/2), ...], flow)

        return warped, flow