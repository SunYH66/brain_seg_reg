# --coding:utf-8--
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums)

    def forward(self, x):
        out = self.activate1(self.bn1(self.down(x)))
        # out = self.residual(out)
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
        # self.residual = ResidualBlock(out_channels, out_channels, 3, 1, nums)
        self.basicblock = BasicBlock(out_channels, out_channels, 3, 1)

    def forward(self, x, skip_x):
        out = self.activate(self.bn(self.conv1(x)))
        out = torch.cat((out, skip_x), 1)
        # out = self.residual(out)
        out = self.basicblock(out)

        return out


def conv3x3(in_channels, out_channels, stride=(1,1,1)):
    """Only change the channel number."""
    return nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,3),
                     stride=stride, padding=1, bias=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out

class NoAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self):
        super().__init__()

    def forward(self, v, k, q, mask=None):
        output = v
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)
        return output


class ScaledDotProductAttentionOnly(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)


        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)

        #Reshape output to original format
        output  = output.view(b, c, h, w)
        return output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module for Hyperspectral Pansharpening (Image Fusion) '''

    def __init__(self, n_head, in_pixels, linear_dim, num_features):
        super().__init__()
        # Parameters
        self.n_head = n_head  # No of heads
        self.in_pixels = in_pixels  # No of pixels in the input image
        self.linear_dim = linear_dim  # Dim of linear-layer (outputs)

        # Linear layers
        self.w_qs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for queries
        self.w_ks = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for keys
        self.w_vs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)  # Linear layer for values
        self.fc = nn.Linear(n_head * linear_dim, in_pixels, bias=False)  # Final fully connected layer

        # Scaled dot product attention
        self.attention = ScaledDotProductAttention(temperature=in_pixels ** 0.5)

        # Batch normalization layer
        self.OutBN = nn.BatchNorm3d(num_features=num_features)

    def forward(self, v, k, q, mask=None):
        # Reshaping matrixes to 2D
        # q = b, c_q, h*w*d
        # k = b, c_k, h*w*d
        # v = b, c_v, h*w*d
        b, c, h, w, d = q.size(0), q.size(1), q.size(2), q.size(3), q.size(4)
        n_head = self.n_head
        linear_dim = self.linear_dim

        # Reshaping K, Q, and Vs...
        q = q.view(b, c, h * w * d)
        k = k.view(b, c, h * w * d)
        v = v.view(b, c, h * w * d)

        # Save V
        output = v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv

        q = self.w_qs(q).view(b, c, n_head, linear_dim)
        k = self.w_ks(k).view(b, c, n_head, linear_dim)
        v = self.w_vs(v).view(b, c, n_head, linear_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        # Computing ScaledDotProduct attention for each head
        v_attn = self.attention(v, k, q, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        v_attn = self.fc(v_attn)

        output = output + v_attn
        # output  = v_attn

        # Reshape output to original image format
        output = output.view(b, c, h, w, d)

        # We can consider batch-normalization here,,,
        # Will complete it later
        output = self.OutBN(output)
        return output


# Extract feature encoder
class SFE(nn.Module):
    def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
        super(SFE, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.conv_head = conv3x3(in_feats, n_feats)

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks):
            self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
                                     res_scale=res_scale))

        self.conv_tail = conv3x3(n_feats, n_feats)

    def forward(self, x):
        x = F.relu(self.conv_head(x))
        x1 = x
        for i in range(self.num_res_blocks):
            x = self.RBs[i](x)
        x = self.conv_tail(x)
        x = x + x1
        return x


# Extract feature encoder
class LFE(nn.Module):
    # TODO: fundamental segmentation framework
    def __init__(self, in_channels, fusion=False):
        super(LFE, self).__init__()
        self.fusion = fusion
        self.in_tr = InputTransition(in_channels, 4)
        self.down_32 = DownTransition(4, 1)
        self.down_64 = DownTransition(8, 1)
        self.down_128 = DownTransition(16, 2)
        self.down_256 = DownTransition(32, 2)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)
        out256 = self.down_256(out128)

        return out16, out32, out64, out128, out256


# Experimenting with soft attention
class HyperTransformer(nn.Module):
    def __init__(self):
        super(HyperTransformer, self).__init__()
        # Settings
        self.in_channels = 1
        self.out_channels =4

        # Parameter setup
        self.num_res_blocks = [4, 4, 4, 4, 4]
        self.n_feats = 64
        self.res_scale = 1

        # Feature-06 image & Feature-12 image
        self.LFE = LFE(in_channels=1)

        self.SFE = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        n_head = 4

        ### Multi-Head Attention ###
        self.TS_lv3 = MultiHeadAttention(n_head=int(n_head),
                                         in_pixels=int(8 * 8 * 6),
                                         linear_dim=int(16),
                                         num_features=64)
        self.TS_lv2 = MultiHeadAttention(n_head=int(n_head),
                                         in_pixels=int(16 * 16 * 12),
                                         linear_dim=int(16),
                                         num_features=32)
        self.TS_lv1 = MultiHeadAttention(n_head=int(n_head),
                                         in_pixels=int(32 * 32 * 24),
                                         linear_dim=int(16),
                                         num_features=16)

        # elif cfg.phase == 'val' or cfg.phase == 'test':
        #     ### Multi-Head Attention ###
        #     self.TS_lv3 = MultiHeadAttention(n_head=int(n_head),
        #                                      in_pixels=int(12 * 14 * 12),
        #                                      linear_dim=int(4),
        #                                      num_features=128)
        #     self.TS_lv2 = MultiHeadAttention(n_head=int(n_head),
        #                                      in_pixels=int(24 * 28 * 24),
        #                                      linear_dim=int(4),
        #                                      num_features=64)
        #     self.TS_lv1 = MultiHeadAttention(n_head=int(n_head),
        #                                      in_pixels=int(48 * 56 * 48),
        #                                      linear_dim=int(4),
        #                                      num_features=32)

        ###############
        ### stage11 ###
        ###############

        self.conv11_head = conv3x3(2 * self.n_feats, self.n_feats)

        self.conv12 = conv3x3(self.n_feats, int(self.n_feats / 2))
        self.ps12 = nn.Upsample(scale_factor=2)
        # Residial blocks
        self.RB11 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
                                      res_scale=self.res_scale))
        self.conv11_tail = conv3x3(self.n_feats, self.n_feats)

        ###############
        ### stage22 ###
        ###############

        self.conv22_head = conv3x3(2 * int(self.n_feats / 2), int(self.n_feats / 2))

        self.conv23 = conv3x3(int(self.n_feats / 2), int(self.n_feats / 4))
        self.ps23 = nn.Upsample(scale_factor=2)
        # Residual blocks
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB22.append(ResBlock(in_channels=int(self.n_feats / 2), out_channels=int(self.n_feats / 2),
                                      res_scale=self.res_scale))
        self.conv22_tail = conv3x3(int(self.n_feats / 2), int(self.n_feats / 2))

        ###############
        ### stage33 ###
        ###############

        self.conv33_head = conv3x3(2 * int(self.n_feats / 4), int(self.n_feats / 4))

        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB33.append(ResBlock(in_channels=int(self.n_feats / 4), out_channels=int(self.n_feats / 4),
                                      res_scale=self.res_scale))
        self.conv33_tail = conv3x3(int(self.n_feats / 4), int(self.n_feats / 4))

        ##############
        ### FINAL ####
        ##############
        """
        self.final_conv = nn.Conv3d(in_channels=self.n_feats + int(self.n_feats / 2) + int(self.n_feats / 4),
                                    out_channels=self.out_channels, kernel_size=(3,3,3), padding=1)
        self.RBF = nn.ModuleList()
        for i in range(self.num_res_blocks[4]):
            self.RBF.append(
                ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.convF_tail = conv3x3(self.out_channels, self.out_channels)
        """

        # self.up_conv13 = nn.ConvTranspose3d(16, 8, kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=1)
        # self.bn_13 = nn.BatchNorm3d(8)
        # self.activate = nn.ReLU(inplace=True)
        # self.up_conv23 = nn.ConvTranspose3d(8, 4, kernel_size=(2, 2, 2), stride=(2, 2, 2), groups=1)
        # self.bn_23 = nn.BatchNorm3d(4)
        # self.activate = nn.ReLU(inplace=True)

        self.up_64 = UpTransition(16, 16, 1)
        self.up_32 = UpTransition(16, 8, 1)
        self.final_conv = OutputTransition(8, 4, 'softmax')

        ###############
        # Batch Norm ##
        ###############
        self.BN_x11 = nn.BatchNorm3d(64)
        self.BN_x22 = nn.BatchNorm3d(32)
        self.BN_x33 = nn.BatchNorm3d(16)


    def forward(self, x):
        # Extracting T and S at multiple-scales
        # lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        Q_16, Q_32, Q_lv1, Q_lv2, Q_lv3 = self.LFE(x[:,0:1, ...])
        _, _, K_lv1, K_lv2, K_lv3 = self.LFE(x[:,1:2, ...])
        _, _, V_lv1, V_lv2, V_lv3 = self.LFE(x[:,2:3, ...])


        T_lv3 = self.TS_lv3(V_lv3, K_lv3, Q_lv3)
        T_lv2 = self.TS_lv2(V_lv2, K_lv2, Q_lv2)
        T_lv1 = self.TS_lv1(V_lv1, K_lv1, Q_lv1)

        # Save feature maps for illustration purpose
        # feature_dic={}
        # feature_dic.update({"V": V_lv3.detach().cpu().numpy()})
        # feature_dic.update({"K": K_lv3.detach().cpu().numpy()})
        # feature_dic.update({"Q": Q_lv3.detach().cpu().numpy()})
        # feature_dic.update({"T": T_lv3.detach().cpu().numpy()})
        # savemat("/home/lidan/Dropbox/Hyperspectral/HyperTransformer/feature_visualization_pavia/soft_attention/multi_head_no_skip_lv3.mat", feature_dic)
        # exit()

        # Shallow Feature Extraction (SFE)
        # x = self.SFE(x[0:cfg.batch_size, ...])
        x = Q_lv3

        #####################################
        #### stage1: (L/4, W/4) scale ######
        #####################################
        x11 = x
        # HyperTransformer at (L/4, W/4) scale
        x11_res = x11

        x11_res = torch.cat((self.BN_x11(x11_res), T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res)  # F.relu(self.conv11_head(x11_res))

        x11 =x11_res
        # Residial learning
        x11_res = x11

        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11_res
        # print(x11.shape)
        #####################################
        #### stage2: (L/2, W/2) scale ######
        #####################################
        x22 = self.conv12(x11)

        x22 = F.relu(self.ps12(x22))
        # HyperTransformer at (L/2, W/2) scale
        x22_res = x22

        x22_res = torch.cat((self.BN_x22(x22_res), T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res)  # F.relu(self.conv22_head(x22_res))

        x22 = x22_res
        # Residial learning
        x22_res = x22
        for i in range(self.num_res_blocks[2]):
            x22_res = self.RB22[i](x22_res)
        x22_res = self.conv22_tail(x22_res)
        x22 = x22_res
        # print(x22.shape)
        #####################################
        ###### stage3: (L, W) scale ########
        #####################################
        x33 = self.conv23(x22)

        x33 = F.relu(self.ps23(x33))
        # HyperTransformer at (L, W) scale
        x33_res = x33

        x33_res = torch.cat((self.BN_x33(x33_res), T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res)  # F.relu(self.conv33_head(x33_res))

        x33 = x33_res
        # Residual learning
        x33_res = x33
        for i in range(self.num_res_blocks[3]):
            x33_res = self.RB33[i](x33_res)
        x33_res = self.conv33_tail(x33_res)
        x33 = x33_res
        # print(x33.shape)
        #####################################
        ############ Feature Pyramid ########
        #####################################
        # x11_up = F.interpolate(x11, scale_factor=4, mode='trilinear')
        # x22_up = F.interpolate(x22, scale_factor=2, mode='trilinear')
        # xF = torch.cat((x11_up, x22_up, x33), dim=1)


        #####################################
        ####  Final convolution   ###########
        #####################################
        # xF = self.activate(self.bn_13(self.up_conv13(x33)))
        # xF = self.activate(self.bn_23(self.up_conv23(xF)))

        xF = self.up_64(x33, Q_32)
        xF = self.up_32(xF, Q_16)
        xF = self.final_conv(xF)

        return [xF]