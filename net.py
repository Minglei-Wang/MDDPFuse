import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ViT as Transformer
from dualselfatt import CAM_Module




class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)
class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5
class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        if kernel_size==3:
            padding = 1
        elif kernel_size==5:
            padding = 2
        elif kernel_size==7:
            padding = 3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)
class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T)) # 垂直边缘特征

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x
class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out

class SpAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class Used(nn.Module):
    def __init__(self, in_channels, out_channels=None, X=3):
        super(Used, self).__init__()
        out_channels = 4*in_channels if out_channels is None else out_channels

        self.dense = DenseBlock(in_channels, kernel_size=X)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)

        x2 = self.sobelconv(x)
        x2 = self.convup(x2)

        return F.leaky_relu(x1 + x2, negative_slope=0.1)


class DenseBlock(nn.Module):
    def __init__(self,channels,kernel_size):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels, kernel_size=kernel_size)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels, kernel_size=kernel_size)
        self.conv3 = ConvLeakyRelu2d(3*channels, channels, kernel_size=kernel_size)
        self.activation = nn.ReLU()
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        x = torch.cat((x, self.conv3(x)), dim=1)

        return self.activation(x)


class RDBX(nn.Module):
    def __init__(self, in_ch, growth_rate, out_channels, kernel_size=3):

        super(RDBX, self).__init__()
        pad = [1, 2, 3]
        if kernel_size == 5:
            pad = [2, 4, 6]
        elif kernel_size == 7:
            pad = [3, 6, 9]
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, kernel_size, padding=pad[0], dilation=1)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, kernel_size, padding=pad[1], dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, kernel_size, padding=pad[2], dilation=3)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, kernel_size, padding=pad[0], dilation=1)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, kernel_size, padding=pad[1], dilation=2)
        in_ch_ += growth_rate
        self.Dcov6 = nn.Conv2d(in_ch_, growth_rate, kernel_size, padding=pad[2], dilation=3)
        in_ch_ += growth_rate
        self.sobelconv = Sobelxy(in_ch)
        self.conv = nn.Conv2d(in_ch_, out_channels, 1, padding=0)
        self.convup = Conv1(in_ch, out_channels)

    def forward(self, x):
        # （1）main branch
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.Dcov6(x5)
        x6 = F.relu(x6)
        x6 = torch.cat([x5, x6], dim=1)

        x7 = self.conv(x6)

        # （2）residual branch
        out = self.sobelconv(x)
        sobel = self.convup(out)

        return F.leaky_relu(x7 + sobel, negative_slope=0.1)

class MSFF(nn.Module):
    def __init__(self, in_channels, r=4):
        super(MSFF, self).__init__()
        outchannels = in_channels // r if in_channels % r == 0 else (in_channels // r) + 1

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear31 = nn.Linear(in_channels, outchannels)
        self.linear32 = nn.Linear(outchannels, in_channels)
        self.linear51 = nn.Linear(in_channels, outchannels)
        self.linear52 = nn.Linear(outchannels, in_channels)
        self.linear71 = nn.Linear(in_channels, outchannels)
        self.linear72 = nn.Linear(outchannels, in_channels)

    def forward(self, x3, x5, x7):
        x3_ct = self.gap(x3)
        x5_ct = self.gap(x5)
        x7_ct = self.gap(x7)

        x3_ct = x3_ct.view(x3_ct.size(0), -1)
        x5_ct = x5_ct.view(x5_ct.size(0), -1)
        x7_ct = x7_ct.view(x7_ct.size(0), -1)

        x3_ct = self.linear31(x3_ct)
        x3_ct = self.linear32(x3_ct)
        x5_ct = self.linear51(x5_ct)
        x5_ct = self.linear52(x5_ct)
        x7_ct = self.linear71(x7_ct)
        x7_ct = self.linear72(x7_ct)

        weight = torch.softmax(torch.cat([x3_ct, x5_ct, x7_ct], dim=1), dim=1)
        x3_softmax, x5_softmax, x7_softmax = torch.chunk(weight, 3, dim=1)

        x3_softmax = x3_softmax.view(x3_softmax.size(0), -1, 1, 1)
        x5_softmax = x5_softmax.view(x5_softmax.size(0), -1, 1, 1)
        x7_softmax = x7_softmax.view(x7_softmax.size(0), -1, 1, 1)

        x3 = x3 * x3_softmax
        x5 = x5 * x5_softmax
        x7 = x7 * x7_softmax

        return x3 + x5 + x7

class GFE(nn.Module):
    def __init__(self, img_zie=(480, 640), ViTmodel = 'vit_base_patch16'):
        super(GFE, self).__init__()
        self.model = Transformer.__dict__[ViTmodel](img_zie)

    def forward(self, x):
        x = self.model(x)
        return x

class CMFF(nn.Module):
    def __init__(self, all_channel, global_and_local=True):
        super(CMFF, self).__init__()

        self.global_and_local = global_and_local
        if global_and_local:
            self.conv_ld1 = BasicConv2d(128, 64, kernel_size=3, padding=1)
            self.conv_ld2 = ConvLeakyRelu2d(64, all_channel, kernel_size=3, padding=1)

        self.conv = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)

        self.sa = SpatialAttention()
        self.cam = CAM_Module(all_channel)

    def forward(self, x, y):
        if self.global_and_local:
            y = self.conv_ld1(y)
            y = self.conv_ld2(y)

        multiplication = x * y
        summation = self.conv(x + y)

        sa = self.sa(multiplication)        # Spatial Attention

        summation_sa = summation.mul(sa)
        sc_feat = self.cam(summation_sa)    # Self-channel attention

        return sc_feat

class SFI(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(SFI, self).__init__()
        self.conv = nn.Conv2d(inchannel*2, inchannel, kernel_size=1)
        self.rconv = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )
        self.ca = ChannelAttention(inchannel)
        self.sa = SpAttention()
        self.rconv0 = nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1)
        self.rbn = nn.BatchNorm2d(inchannel)
        self.convfinal = nn.Conv2d(inchannel, outchannel, kernel_size=1)

    def forward(self, laster, current):
        out1 = torch.cat((laster, current), dim=1)
        out1 = self.conv(out1)
        x1 = laster * out1
        ir1 = current * out1
        f = x1 + ir1
        f = self.rconv(f)
        ca = self.ca(f)         # channel attention
        ca_f = f.mul(ca)
        sa = self.sa(ca_f)      # spatial attention
        sa_f = ca_f.mul(sa)
        f = self.rbn(self.rconv0(sa_f))
        f = f + laster
        f = self.convfinal(f)

        return f












class FusionNet(nn.Module):
    def __init__(self, output=1, img_size=(480, 640), ViTmodel = 'vit_base_patch16'):
        super(FusionNet, self).__init__()
        vis_ch = [8, 16, 24, 32, 48]
        inf_ch = [8, 16, 24, 32, 48]
        growth_rate = [4, 8]
        output = 1
        self.vis_conv1=ConvLeakyRelu2d(1,vis_ch[0])
        self.vis_rgbd1_k3 = RDBX(vis_ch[0], growth_rate[0], vis_ch[1], kernel_size=3)
        self.vis_rgbd2_k3 = RDBX(vis_ch[1], growth_rate[1], vis_ch[2], kernel_size=3)
        self.vis_rgbd1_k5 = RDBX(vis_ch[0], growth_rate[0], vis_ch[1], kernel_size=5)
        self.vis_rgbd2_k5 = RDBX(vis_ch[1], growth_rate[1], vis_ch[2], kernel_size=5)
        self.vis_rgbd1_k7 = RDBX(vis_ch[0], growth_rate[0], vis_ch[1], kernel_size=7)
        self.vis_rgbd2_k7 = RDBX(vis_ch[1], growth_rate[1], vis_ch[2], kernel_size=7)

        self.inf_conv1=ConvLeakyRelu2d(1,vis_ch[0])
        self.inf_rgbd1_k3 = RDBX(inf_ch[0], growth_rate[0], inf_ch[1], kernel_size=3)
        self.inf_rgbd2_k3 = RDBX(inf_ch[1], growth_rate[1], inf_ch[2], kernel_size=3)
        self.inf_rgbd1_k5 = RDBX(inf_ch[0], growth_rate[0], inf_ch[1], kernel_size=5)
        self.inf_rgbd2_k5 = RDBX(inf_ch[1], growth_rate[1], inf_ch[2], kernel_size=5)
        self.inf_rgbd1_k7 = RDBX(inf_ch[0], growth_rate[0], inf_ch[1], kernel_size=7)
        self.inf_rgbd2_k7 = RDBX(inf_ch[1], growth_rate[1], inf_ch[2], kernel_size=7)

        self.MSFF_vis = MSFF(vis_ch[2])
        self.MSFF_inf = MSFF(inf_ch[2])
        self.vis_conv2 = ConvLeakyRelu2d(1, 3)
        self.vis_glb = GFE(img_size, ViTmodel)
        self.inf_conv2 = ConvLeakyRelu2d(1, 3)
        self.inf_glb = GFE(img_size, ViTmodel)

        self.fuse_vis = CMFF(inf_ch[2])
        self.fuse_inf = CMFF(inf_ch[2])
        self.fuse = CMFF(inf_ch[2], global_and_local=False)
        self.decode4 = ConvBnLeakyRelu2d(vis_ch[2], vis_ch[1] + vis_ch[1])
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[1] + inf_ch[1], vis_ch[0] + inf_ch[0])
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[0] + inf_ch[0], vis_ch[0])
        self.decode1 = ConvBnLeakyRelu2d(vis_ch[0], 9)
        self.sim = SFI(9, 9)
        self.output1 = ConvBnLeakyRelu2d(9, vis_ch[0])
        self.output = ConvBnTanh2d(vis_ch[0], output)

    def forward(self, image_vis, image_ir, mask):
        size = image_vis.size()[-1] * image_ir.size()[-2]
        threshold_person = 0.3137
        torch.tensor(threshold_person).to('cuda')
        y_threshold = torch.sum(image_ir >= threshold_person)
        ir_threshold = torch.sum(image_vis >= threshold_person)
        part = (image_vis >= threshold_person) & (image_ir >= threshold_person)
        part = part.sum().item()
        x_vis_origin = image_vis[:, :1]
        x_inf_origin = image_ir

        x_vis_p = self.vis_conv1(x_vis_origin)
        x_vis_p1_k3 = self.vis_rgbd1_k3(x_vis_p)
        x_vis_p2_k3 = self.vis_rgbd2_k3(x_vis_p1_k3)
        x_vis_p1_k5 = self.vis_rgbd1_k5(x_vis_p)
        x_vis_p2_k5 = self.vis_rgbd2_k5(x_vis_p1_k5)
        x_vis_p1_k7 = self.vis_rgbd1_k7(x_vis_p)
        x_vis_p2_k7 = self.vis_rgbd2_k7(x_vis_p1_k7)

        x_inf_p = self.inf_conv1(x_inf_origin)
        x_inf_p1_k3 = self.inf_rgbd1_k3(x_inf_p)
        x_inf_p2_k3 = self.inf_rgbd2_k3(x_inf_p1_k3)
        x_inf_p1_k5 = self.inf_rgbd1_k5(x_inf_p)
        x_inf_p2_k5 = self.inf_rgbd2_k5(x_inf_p1_k5)
        x_inf_p1_k7 = self.inf_rgbd1_k7(x_inf_p)
        x_inf_p2_k7 = self.inf_rgbd2_k7(x_inf_p1_k7)

        x_vis_pf = self.MSFF_vis(x_vis_p2_k3, x_vis_p2_k5, x_vis_p2_k7)
        x_inf_pf = self.MSFF_inf(x_inf_p2_k3, x_inf_p2_k5, x_inf_p2_k7)

        x_vis_g = self.vis_conv2(x_vis_origin)
        x_vis_glb = self.vis_glb(x_vis_g)
        x_inf_g = self.inf_conv2(x_inf_origin)
        x_inf_glb = self.inf_glb(x_inf_g)

        x_vis = self.fuse_vis(x_vis_pf, x_vis_glb)
        x_inf = self.fuse_inf(x_inf_pf, x_inf_glb)
        n = (y_threshold - part) / y_threshold
        m = (ir_threshold - part) / ir_threshold

        if y_threshold / size >= 0.5:
            x = self.fuse(x_vis, m * x_inf)
        else:
            x = self.fuse(n * x_vis, m * x_inf)
        x = self.decode4(x)
        x = self.decode3(x)
        x = self.decode2(x)
        x = self.decode1(x)
        x = self.sim(x, mask)
        x = self.output1(x)
        x = self.output(x)
        return x
