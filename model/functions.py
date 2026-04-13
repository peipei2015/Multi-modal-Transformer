import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from model.HolisticAttention import HA
from torch.autograd import Variable
from torch.distributions import Normal, Independent, kl
import numpy as np
import math

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class CostVolume(nn.Module):
    def __init__(self, max_disp=192, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures
        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolume, self).__init__()

        self.max_disp = max_disp
        self.feature_similarity = feature_similarity

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()

        if self.feature_similarity == 'difference':
            cost_volume = left_feature.new_zeros(b, c, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = left_feature[:, :, :, i:] - right_feature[:, :, :, :-i]
                else:
                    cost_volume[:, :, i, :, :] = left_feature - right_feature

        elif self.feature_similarity == 'concat':
            cost_volume = left_feature.new_zeros(b, 2 * c, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, i:] = torch.cat((left_feature[:, :, :, i:], right_feature[:, :, :, :-i]),
                                                            dim=1)
                else:
                    cost_volume[:, :, i, :, :] = torch.cat((left_feature, right_feature), dim=1)

        elif self.feature_similarity == 'correlation':
            cost_volume = left_feature.new_zeros(b, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                                right_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)

        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()  # [B, C, D, H, W] or [B, D, H, W]

        return cost_volume

class CostVolumeRight(nn.Module):
    def __init__(self, max_disp, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures
        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolumeRight, self).__init__()
        self.max_disp = max_disp
        self.feature_similarity = feature_similarity
        self.stride = 1

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()
        if self.feature_similarity == 'difference':
            cost_volume = right_feature.new_zeros(b, c, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, :-i] = right_feature[:, :, :, :-i] - left_feature[:, :, :, i:]
                else:
                    cost_volume[:, :, i, :, :] = right_feature - left_feature
        elif self.feature_similarity == 'concat':
            cost_volume = right_feature.new_zeros(b, 2 * c, self.max_disp, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, :, i, :, :-i] = torch.cat((right_feature[:, :, :, :-i], left_feature[:, :, :, i:]),
                                                            dim=1)
                else:
                    cost_volume[:, :, i, :, :] = torch.cat((right_feature, left_feature), dim=1)

        elif self.feature_similarity == 'correlation':
            cost_volume = right_feature.new_zeros(b, self.max_disp, h, w)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, :-i] = (right_feature[:, :, :, :-i] *
                                                left_feature[:, :, :, i:]).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)

        elif self.feature_similarity == 'L1':
            cost_volume = torch.zeros((b, self.max_disp, h, w), device='cuda')

            for i in range(0, self.max_disp):
                cost_volume[:, i, :, -i:] = right_feature[:, :, :, -i:].abs().sum(1)
                if i > 0:
                    cost_volume[:, i, :, :-i] = torch.norm(right_feature[:, :, :, :-i] - left_feature[:, :, :, i:], 1, 1)
                else:
                    cost_volume[:, i, :, :] = torch.norm(right_feature[:, :, :, :] - left_feature[:, :, :, :], 1, 1)
        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()  # [B, C, D, H, W] or [B, D, H, W]

        return cost_volume

class CostVolumeBi(nn.Module):
    def __init__(self, max_disp=192, feature_similarity='correlation'):
        """Construct cost volume based on different
        similarity measures
        Args:
            max_disp: max disparity candidate
            feature_similarity: type of similarity measure
        """
        super(CostVolumeBi, self).__init__()

        self.max_disp = max_disp
        self.feature_similarity = feature_similarity
        self.stride = 1
        print('Bi-directional correlation')

    def forward(self, right_feature, left_feature):
        b, c, h, w = left_feature.size()
        if self.feature_similarity == 'correlation':
            cost_volume = right_feature.new_zeros(b, self.max_disp*2, h, w)
            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i, :, i:] = (right_feature[:, :, :, i:] *
                                                left_feature[:, :, :, :-i]).mean(dim=1)
                else:
                    cost_volume[:, i, :, :] = (right_feature * left_feature).mean(dim=1)

            for i in range(self.max_disp):
                if i > 0:
                    cost_volume[:, i+self.max_disp, :, :-i] = (right_feature[:, :, :, :-i] *
                                                left_feature[:, :, :, i:]).mean(dim=1)
                else:
                    cost_volume[:, i+self.max_disp, :, :] = (left_feature * right_feature).mean(dim=1)
        else:
            raise NotImplementedError

        cost_volume = cost_volume.contiguous()  # [B, C, D, H, W] or [B, D, H, W]

        return cost_volume

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
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

class CSAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSAM, self).__init__()
        self.atten_channel = ChannelAttention(in_channels)
        self.atten_spatial = SpatialAttention()
        self.conv_out = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x):
        x = x.mul(self.atten_channel(x))
        x = x.mul(self.atten_spatial(x))
        x = self.conv_out(x)
        return x

class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.atten_spatial = SpatialAttention()

        self.conv1 = nn.Sequential(*[
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        ])

        self.conv2 = nn.Sequential(*[
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        ])

        self.conv3 = nn.Sequential(*[
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        ])

        self.conv4 = nn.Sequential(*[
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        ])

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x1, x2):
        x_rgb = self.conv1(x1)
        x_depth = self.conv2(x2)
        x = x_rgb.mul(self.atten_spatial(x_depth))
        x = self.conv3(x)
        x_depth = self.conv4(x_depth)
        x_out = x1 + x + x_depth

        return x_out

class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels, channels, latent_size, sz):
        super(Mutual_info_reg, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels
        self.sz = sz

        self.fc1_rgb = nn.Linear(channels * 1 * sz * sz, latent_size)
        self.fc2_rgb = nn.Linear(channels * 1 * sz * sz, latent_size)
        self.fc1_depth = nn.Linear(channels * 1 * sz * sz, latent_size)
        self.fc2_depth = nn.Linear(channels * 1 * sz * sz, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.bn1(self.layer1(rgb_feat))))
        depth_feat = self.layer4(self.leakyrelu(self.bn2(self.layer2(depth_feat))))
        # print(rgb_feat.size())
        # print(depth_feat.size())
        CE = torch.nn.BCELoss(reduction='sum')

        rgb_feat = rgb_feat.view(-1, self.channel * 1 * self.sz * self.sz)
        depth_feat = depth_feat.view(-1, self.channel * 1 * self.sz * self.sz)

        mu_rgb = self.fc1_rgb(rgb_feat)
        logvar_rgb = self.fc2_rgb(rgb_feat)
        mu_depth = self.fc1_depth(depth_feat)
        logvar_depth = self.fc2_depth(depth_feat)

        mu_depth = self.tanh(mu_depth)
        mu_rgb = self.tanh(mu_rgb)
        logvar_depth = self.tanh(logvar_depth)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb, scale=torch.exp(logvar_rgb)), 1)
        z_depth = self.reparametrize(mu_depth, logvar_depth)
        dist_depth = Independent(Normal(loc=mu_depth, scale=torch.exp(logvar_depth)), 1)

        bi_di_kld = torch.mean(self.kl_divergence(dist_rgb, dist_depth)) + torch.mean(
            self.kl_divergence(dist_depth, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_depth_norm = torch.sigmoid(z_depth)
        ce_rgb_depth = CE(z_rgb_norm,z_depth_norm.detach())
        ce_depth_rgb = CE(z_depth_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_depth + ce_depth_rgb - bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_depth)).sum()

        return latent_loss, z_rgb, z_depth

from mmengine.model import constant_init, kaiming_init


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class DAM(nn.Module):

    def __init__(self, inplanes, planes):
        super(DAM, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.softmax_channel = nn.Softmax(dim=1)
        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_mul_conv)


    def spatial_pool(self, depth_feature):
        batch, channel, height, width = depth_feature.size()
        input_x = depth_feature
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(depth_feature)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        # context attention
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x, depth_feature):
        # [N, C, 1, 1]
        context = self.spatial_pool(depth_feature)
        # [N, C, 1, 1]
        channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        # channel-wise attention
        out1 = torch.sigmoid(depth_feature * channel_mul_term)
        # fusion
        out = x * out1

        return torch.sigmoid(out)
# BAM
class BAM(nn.Module):
    def __init__(self, in_c):
        super(BAM, self).__init__()
        self.reduce = nn.Conv2d(in_c * 2, in_c, 1)
        self.ff_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.bf_conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb_feat, dep_feat, pred):
        feat = torch.cat((rgb_feat, dep_feat), 1)
        feat = self.reduce(feat)
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        new_feat = torch.cat((ff_feat, bf_feat), 1)
        return new_feat

class SCA(nn.Module):
    def __init__(self, channels):
        super(SCA, self).__init__()

        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())


        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.cross_conv = nn.Conv2d(channels*2, channels, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x3_r,x3_d):
        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        SCA_3_o = x3_r * SCA_ca.expand_as(x3_r)

        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        SCA_3d_o = x3_d * SCA_d_ca.expand_as(x3_d)

        Co_ca3 = torch.softmax(SCA_ca + SCA_d_ca,dim=1)

        SCA_3_co = x3_r * Co_ca3.expand_as(x3_r)
        SCA_3d_co= x3_d * Co_ca3.expand_as(x3_d)

        CR_fea3_rgb = SCA_3_o + SCA_3_co
        CR_fea3_d = SCA_3d_o + SCA_3d_co

        CR_fea3 = torch.cat([CR_fea3_rgb,CR_fea3_d],dim=1)
        CR_fea3 = self.cross_conv(CR_fea3)

        return CR_fea3

class CFC(nn.Module):
    def __init__(self, channel):
        super(CFC, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 1, padding=0)
        self.conv2 = BasicConv2d(channel, channel, 1, padding=0)
        self.conv3 = BasicConv2d(2 * channel, channel, 1, padding=0)
        self.conv4 = BasicConv2d(3 * channel, channel, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, s, d):
        s_conv = self.conv1(s)
        s_conv = self.relu(s_conv)
        d_conv = self.conv2(d)
        d_conv = self.relu(d_conv)
        sd1 = torch.cat((s_conv, d_conv), 1)
        sd2 = self.conv3(sd1)
        sd2 = self.relu(sd2)
        sd_last = torch.cat((sd2, sd1), 1)
        return self.conv4(sd_last)

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super(SelfAttention, self).__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True), # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        B, T, C = x.size()

        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, vert_anchors=11, horz_anchors=11, n_head=8, block_exp=1, n_layer=1,seq_len=1,
                                  embd_pdrop=0.5, attn_pdrop=0.1, resid_pdrop=0.1, n_views=1):
        super(GPT, self).__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.n_views = n_views

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(1, (self.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))

        # velocity embedding
        self.vel_emb = nn.Linear(1, n_embd)
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor):
        raw_image_tensor = image_tensor
        raw_lidar_tensor = lidar_tensor
        image_tensor = F.upsample(image_tensor, size=(self.vert_anchors, self.horz_anchors), mode='bilinear',
                                          align_corners=True)
        lidar_tensor = F.upsample(lidar_tensor, size=(self.vert_anchors, self.horz_anchors), mode='bilinear',
                                  align_corners=True)
        # print(image_tensor.size())
        # print(lidar_tensor.size())
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """

        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.n_views * self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)

        # print(image_tensor.size())
        # print(lidar_tensor.size())

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd)  # (B, an * T, C)

        # project velocity to n_embed
        # [dxli] we don't need veloctiy embeddings anyway.
        # velocity_embeddings = self.vel_emb(velocity.unsqueeze(1)) # (B, C)

        # add (learnable) positional embedding and velocity embedding for all tokens
        x = self.drop(self.pos_emb + token_embeddings)  # + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        # x = self.drop(token_embeddings + velocity_embeddings.unsqueeze(1)) # (B, an * T, C)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)
        x = x.view(bz, (self.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x[:, :self.n_views * self.seq_len, :, :, :].contiguous().view(
            bz * self.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.n_views * self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1,
                                                                                                h, w)
        image_tensor_out = raw_image_tensor + F.upsample(image_tensor_out,
                                                         size=(raw_image_tensor.shape[2], raw_image_tensor.shape[3]),
                                                         mode='bilinear',
                                                         align_corners=True)

        lidar_tensor_out = raw_lidar_tensor + F.upsample(lidar_tensor_out,
                                                         size=(raw_lidar_tensor.shape[2], raw_lidar_tensor.shape[3]),
                                                         mode='bilinear',
                                                         align_corners=True)

        return image_tensor_out, lidar_tensor_out

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

#### BASNet
class RefUnet(nn.Module):
    def __init__(self, inc_ch):
        super(RefUnet, self).__init__()
        self.conv_x = nn.Conv2d(1, 8, 3, padding=1)
        # self.conv_rgb = nn.Conv2d(3, 8, 3, padding=1)
        self.conv_depth = nn.Conv2d(1, 8, 3, padding=1)
        self.conv0 = nn.Conv2d(8*2, inc_ch, 3,padding=1)
        self.conv00 = nn.Conv2d(8, inc_ch, 3, padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x, x_depth=None):
        if x_depth is None:
            hx = self.conv_x(x)
            hx = self.conv00(hx)
        else:
            hx = torch.cat((self.conv_x(x), self.conv_depth(x_depth)), dim=1)
            hx = self.conv0(hx)
        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)
        return x + residual
#### BASNet
class Three_RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(Three_RefUnet, self).__init__()
        self.conv_x = nn.Conv2d(1, 8, 3, padding=1)
        self.conv_rgb = nn.Conv2d(3, 8, 3, padding=1)
        self.conv_depth = nn.Conv2d(1, 8, 3, padding=1)


        self.conv0 = nn.Conv2d(8*3,inc_ch,3,padding=1)

        self.conv1 = nn.Conv2d(inc_ch,64,3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(64,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(64,64,3,padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64,64,3,padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128,64,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self,x, x_depth, x_rgb):

        hx = torch.cat((self.conv_x(x), self.conv_depth(x_depth), self.conv_rgb(x_rgb)), dim=1)
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx,hx4),1))))
        hx = self.upscore2(d4)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual
class SCPC(nn.Module):
    def __init__(self, in_c, dilation_list):
        super(SCPC, self).__init__()
        self.conv11 = nn.Sequential(*[
            nn.Conv2d(in_c, in_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        ])
        self.conv11_2 = nn.Sequential(*[
            nn.Conv2d(in_c, in_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        ])
        channels = int(in_c/4)
        self.atrous_conv4 = nn.Sequential(*[
            nn.Conv2d(channels, channels, 3, padding=dilation_list[3], dilation=dilation_list[3], bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ])
        self.atrous_conv3 = nn.Sequential(*[
            nn.Conv2d(channels, channels, 3, padding=dilation_list[2], dilation=dilation_list[2], bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ])
        self.atrous_conv2 = nn.Sequential(*[
            nn.Conv2d(channels, channels, 3, padding=dilation_list[1], dilation=dilation_list[1], bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ])
        self.atrous_conv1 = nn.Sequential(*[
            nn.Conv2d(channels, channels, 3, padding=dilation_list[0], dilation=dilation_list[0], bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        ])

    def forward(self, x):
        x1 = self.conv11(x)
        # split into 4 groups
        _,c,_,_ = x.size()
        s1 = int(c/4)
        s2 = int(c/4*2)
        s3 = int(c/4*3)
        x2_1 = x1[:,:s1]
        x2_2 = x1[:,s1:s2]
        x2_3 = x1[:,s2:s3]
        x2_4 = x1[:,s3:]

        #
        x3_1 = self.atrous_conv1(x2_1)
        x3_2 = self.atrous_conv2(x2_2 + x3_1)
        x3_3 = self.atrous_conv3(x2_3 + x3_2)
        x3_4 = self.atrous_conv3(x2_4 + x3_3)
        x4 = torch.cat((x3_1, x3_2, x3_3, x3_4), dim=1)
        x4 = self.conv11_2(x4)+x
        return x4

class DownSample(nn.Module):
    def __init__(self, in_c):
        super(DownSample, self).__init__()
        self.down1 = nn.Sequential(*[
            nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.down2 = nn.Sequential(*[
            nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.conv11_1 = nn.Sequential(*[
            nn.Conv2d(in_c, in_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        ])
        self.conv11_2 = nn.Sequential(*[
            nn.Conv2d(in_c, in_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        ])
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.scpc = SCPC(in_c, dilation_list=[1,2,4,8])
        self.scpc2 = SCPC(2*in_c, dilation_list=[1,2,4,8])

        self.conv_final = nn.Sequential(*[
            nn.Conv2d(2*in_c, in_c, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        ])
    def forward(self, x):
        x1 = self.down1(x)
        bs1, c1, h1, w1 = x1.size()
        x2 = self.down2(x1)
        bs2, c2, h2, w2 = x2.size()
        x3 = self.gap(x2)
        x3 = self.sigmoid(x3)

        x1_1 = x1*x3

        x2_1 = x2*x3

        x2_2 = self.up(self.conv11_1(self.scpc(x2_1)))

        if x2_2.shape[2] != x1_1.shape[2] or x2_2.shape[3] != x1_1.shape[3]:
            x2_2 = F.upsample(x2_2, size=(h1, w1), mode='bilinear')

        y = self.scpc2(torch.cat((self.conv11_2(x1_1),x2_2), dim=1))
        return self.conv_final(y)

## ABMDRNET: adaptive-weighted bi-directional modality difference reduction ...
class ChannelWeightedFusion(nn.Module):
    def __init__(self, in_c):
        super(ChannelWeightedFusion, self).__init__()
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(2*in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c)
        ])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        #
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        w = self.gap(x)
        x1_2 = x1*w
        x2_2 = x2*(1-w)
        x = x1_2+x2_2
        return x

class ChannelSpatialWeightedFusion(nn.Module):
    def __init__(self, in_c):
        super(ChannelSpatialWeightedFusion, self).__init__()
        self.conv1 = nn.Sequential(*[
            nn.Conv2d(2*in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        ])
        self.conv2_1 = nn.Sequential(*[
            nn.Conv2d(2 * in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        ])
        self.conv2_2 = nn.Sequential(*[
            nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        ])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        #channel weighted
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        w = self.gap(x)
        x_cw = x1*w+x2*(1-w)

        # spatial weighted
        x = torch.cat((x1, x2), dim=1)
        x = self.conv2_1(x)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv2_2(x)
        w = self.sigmoid(x)
        x_sw = x1*w+x2*(1-w)
        return x_cw+x_sw

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
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

class DisparityRegression(nn.Module):
    def __init__(self, start, end, stride=1):
        super(DisparityRegression, self).__init__()
        self.disp = torch.arange(start*stride, end*stride, stride, device='cuda', requires_grad=False).view(1, -1, 1, 1).float()

    def forward(self, x):
        ### compute the corresponding disparity map.
        x = F.softmax(-x, dim=1)
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, keepdim=True)
        return out

class SpatialContextModule(nn.Module):
    def __init__(self, channel):
        super(SpatialContextModule, self).__init__()

        self.aspp = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, channel)
        # self.conv = nn.Conv2d(channel, channel, 3, 1, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)
    def normalize(self, x):
        b = x.shape[0]
        for i in range(b):
            t = x[i,:,:]
            t = t-t.min()
            t = t/(t.max()+1e-8)
            x[i, :, :] = t
        return x
    def forward(self, x):
        b, c, h, w = x.shape
        x2 = self.aspp(x)
        x2 = x2.permute(0, 2, 3, 1)
        x2 = x2.view(b, -1, c)
        #print('x2', x2.shape)
        m_ss = torch.matmul(x2, x2.permute(0,2,1))
        #print('m_ss', m_ss.shape)

        x3 = x.permute(0, 2, 3, 1)
        x3 = x3.view(b, -1, c)
        #print('x3', x3.shape)
        m_cs = torch.matmul(x3, x3.permute(0, 2, 1))
        #print('m_cs', m_cs.shape)

        m = m_ss+m_cs
        # print('m', m.shape)
        m = self.softmax(m)

        x4 = torch.matmul(m, x2)
        #print('x4',x4.shape)
        x4 = x4.permute(0,2,1)
        x4 = x4.view(b, c, h, w)

        return x+x4

class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
#aggregation of the high-level(teacher) features
class aggregation_init(nn.Module):

    def __init__(self, channel):
        super(aggregation_init, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x
#aggregation of the low-level(student) features
class aggregation_final(nn.Module):

    def __init__(self, channel):
        super(aggregation_final, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x1)) \
               * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        return x3_2

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class Refine(nn.Module):
    def __init__(self):
        super(Refine,self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, attention,x1,x2,x3):
        #Note that there is an error in the manuscript. In the paper, the refinement strategy is depicted as ""f'=f*S1"", it should be ""f'=f+f*S1"".
        x1 = x1+torch.mul(x1, self.upsample2(attention))
        x2 = x2+torch.mul(x2,self.upsample2(attention))
        x3 = x3+torch.mul(x3,attention)

        return x1,x2,x3
class Pred_decoder_bbsnet(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Pred_decoder_bbsnet, self).__init__()
        # Decoder 1
        self.rfb2_1 = GCM(256, channel)
        self.rfb3_1 = GCM(512, channel)
        self.rfb4_1 = GCM(512, channel)
        self.agg1 = aggregation_init(channel)

        # Decoder 2
        self.rfb0_2 = GCM(64, channel)
        self.rfb1_2 = GCM(128, channel)
        self.rfb2_2 = GCM(256, channel)
        self.agg2 = aggregation_final(channel)
        self.HA = Refine()

        # Components of PTM module
        self.inplanes = 32 * 2
        self.deconv1 = self._make_transpose(TransBasicBlock, 32 * 2, 3, stride=2)
        self.inplanes = 32
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.agant1 = self._make_agant_layer(32 * 3, 32 * 2)
        self.agant2 = self._make_agant_layer(32 * 2, 32)
        self.out0_conv = nn.Conv2d(32 * 3, 1, kernel_size=1, stride=1, bias=True)
        self.out1_conv = nn.Conv2d(32 * 2, 1, kernel_size=1, stride=1, bias=True)
        self.out2_conv = nn.Conv2d(32 * 1, 1, kernel_size=1, stride=1, bias=True)
    def forward(self, x0, x1, x2, x3, x4):
        # produce initial saliency map by decoder1
        x2_1 = self.rfb2_1(x2)
        x3_1 = self.rfb3_1(x3)
        x4_1 = self.rfb4_1(x4)
        attention_map = self.agg1(x4_1, x3_1, x2_1)

        # Refine low-layer features by initial map
        x0_1, x1_1, x2_1 = self.HA(attention_map.sigmoid(), x0, x1, x2)

        # produce final saliency map by decoder2
        x0_2 = self.rfb0_2(x0_1)
        x1_2 = self.rfb1_2(x1_1)
        x2_2 = self.rfb2_2(x2_1)
        y = self.agg2(x2_2, x1_2, x0_2)  # *4

        # PTM module
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)

        return attention_map, y
    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

class TransBlock(nn.Module):
    def __init__(self, n_embd, vert_anchors=11, horz_anchors=11, n_head=8, block_exp=1, n_layer=1,seq_len=1,
                                  embd_pdrop=0.5, attn_pdrop=0.1, resid_pdrop=0.1, n_views=1):
        super(TransBlock, self).__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.n_views = n_views

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(1, (self.n_views + 1) * seq_len * vert_anchors * horz_anchors, n_embd))

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, lidar_tensor):
        raw_image_tensor = image_tensor
        raw_lidar_tensor = lidar_tensor
        image_tensor = F.upsample(image_tensor, size=(self.vert_anchors, self.horz_anchors), mode='bilinear',
                                          align_corners=True)
        lidar_tensor = F.upsample(lidar_tensor, size=(self.vert_anchors, self.horz_anchors), mode='bilinear',
                                  align_corners=True)
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
        """

        bz = lidar_tensor.shape[0] // self.seq_len
        h, w = lidar_tensor.shape[2:4]

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.n_views * self.seq_len, -1, h, w)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        token_embeddings = torch.cat([image_tensor, lidar_tensor], dim=1).permute(0, 1, 3, 4, 2).contiguous()
        token_embeddings = token_embeddings.view(bz, -1, self.n_embd)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)
        x = x.view(bz, (self.n_views + 1) * self.seq_len, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # same as token_embeddings

        image_tensor_out = x[:, :self.n_views * self.seq_len, :, :, :].contiguous().view(
            bz * self.n_views * self.seq_len, -1, h, w)
        lidar_tensor_out = x[:, self.n_views * self.seq_len:, :, :, :].contiguous().view(bz * self.seq_len, -1,
                                                                                                h, w)
        image_tensor_out = raw_image_tensor + F.upsample(image_tensor_out,
                                                         size=(raw_image_tensor.shape[2], raw_image_tensor.shape[3]),
                                                         mode='bilinear',
                                                         align_corners=True)

        lidar_tensor_out = raw_lidar_tensor + F.upsample(lidar_tensor_out,
                                                         size=(raw_lidar_tensor.shape[2], raw_lidar_tensor.shape[3]),
                                                         mode='bilinear',
                                                         align_corners=True)

        return image_tensor_out, lidar_tensor_out