import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import functools
import operator
from typing import List, Optional

import networks.blocks as blocks
from networks.resnet import ResNet18
from networks.rgb_depth_fusion import SqueezeAndExciteFusionAdd, ExciteFusionAdd, ResidualExciteFusion, SelfAttentionFusion, ResidualAttentionFusion
try:
    import MinkowskiEngine as ME  # optional dependency
    _HAS_MINKOWSKI = True
except Exception:
    ME = None
    _HAS_MINKOWSKI = False

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Encoder, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderPooled(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderPooled, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderBig(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderBig, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

class EncoderBigPooled(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(EncoderBigPooled, self).__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        # convolutions
        encoder = [
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            nn.Flatten()
            ]

        self.encoder = nn.Sequential(*encoder)
        self.num_features = functools.reduce(operator.mul, list(self.encoder(torch.rand(1, *input_dim)).shape))

        # linears
        fcn = [
            nn.Linear(self.num_features, self.latent_size, bias=True)
        ]

        encoder += fcn

        self.encoder = nn.Sequential(*encoder)

    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()

    
class DoubleEncoder(nn.Module):
    def __init__(self,
                out_channels,
                encoder_block='BasicBlock',
                bottleneck_dim=32,
                fuse_depth_in_rgb_encoder='SE-add',
                size = 256
                ):

        super(DoubleEncoder, self).__init__()
        self.latent_size = out_channels
        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder
        self.activation = nn.ReLU(inplace=True)
        self.encoder_rgb = ResNet18(
            block=encoder_block,
            pretrained_on_imagenet=False,
            activation=self.activation)
        self.encoder_depth = ResNet18(
            block=encoder_block,
            pretrained_on_imagenet=False,
            activation=self.activation,
            input_channels=1)

        h0, w0 = size // 2, size // 2
        h1, w1 = h0 // 2, w0 // 2
        h2, w2 = h1 // 2, w1 // 2
        h3, w3 = h2 // 2, w2 // 2
        h4, w4 = h3 // 2, w3 // 2
        patch_sizes = [np.gcd(h4, w4)] * 5

        if self.fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        elif self.fuse_depth_in_rgb_encoder == 'SelfAttention':
            self.se_layer0 = SelfAttentionFusion(patch_sizes[0], self.encoder_rgb.down_2_channels_out, bottleneck_dim)
            self.se_layer1 = SelfAttentionFusion(patch_sizes[1], self.encoder_rgb.down_4_channels_out, bottleneck_dim)
            self.se_layer2 = SelfAttentionFusion(patch_sizes[2], self.encoder_rgb.down_8_channels_out, bottleneck_dim)
            self.se_layer3 = SelfAttentionFusion(patch_sizes[3], self.encoder_rgb.down_16_channels_out, bottleneck_dim)
            self.se_layer4 = SelfAttentionFusion(patch_sizes[4], self.encoder_rgb.down_32_channels_out, bottleneck_dim)
        elif self.fuse_depth_in_rgb_encoder == 'ResidualAttention':
            self.se_layer0 = ResidualAttentionFusion(patch_sizes[0], self.encoder_rgb.down_2_channels_out, bottleneck_dim)
            self.se_layer1 = ResidualAttentionFusion(patch_sizes[1], self.encoder_rgb.down_4_channels_out, bottleneck_dim)
            self.se_layer2 = ResidualAttentionFusion(patch_sizes[2], self.encoder_rgb.down_8_channels_out, bottleneck_dim)
            self.se_layer3 = ResidualAttentionFusion(patch_sizes[3], self.encoder_rgb.down_16_channels_out, bottleneck_dim)
            self.se_layer4 = ResidualAttentionFusion(patch_sizes[4], self.encoder_rgb.down_32_channels_out, bottleneck_dim)
        elif self.fuse_depth_in_rgb_encoder == 'excite':
            self.se_layer0 = ExciteFusionAdd(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = ExciteFusionAdd(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = ExciteFusionAdd(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = ExciteFusionAdd(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = ExciteFusionAdd(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        elif self.fuse_depth_in_rgb_encoder == 'ResidualExcite':
            self.se_layer0 = ResidualExciteFusion(self.encoder_rgb.down_2_channels_out, activation=self.activation)
            self.se_layer1 = ResidualExciteFusion(self.encoder_rgb.down_4_channels_out, activation=self.activation)
            self.se_layer2 = ResidualExciteFusion(self.encoder_rgb.down_8_channels_out, activation=self.activation)
            self.se_layer3 = ResidualExciteFusion(self.encoder_rgb.down_16_channels_out, activation=self.activation)
            self.se_layer4 = ResidualExciteFusion(self.encoder_rgb.down_32_channels_out, activation=self.activation)
        else:
            if self.fuse_depth_in_rgb_encoder != 'add':
                print('WARNING! You passed an invalid RGB + D fusion. Sum will be used!')
                self.fuse_depth_in_rgb_encoder == 'add'

        ## TODO this is hardcoded for the moment, I'm thinking about how to compute it based on the image size
        self.conv_last = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu_last = nn.ReLU()
        self.maxpool_last = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_last = nn.Linear(64 * 5 * 5, self.latent_size)  # Adjust the input size according to your specific case
        # self.last_layer = nn.Conv2d(512,self.latent_size,1,8).cuda()

    def forward(self, rgbd):

        rgb_input = rgbd[:,:-1,:,:]
        depth_input = rgbd[:,-1,:,:].unsqueeze(1)
        rgb = self.encoder_rgb.forward_first_conv(rgb_input)
        depth = self.encoder_depth.forward_first_conv(depth_input)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer0(rgb, depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer1(rgb, depth)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer2(rgb, depth)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer3(rgb, depth)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer4(rgb, depth)

        out = self.conv_last(fuse)
        out = self.relu_last(out)
        out = self.maxpool_last(out)
        out = out.view(out.size(0), -1)  # Flatten the output of conv layers
        out = self.fc_last(out)
        return out
    
        # out = self.last_layer(fuse)
        # return out.squeeze() # TODO remove this ugly stuff

class ERFNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()

        self.latent_size = out_channels
        self.size = size
        
        input_dim = (in_channels, size, size)

        self.initial_block = blocks.DownsamplerBlock(in_channels,16)

        self.layers = [self.initial_block]

        self.layers.append(blocks.DownsamplerBlock(16,64))

        # for x in range(0, 1):    #2 times
        #    self.layers.append(blocks.non_bottleneck_1d(64, 0.03, 1)) 

        for x in range(0, 1):    #2 times
            self.layers.append(blocks.non_bottleneck_1d(64, 0.3, 2))
            self.layers.append(blocks.non_bottleneck_1d(64, 0.3, 4))

        # self.layers.append(blocks.DownsamplerBlock(64,128))
        # for x in range(0, 1):    #2 times
        #     self.layers.append(blocks.non_bottleneck_1d(128, 0.3, 8))
        #     self.layers.append(blocks.non_bottleneck_1d(128, 0.3, 16))

        self.layers.append(nn.Flatten())
        
        encoder = nn.Sequential(*self.layers)
        self.num_features = functools.reduce(operator.mul, list(encoder(torch.rand(1, *input_dim)).shape))

        self.layers.append(nn.Linear(self.num_features, self.latent_size, bias=True))
        self.encoder = nn.Sequential(*self.layers)
    
    def forward(self, input):
        batch_size = input.shape[0]
        out = self.encoder(input)
        out = out.reshape(batch_size, self.latent_size)  # --> maybe not needed
        return out.float()


## thanks to: https://github.com/cihanongun/Point-Cloud-Autoencoder
class PointCloudEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointCloudEncoder, self).__init__()
        
        self.latent_size = out_channels

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, self.latent_size)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Regularization: Dropout
        self.dropout = nn.Dropout(p=0.3)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first FC
        x = self.fc2(x)
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        return x


class PointCloudEncoderMaxMean(nn.Module):
    """PointCloudEncoder with Max + Mean pooling for occlusion-aware features.

    Concatenates global max and mean pooling over points, doubling the
    aggregated feature dimension (512 -> 1024) before the FC head.  Mean
    pooling captures the distribution of visible points, making the latent
    sensitive to which regions of the point cloud are observed.

    Conv layers are architecturally identical to PointCloudEncoder so that
    pretrained Conv weights can be loaded directly.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.latent_size = out_channels

        # Same conv backbone as PointCloudEncoder
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # FC head: 1024 (512 max + 512 mean) -> 256 -> latent_size
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, self.latent_size)
        self.dropout = nn.Dropout(p=0.3)

    def encode_features(self, x):
        """Return post-pooling 1024d feature (before FC head)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x_max = torch.max(x, 2)[0]
        x_mean = torch.mean(x, 2)
        return torch.cat([x_max, x_mean], dim=1)  # (B, 1024)

    def encoder(self, x):
        # x: (B, C, N) where N = number of points
        x = self.encode_features(x)  # (B, 1024)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        return self.encoder(x)


class PointCloudEncoderMaxMeanMasked(nn.Module):
    """PointCloudEncoderMaxMean with mask-aware pooling.

    Same conv backbone as ``PointCloudEncoderMaxMean`` (3->64->128->256->512),
    but mean-pool averages only over real (non-padded) points using a per-sample
    boolean mask. Max-pool sets padded positions to -inf so they don't win.
    Falls back to unmasked max+mean when mask is None.
    """

    def __init__(self, in_channels, out_channels, volume_head=False, nll_head=False):
        super().__init__()
        self.latent_size = out_channels
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, self.latent_size)
        self.dropout = nn.Dropout(p=0.3)

        self.volume_head_enabled = bool(volume_head)
        if self.volume_head_enabled:
            self.vol_head = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        # Heteroscedastic NLL head: predicts log_scale per latent dim → uncertainty estimate.
        # Used as Laplace b (when super_loss_type=l1) or Gaussian σ (when mse).
        self.nll_head_enabled = bool(nll_head)
        if self.nll_head_enabled:
            self.fc_log_scale = nn.Linear(256, self.latent_size)

    def encode_features(self, x, mask=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(1)
            m_f = m.float()
            very_neg = torch.finfo(x.dtype).min / 2
            x_masked = x.masked_fill(m_f == 0, very_neg)
            x_max = torch.max(x_masked, 2)[0]
            denom = m_f.sum(dim=2).clamp(min=1.0)
            x_mean = (x * m_f).sum(dim=2) / denom
        else:
            x_max = torch.max(x, 2)[0]
            x_mean = torch.mean(x, 2)
        return torch.cat([x_max, x_mean], dim=1)

    def encoder(self, x, mask=None):
        x = self.encode_features(x, mask)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def encode_with_vol(self, x, mask=None, detach_for_vol=False):
        """Returns (latent, log_vol).

        When ``detach_for_vol=True``, vol_head's gradient does not flow back
        to the shared backbone — used when softvol via decoder is the primary
        encoder-supervising vol loss, and vol_head is trained as a parallel
        fast-inference predictor.
        """
        feat = self.encode_features(x, mask)
        h = F.relu(self.fc1(feat))
        h = self.dropout(h)
        latent = self.fc2(h)
        feat_for_vol = feat.detach() if detach_for_vol else feat
        log_vol = self.vol_head(feat_for_vol).squeeze(-1)
        return latent, log_vol

    def encode_with_aux(self, x, mask=None, detach_for_vol=False):
        """Returns dict: 'latent' (mu), 'log_vol' (if volume_head), 'log_scale' (if nll_head)."""
        feat = self.encode_features(x, mask)
        h = F.relu(self.fc1(feat))
        h = self.dropout(h)
        out = {'latent': self.fc2(h)}
        if self.volume_head_enabled:
            feat_for_vol = feat.detach() if detach_for_vol else feat
            out['log_vol'] = self.vol_head(feat_for_vol).squeeze(-1)
        if self.nll_head_enabled:
            out['log_scale'] = self.fc_log_scale(h)
        return out

    def forward(self, x, mask=None):
        return self.encoder(x, mask)


## thanks to: https://github.com/HAN-oQo/Pointnetautoencoder_pytorch/blob/main/model/model.py
class PointCloudEncoderLarge(nn.Module):
    def __init__(self, in_channels, out_channels, volume_head=False):
        super(PointCloudEncoderLarge, self).__init__()

        self.latent_size = out_channels

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512) # make global feature dimension smaller
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latent_size) #latent space

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.volume_head_enabled = bool(volume_head)
        if self.volume_head_enabled:
            self.vol_head = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def encode_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, 2)[0]  # (B, 1024)
        return x

    def encoder(self, x):
        feat = self.encode_features(x)
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def encode_with_vol(self, x, detach_for_vol=False):
        """Returns (latent, log_vol_pred).  Requires volume_head=True in init."""
        feat = self.encode_features(x)
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        latent = self.fc3(h)
        feat_for_vol = feat.detach() if detach_for_vol else feat
        log_vol = self.vol_head(feat_for_vol).squeeze(-1)
        return latent, log_vol

    def forward(self, x):
        x = self.encoder(x)
        return x


class PointCloudEncoderMaxMeanLargeMasked(nn.Module):
    """MaxMeanLarge with mask-aware pooling.

    Same backbone as ``PointCloudEncoderMaxMeanLarge``, but mean-pooling
    averages only over real (non-padded) points using a per-sample boolean
    mask.  Max-pool is unchanged (replication-padding doesn't affect max,
    but mask is still applied as a safety net via -inf replacement).

    Forward signature: ``forward(x, mask=None)`` where
      * ``x``: (B, 3, N)
      * ``mask``: (B, N) bool / float — 1 for real points, 0 for padding.

    If ``mask`` is None, falls back to unmasked max + mean (equivalent to
    ``PointCloudEncoderMaxMeanLarge``).
    """

    def __init__(self, in_channels, out_channels, volume_head=False, nll_head=False):
        super().__init__()
        self.latent_size = out_channels
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latent_size)

        # Optional volume regression head: uses pooled 2048 features, parallel to latent FC.
        # Output: log-volume scalar (log(ml)) for better dynamic range.
        self.volume_head_enabled = bool(volume_head)
        if self.volume_head_enabled:
            self.vol_head = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        # Heteroscedastic NLL head: predicts per-dim log_scale (Laplace b or Gaussian σ).
        self.nll_head_enabled = bool(nll_head)
        if self.nll_head_enabled:
            self.fc_log_scale = nn.Linear(256, self.latent_size)

    def encode_features(self, x, mask=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  # (B, 1024, N)
        if mask is not None:
            m = mask
            if m.dim() == 2:
                m = m.unsqueeze(1)           # (B, 1, N)
            m_f = m.float()
            # Masked max: set padded positions to -inf so they don't win max
            very_neg = torch.finfo(x.dtype).min / 2
            x_masked = x.masked_fill(m_f == 0, very_neg)
            x_max = torch.max(x_masked, 2)[0]
            denom = m_f.sum(dim=2).clamp(min=1.0)
            x_mean = (x * m_f).sum(dim=2) / denom
        else:
            x_max = torch.max(x, 2)[0]
            x_mean = torch.mean(x, 2)
        return torch.cat([x_max, x_mean], dim=1)  # (B, 2048)

    def encoder(self, x, mask=None):
        x = self.encode_features(x, mask)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def encode_with_vol(self, x, mask=None, detach_for_vol=False):
        """Returns (latent, log_vol_pred).  Requires volume_head=True in init.

        When ``detach_for_vol=True``, vol_head's gradient does not flow back
        to the shared backbone.
        """
        feat = self.encode_features(x, mask)
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        latent = self.fc3(h)
        feat_for_vol = feat.detach() if detach_for_vol else feat
        log_vol = self.vol_head(feat_for_vol).squeeze(-1)  # (B,)
        return latent, log_vol

    def encode_with_aux(self, x, mask=None, detach_for_vol=False):
        """Returns dict: 'latent' (mu), 'log_vol' (if volume_head), 'log_scale' (if nll_head)."""
        feat = self.encode_features(x, mask)
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        out = {'latent': self.fc3(h)}
        if self.volume_head_enabled:
            feat_for_vol = feat.detach() if detach_for_vol else feat
            out['log_vol'] = self.vol_head(feat_for_vol).squeeze(-1)
        if self.nll_head_enabled:
            out['log_scale'] = self.fc_log_scale(h)
        return out

    def forward(self, x, mask=None):
        return self.encoder(x, mask)


class PointCloudEncoderMaxMeanLarge(nn.Module):
    """Deeper PointCloud backbone (Large) with Max + Mean global pooling.

    Same conv backbone as PointCloudEncoderLarge (3 → 64 → 64 → 64 → 128 → 1024)
    but aggregates features via concat(max, mean) across points, doubling the
    global feature dim (1024 -> 2048) before the FC head.

    Mean pooling captures the distribution of visible points, which adds a
    density-aware signal that is otherwise lost by max-only pooling.  This makes
    the latent more robust to variable point counts / sparse partial views —
    useful when the encoder must generalise from dense training samples to
    sparse test inputs.

    Note: PCDs in this pipeline are replicated to a fixed ``pad_size`` (not
    zero-padded), so the mean pool is a weighted mean of real points with no
    contamination from padding; the same is true for PointCloudEncoderMaxMean.
    """

    def __init__(self, in_channels, out_channels, volume_head=False):
        super().__init__()
        self.latent_size = out_channels

        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # FC head: 2048 (1024 max + 1024 mean) -> 512 -> 256 -> latent
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latent_size)

        self.volume_head_enabled = bool(volume_head)
        if self.volume_head_enabled:
            self.vol_head = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def encode_features(self, x):
        """Return post-pooling 2048d feature (before FC head)."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x_max = torch.max(x, 2)[0]
        x_mean = torch.mean(x, 2)
        return torch.cat([x_max, x_mean], dim=1)  # (B, 2048)

    def encoder(self, x):
        x = self.encode_features(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def encode_with_vol(self, x):
        """Returns (latent, log_vol_pred).  Requires volume_head=True in init."""
        feat = self.encode_features(x)
        h = F.relu(self.fc1(feat))
        h = F.relu(self.fc2(h))
        latent = self.fc3(h)
        log_vol = self.vol_head(feat).squeeze(-1)
        return latent, log_vol

    def forward(self, x):
        return self.encoder(x)


# ─────────────────────────────────────────────────────────────────────────────
# PointNet++ encoder (CUDA-accelerated via torch_cluster, density-robust)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from torch_cluster import fps as _tc_fps, radius as _tc_radius, knn as _tc_knn
    _HAS_TORCH_CLUSTER = True
except Exception:
    _HAS_TORCH_CLUSTER = False


def _batched_fps(xyz: torch.Tensor, n_sample: int) -> torch.Tensor:
    """Farthest Point Sampling on batched input (B, N, 3).

    Uses torch_cluster CUDA kernel when available, falls back to a Python
    sequential implementation otherwise.
    Returns long indices of shape (B, n_sample) relative to each batch.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    if _HAS_TORCH_CLUSTER and xyz.is_cuda:
        # torch_cluster expects flat (sum_N, 3) with batch vector
        flat = xyz.reshape(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)
        ratio = float(n_sample) / float(N)
        idx_flat = _tc_fps(flat, batch, ratio=ratio, random_start=True)
        # torch_cluster may return slightly more/less than requested due to rounding;
        # reshape carefully per batch.
        idx_per_batch = []
        # Map flat indices back to per-batch local indices and pick first n_sample per batch.
        for b in range(B):
            m = idx_flat[(idx_flat >= b * N) & (idx_flat < (b + 1) * N)] - b * N
            if m.numel() >= n_sample:
                idx_per_batch.append(m[:n_sample])
            else:
                # Pad with the first index (should rarely happen)
                pad = m[:1].repeat(n_sample - m.numel()) if m.numel() > 0 else torch.zeros(n_sample, dtype=torch.long, device=device)
                idx_per_batch.append(torch.cat([m, pad]) if m.numel() > 0 else pad)
        return torch.stack(idx_per_batch, dim=0)

    # Fallback: Python loop
    idx = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    far = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(n_sample):
        idx[:, i] = far
        centroid = xyz.gather(1, far.view(B, 1, 1).expand(-1, 1, 3))
        d = ((xyz - centroid) ** 2).sum(dim=-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(dim=-1)
    return idx


def _batched_ball_query(xyz: torch.Tensor, centroids: torch.Tensor,
                        radius_val: float, n_neighbors: int) -> torch.Tensor:
    """Local-neighborhood indices for each centroid on batched inputs.

    Uses ``torch_cluster.knn`` when available (vectorized CUDA) — returns
    k-nearest neighbors regardless of radius; radius acts as a soft cutoff.
    Falls back to pairwise-distance argsort otherwise.
    Returns (B, S, n_neighbors) long tensor of per-batch local indices.
    """
    B, N, _ = xyz.shape
    S = centroids.shape[1]
    device = xyz.device
    if _HAS_TORCH_CLUSTER and xyz.is_cuda:
        flat_x = xyz.reshape(B * N, 3)
        flat_y = centroids.reshape(B * S, 3)
        batch_x = torch.arange(B, device=device).repeat_interleave(N)
        batch_y = torch.arange(B, device=device).repeat_interleave(S)
        # knn returns (2, S_total * K) sorted by y_idx ascending, distance ascending within
        edge = _tc_knn(flat_x, flat_y, k=n_neighbors,
                       batch_x=batch_x, batch_y=batch_y)
        # edge[1] is global x idx; convert to per-batch local
        x_glob = edge[1]
        x_local = x_glob - (edge[0] // S) * N
        # Reshape to (B*S, K); torch_cluster returns exactly K neighbors per y
        out = x_local.view(B * S, n_neighbors)
        return out.view(B, S, n_neighbors)

    # Fallback: pairwise distance
    dist2 = ((centroids.unsqueeze(2) - xyz.unsqueeze(1)) ** 2).sum(dim=-1)
    group_idx = dist2.argsort(dim=-1)[:, :, :n_neighbors]
    sorted_d2 = dist2.gather(-1, group_idx)
    mask = sorted_d2 > (radius_val ** 2)
    first = group_idx[:, :, :1].expand(-1, -1, n_neighbors)
    return torch.where(mask, first, group_idx)


def _gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """points: (B, N, C), idx: (B, S) or (B, S, K) -> (B, S, C) or (B, S, K, C)."""
    B = points.shape[0]; C = points.shape[-1]
    if idx.dim() == 2:
        return points.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
    else:
        S, K = idx.shape[1], idx.shape[2]
        flat = idx.reshape(B, S * K)
        out = points.gather(1, flat.unsqueeze(-1).expand(-1, -1, C))
        return out.reshape(B, S, K, C)


class PointNet2SetAbstraction(nn.Module):
    """One PointNet++ Set Abstraction layer (SSG variant)."""
    def __init__(self, n_sample: int, radius: float, n_neighbors: int,
                 in_channels: int, mlp_channels: list):
        super().__init__()
        self.n_sample = n_sample
        self.radius = radius
        self.n_neighbors = n_neighbors
        layers = []
        prev = in_channels + 3
        for c in mlp_channels:
            layers += [nn.Conv2d(prev, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            prev = c
        self.mlp = nn.Sequential(*layers)
        self.out_channels = mlp_channels[-1]

    def forward(self, xyz: torch.Tensor, features):
        """xyz: (B, N, 3), features: (B, N, C) or None -> new_xyz (B, S, 3), new_feat (B, S, out_channels)."""
        B, N, _ = xyz.shape
        S = min(self.n_sample, N)
        center_idx = _batched_fps(xyz, S)
        new_xyz = _gather_points(xyz, center_idx)
        group_idx = _batched_ball_query(xyz, new_xyz, self.radius, self.n_neighbors)
        grouped_xyz = _gather_points(xyz, group_idx)
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        if features is not None:
            grouped_feat = _gather_points(features, group_idx)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
        else:
            grouped = grouped_xyz
        grouped = grouped.permute(0, 3, 2, 1).contiguous()
        out = self.mlp(grouped)
        out = out.max(dim=2)[0]
        out = out.permute(0, 2, 1).contiguous()
        return new_xyz, out


class PointNetPlusPlusEncoder(nn.Module):
    """Hierarchical PointNet++ (SSG) encoder. CUDA via torch_cluster when available."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert in_channels == 3
        self.sa1 = PointNet2SetAbstraction(n_sample=256, radius=0.03, n_neighbors=16,
                                            in_channels=0, mlp_channels=[32, 32, 64])
        self.sa2 = PointNet2SetAbstraction(n_sample=64, radius=0.06, n_neighbors=16,
                                            in_channels=64, mlp_channels=[64, 64, 128])
        self.sa3 = PointNet2SetAbstraction(n_sample=16, radius=0.12, n_neighbors=16,
                                            in_channels=128, mlp_channels=[128, 128, 256])
        self.global_mlp = nn.Sequential(
            nn.Conv1d(256 + 3, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 3:
            xyz = x.permute(0, 2, 1).contiguous()
        else:
            xyz = x
        xyz1, feat1 = self.sa1(xyz, None)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        g_input = torch.cat([xyz3, feat3], dim=-1).permute(0, 2, 1).contiguous()
        g_feat = self.global_mlp(g_input)
        g_feat = g_feat.max(dim=2)[0]
        z = F.relu(self.bn_fc1(self.fc1(g_feat)))
        z = self.fc2(z)
        return z


## thanks to: https://github.com/antao97/UnsupervisedPointCloudReconstruction/blob/master/model.py
class FoldNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FoldNetEncoder, self).__init__()
        self.latent_size = out_channels
        self.k = 16
        self.feat_dims = 512
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, self.feat_dims, 1),
            nn.ReLU(),
            nn.Conv1d(self.feat_dims, self.latent_size, 1),
        )

    def knn(self, x, k):
        batch_size = x.size(0)
        num_points = x.size(2)

        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]            # (batch_size, num_points, k)

        if idx.get_device() == -1:
            idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
        else:
            idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        return idx


    def local_cov(self, pts, idx):
        batch_size = pts.size(0)
        num_points = pts.size(2)
        pts = pts.view(batch_size, -1, num_points)              # (batch_size, 3, num_points)
    
        _, num_dims, _ = pts.size()

        x = pts.transpose(2, 1).contiguous()                    # (batch_size, num_points, 3)
        x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*num_points*2, 3)
        x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, 3)

        x = torch.matmul(x[:,:,0].unsqueeze(3), x[:,:,1].unsqueeze(2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
        # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
        x = x.view(batch_size, num_points, 9).transpose(2, 1)   # (batch_size, 9, num_points)

        x = torch.cat((pts, x), dim=1)                          # (batch_size, 12, num_points)
        return x


    def local_maxpool(self, x, idx):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
    
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()                      # (batch_size, num_points, num_dims)
        x = x.view(batch_size*num_points, -1)[idx, :]           # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
        x = x.view(batch_size, num_points, -1, num_dims)        # (batch_size, num_points, k, num_dims)
        x, _ = torch.max(x, dim=2)                              # (batch_size, num_points, num_dims)
        return x

    def graph_layer(self, x, idx):           
        x = self.local_maxpool(x, idx)    
        x = self.linear1(x)  
        x = x.transpose(2, 1)                                     
        x = F.relu(self.conv1(x))                            
        x = self.local_maxpool(x, idx)  
        x = self.linear2(x) 
        x = x.transpose(2, 1)                                   
        x = self.conv2(x)                       
        return x

    def forward(self, pts):
        idx = self.knn(pts, k=self.k)
        x = self.local_cov(pts, idx)            # (batch_size, 3, num_points) -> (batch_size, 12, num_points])            
        x = self.mlp1(x)                        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)            # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)                        # (batch_size, 1024, 1) -> (batch_size, latent_size, 1)
        x = x.view(-1, self.latent_size)        # (batch_size, latent_size, 1) -> (batch_size, latent_size)
        return x                                # (batch_size, latent_size)


# Added by sungjay: MinkowskiEngine-based latent encoder for point clouds
class MinkLatentEncoder(nn.Module):
    """
    Sparse-convolutional encoder for point clouds backed by MinkowskiEngine.

    The encoder mirrors the interfaces offered by the other point-cloud models
    in this module: it accepts either

    * a dense tensor shaped ``[B, N, 3]`` (or ``[N, 3]`` for a single cloud), or
    * a tuple of ``(coords, feats)`` following the ``ME.SparseTensor`` layout.

    In the dense case, points are normalised (optionally) before voxelisation,
    then used both as sparse coordinates and features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        D: int = 3,
        voxel_size: float = 0.01,
        normalize_input: bool = True,
        quantization_mode: Optional["ME.SparseTensorQuantizationMode"] = None,
    ) -> None:
        super().__init__()
        if not _HAS_MINKOWSKI:
            raise ImportError(
                "MinkowskiEngine is not installed. Please install it to use MinkLatentEncoder."
            )

        self.latent_size = out_channels
        self.in_channels = in_channels
        self.D = D
        self.voxel_size = float(voxel_size)
        self.normalize_input = normalize_input
        self.quantization_mode = (
            quantization_mode
            if quantization_mode is not None
            else ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
        )

        channels = [64, 128, 256]
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, channels[0], kernel_size=3, stride=1, dimension=self.D),
            ME.MinkowskiBatchNorm(channels[0]),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(channels[0], channels[1], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(channels[1]),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(channels[1], channels[2], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(channels[2]),
            ME.MinkowskiReLU(inplace=True),
        )
        self.pool = ME.MinkowskiGlobalMaxPooling()
        self.proj = ME.MinkowskiLinear(channels[-1], self.latent_size)

    def _module_device(self) -> torch.device:
        return next(self.parameters()).device

    def _ensure_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError(
                f"Expected a tensor shaped [B, N, 3] (or [N, 3]); got {tuple(tensor.shape)}"
            )
        return tensor

    def _build_sparse_from_dense(
        self,
        points: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        centers: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
    ) -> "ME.SparseTensor":
        if isinstance(points, torch.Tensor):
            points = self._ensure_batch(points)
            batch_points: List[torch.Tensor] = [pts for pts in points]
        elif isinstance(points, list):
            # Handle list of tensors directly (from custom collate function)
            batch_points: List[torch.Tensor] = points
        else:
            raise TypeError("Points must be provided as a torch.Tensor or list of torch.Tensor when using dense input mode.")

        if features is not None:
            if isinstance(features, torch.Tensor):
                features = self._ensure_batch(features)
                batch_features: List[Optional[torch.Tensor]] = [feat for feat in features]
            elif isinstance(features, list):
                batch_features: List[Optional[torch.Tensor]] = features
            else:
                raise TypeError("Features must be provided as a torch.Tensor or list when using dense input mode.")
            if len(batch_features) != len(batch_points):
                raise ValueError("Features batch does not match the number of point clouds.")
        else:
            batch_features = [None] * len(batch_points)

        def prepare_auxiliary(aux: Optional[torch.Tensor], name: str) -> List[Optional[torch.Tensor]]:
            if aux is None:
                return [None] * len(batch_points)
            if isinstance(aux, list):
                return aux
            if not isinstance(aux, torch.Tensor):
                raise TypeError(f"{name} must be provided as a torch.Tensor or list when using dense input mode.")
            if aux.dim() == 1:
                aux = aux.unsqueeze(0)
            elif aux.dim() != 2:
                raise ValueError(f"{name} must have shape [B, 3] or [3].")
            if aux.shape[0] != len(batch_points):
                raise ValueError(f"{name} batch does not match the number of point clouds.")
            return [a for a in aux]

        batch_centers = prepare_auxiliary(centers, "Centers")
        batch_scales = prepare_auxiliary(scales, "Scales")

        batch_coords: List[torch.Tensor] = []
        batch_feats: List[torch.Tensor] = []

        for batch_idx, (pts, feat, center, scale) in enumerate(zip(batch_points, batch_features, batch_centers, batch_scales)):
            if pts.numel() == 0:
                continue

            pts = pts.to(torch.float32)

            if center is not None:
                center = center.to(pts.device, pts.dtype)
                if center.dim() == 1:
                    center = center.unsqueeze(0)
                pts = pts - center
            elif self.normalize_input:
                pts = pts - pts.mean(dim=0, keepdim=True)

            if scale is not None:
                scale = scale.to(pts.device, pts.dtype)
                if scale.dim() == 1:
                    scale = scale.unsqueeze(0)
                scale = torch.clamp(scale, min=1e-6)
                pts = pts / scale

            coords = torch.round(pts / self.voxel_size).to(torch.int32)
            batch_index = torch.full(
                (coords.shape[0], 1),
                batch_idx,
                dtype=torch.int32,
                device=coords.device,
            )
            batch_coords.append(torch.cat([coords, batch_index], dim=1))

            if feat is None:
                feat_tensor = pts
            else:
                feat_tensor = feat.to(torch.float32)
                if feat_tensor.shape[0] != pts.shape[0]:
                    raise ValueError("Features must have the same number of points as the coordinates.")

            if feat_tensor.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected features with {self.in_channels} channels, received {feat_tensor.shape[1]}."
                )
            batch_feats.append(feat_tensor)

        if not batch_coords:
            raise ValueError("Received empty point clouds; cannot construct a SparseTensor.")

        coords = torch.cat(batch_coords, dim=0)
        feats = torch.cat(batch_feats, dim=0)
        target_device = self._module_device()
        coords = coords.to(target_device)
        feats = feats.to(target_device)
        return ME.SparseTensor(
            features=feats,
            coordinates=coords,
            quantization_mode=self.quantization_mode,
            device=target_device,
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) == 1:
            # Handle both tensors and lists
            if isinstance(inputs[0], list):
                batch_size = len(inputs[0])
            else:
                points_batch = self._ensure_batch(inputs[0])
                batch_size = points_batch.shape[0]
            sparse = self._build_sparse_from_dense(inputs[0])
        elif len(inputs) == 2:
            first, second = inputs
            if (
                isinstance(first, (torch.Tensor, list))
                and isinstance(second, (torch.Tensor, list))
            ):
                # Check if this is points + centers (both have shape [..., 3])
                is_points_centers = False
                if isinstance(first, torch.Tensor) and first.shape[-1] == 3:
                    is_points_centers = isinstance(second, torch.Tensor) and second.shape[-1] == 3
                elif isinstance(first, list) and len(first) > 0:
                    is_points_centers = isinstance(first[0], torch.Tensor) and first[0].shape[-1] == 3
                
                if is_points_centers:
                    if isinstance(first, list):
                        batch_size = len(first)
                        centers_batch = second
                    else:
                        points_batch = self._ensure_batch(first)
                        if isinstance(second, torch.Tensor):
                            if second.dim() == 1:
                                centers_batch = second.unsqueeze(0)
                            elif second.dim() == 2:
                                centers_batch = second
                            else:
                                raise ValueError("Centers tensor must have shape [B, 3] or [3].")
                            if centers_batch.shape[0] != points_batch.shape[0]:
                                raise ValueError("Points and centers batch size mismatch.")
                        else:
                            centers_batch = second
                        batch_size = points_batch.shape[0]
                    sparse = self._build_sparse_from_dense(first, centers=centers_batch)
            else:
                coords, feats = first, second
                target_device = self._module_device()
                # Infer batch size from coordinates
                batch_size = coords[:, -1].max().item() + 1
                sparse = ME.SparseTensor(
                    features=feats.to(target_device, torch.float32),
                    coordinates=coords.to(target_device, torch.int32),
                    quantization_mode=self.quantization_mode,
                    device=target_device,
                )
        elif len(inputs) == 3:
            points, centers, scales = inputs
            
            # Handle lists or tensors
            if isinstance(points, list):
                batch_size = len(points)
                centers_batch = centers if isinstance(centers, list) else [c for c in centers]
                scales_batch = scales if isinstance(scales, list) else [s for s in scales]
            else:
                points_batch = self._ensure_batch(points)
                if isinstance(centers, torch.Tensor):
                    if centers.dim() == 1:
                        centers_batch = centers.unsqueeze(0)
                    elif centers.dim() == 2:
                        centers_batch = centers
                    else:
                        raise ValueError("Centers tensor must have shape [B, 3] or [3].")
                else:
                    centers_batch = centers

                if isinstance(scales, torch.Tensor):
                    if scales.dim() == 1:
                        scales_batch = scales.unsqueeze(0)
                    elif scales.dim() == 2:
                        scales_batch = scales
                    else:
                        raise ValueError("Scales tensor must have shape [B, 3] or [3].")
                else:
                    scales_batch = scales

                if not (points_batch.shape[0] == (centers_batch.shape[0] if isinstance(centers_batch, torch.Tensor) else len(centers_batch)) == (scales_batch.shape[0] if isinstance(scales_batch, torch.Tensor) else len(scales_batch))):
                    raise ValueError("Points, centers, and scales must have matching batch sizes.")
                batch_size = points_batch.shape[0]
            
            sparse = self._build_sparse_from_dense(points, centers=centers, scales=scales)
        else:
            raise TypeError("MinkLatentEncoder expects either (points,), (points, centers), or (coords, feats) as inputs.")

        x = self.backbone(sparse)
        x = self.pool(x)
        x = self.proj(x)
        
        # MinkowskiGlobalPooling returns features only for non-empty batches
        # We need to reconstruct full batch with zeros for empty samples
        pooled_features = x.F  # Shape: [num_non_empty_batches, latent_size]
        
        # Get batch indices from the pooled sparse tensor
        if hasattr(x, 'coordinate_manager'):
            batch_indices = x.coordinates[:, -1].cpu().numpy()
            unique_batches = sorted(set(batch_indices))
        else:
            # Fallback: assume all batches are present
            unique_batches = list(range(batch_size))
        
        # Create output tensor with correct batch size
        target_device = self._module_device()
        output = torch.zeros(batch_size, self.latent_size, device=target_device, dtype=pooled_features.dtype)
        
        # Fill in features for non-empty batches
        for i, batch_idx in enumerate(unique_batches):
            if batch_idx < batch_size:
                output[batch_idx] = pooled_features[i]
        
        return output

# import MinkowskiEngine as ME
# from tcore.utils.interpolate import knn_up

# class MinkEncoderDecoder(nn.Module):
#     """
#     Basic ResNet architecture using sparse convolutions
#     """

#     def __init__(self, cfg, template_points):
#         super().__init__()

#         n_classes = 20

#         cr = cfg.CR
#         self.D = cfg.DIMENSION
#         input_dim = cfg.INPUT_DIM
#         self.res = cfg.RESOLUTION
#         self.interpolate = False  #cfg.INTERPOLATE
#         self.feat_key = cfg.FEAT_KEY

#         self.template_points = template_points

#         if self.interpolate:
#             # from tcore.utils.interpolate import knn_up
#             self.knn_up = knn_up(cfg.KNN_UP)

#         cs = cfg.CHANNELS
#         cs = [int(cr*x) for x in cs]
#         self.stem = nn.Sequential(
#             ME.MinkowskiConvolution(input_dim,
#                                     cs[0],
#                                     kernel_size=3,
#                                     stride=1,
#                                     dimension=self.D),
#             ME.MinkowskiBatchNorm(cs[0]),
#             ME.MinkowskiReLU(True),
#             ME.MinkowskiConvolution(cs[0],
#                                     cs[0],
#                                     kernel_size=3,
#                                     stride=1,
#                                     dimension=self.D),
#             ME.MinkowskiBatchNorm(cs[0]),
#             ME.MinkowskiReLU(inplace=True),
#         )

#         self.stage1 = nn.Sequential(
#             BasicConvolutionBlock(cs[0],
#                                   cs[0],
#                                   ks=2,
#                                   stride=2,
#                                   dilation=1,
#                                   D=self.D),
#             ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1, D=self.D),
#             ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1, D=self.D),
#         )

#         self.stage2 = nn.Sequential(
#             BasicConvolutionBlock(cs[1],
#                                   cs[1],
#                                   ks=2,
#                                   stride=2,
#                                   dilation=1,
#                                   D=self.D),
#             ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1, D=self.D),
#             ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1, D=self.D),
#         )

#         self.stage3 = nn.Sequential(
#             BasicConvolutionBlock(cs[2],
#                                   cs[2],
#                                   ks=2,
#                                   stride=2,
#                                   dilation=1,
#                                   D=self.D),
#             ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1, D=self.D),
#             ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1, D=self.D),
#         )

#         self.stage4 = nn.Sequential(
#             BasicConvolutionBlock(cs[3],
#                                   cs[3],
#                                   ks=2,
#                                   stride=2,
#                                   dilation=1,
#                                   D=self.D),
#             ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1, D=self.D),
#             ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1, D=self.D),
#         )

#         self.up1 = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2, D=self.D),
#             nn.Sequential(
#                 ResidualBlock(cs[5] + cs[3],
#                               cs[5],
#                               ks=3,
#                               stride=1,
#                               dilation=1,
#                               D=self.D),
#                 ResidualBlock(cs[5],
#                               cs[5],
#                               ks=3,
#                               stride=1,
#                               dilation=1,
#                               D=self.D),
#             ),
#         ])

#         self.up2 = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2, D=self.D),
#             nn.Sequential(
#                 ResidualBlock(cs[6] + cs[2],
#                               cs[6],
#                               ks=3,
#                               stride=1,
#                               dilation=1,
#                               D=self.D),
#                 ResidualBlock(cs[6],
#                               cs[6],
#                               ks=3,
#                               stride=1,
#                               dilation=1,
#                               D=self.D),
#             ),
#         ])

#         self.up3 = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2, D=self.D),
#             nn.Sequential(
#                 ResidualBlock(cs[7] + cs[1],
#                               cs[7],
#                               ks=3,
#                               stride=1,
#                               dilation=1,
#                               D=self.D),
#                 ResidualBlock(cs[7],
#                               cs[7],
#                               ks=3,
#                               stride=1,
#                               dilation=1,
#                               D=self.D),
#             ),
#         ])

#         self.up4 = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2, D=self.D),
#             nn.Sequential(
#                 ResidualBlock(cs[8] + cs[0],
#                               cs[8],
#                               ks=3,
#                               stride=1,
#                               dilation=1,
#                               D=self.D),
#                 ResidualBlock(cs[8],
#                               cs[8],
#                               ks=3,
#                               stride=1,
#                               dilation=1,
#                               D=self.D),
#             ),
#         ])

#         self.sem_head = nn.Linear(cs[-1], n_classes)

#         levels = [cs[-i] for i in range(4, 0, -1)]
#         self.out_bnorm = nn.ModuleList([nn.BatchNorm1d(l) for l in levels])

#     def forward(self, x):
#         batch_size = len(x['points'])
#         in_feats = {}
#         in_feats['points'] = x['points']
#         if 'normals' in x.keys():
#             in_feats['feats'] = [
#                 np.hstack((x['points'][i], x['normals'][i]))
#                 for i in range(batch_size)
#             ]
#         elif 'colors' in x.keys():
#             in_feats['feats'] = [
#                 np.hstack((x['points'][i], x['colors'][i]))
#                 for i in range(batch_size)
#             ]
#         else:
#             assert False

#         in_field = self.TensorField(in_feats)

#         x0 = self.stem(in_field.sparse())
#         x1 = self.stage1(x0)
#         x2 = self.stage2(x1)
#         x3 = self.stage3(x2)
#         x4 = self.stage4(x3)

#         y1 = self.up1[0](x4)
#         y1 = ME.cat(y1, x3)
#         y1 = self.up1[1](y1)

#         y2 = self.up2[0](y1)
#         y2 = ME.cat(y2, x2)
#         y2 = self.up2[1](y2)

#         y3 = self.up3[0](y2)
#         y3 = ME.cat(y3, x1)
#         y3 = self.up3[1](y3)

#         y4 = self.up4[0](y3)
#         y4 = ME.cat(y4, x0)
#         y4 = self.up4[1](y4)

#         out_feats = [y1, y2, y3, y4]

#         # vox2feat and apply batchnorm
#         coors = [
#             in_field.decomposed_coordinates for _ in range(len(out_feats))
#         ]
#         coors = [[c * self.res for c in coors[i]] for i in range(len(coors))]
#         if self.interpolate:  # vox2feat using knn interpolation
#             bs = in_field.coordinate_manager.number_of_unique_batch_indices()
#             vox_coors = [[l.coordinates_at(i) * self.res for i in range(bs)]
#                          for l in out_feats]
#             feats = [[
#                 bn(self.knn_up(vox_c, vox_f, pt_c))
#                 for vox_c, vox_f, pt_c in zip(vc, vf.decomposed_features, pc)
#             ] for vc, vf, pc, bn in zip(vox_coors, out_feats, coors,
#                                         self.out_bnorm)]
#         else:
#             out_fields = [feat.slice(in_field) for feat in out_feats]
#             feats = [[self.out_bnorm[i](df) for df in f.decomposed_features]
#                      for i, f in enumerate(out_fields)]

#         feats, coors, pad_masks = self.pad_batch(coors, feats)
#         return feats, coors, pad_masks

#     def TensorField(self, x):
#         """
#         Build a tensor field from coordinates and features in the
#         input batch
#         The coordinates are quantized using the provided resolution

#         """

#         feat_tfield = ME.TensorField(
#             features=torch.from_numpy(np.concatenate(x["feats"], 0)).float(),
#             coordinates=ME.utils.batched_coordinates(
#                 [i / self.res for i in x["points"]], dtype=torch.float32),
#             quantization_mode=ME.SparseTensorQuantizationMode.
#             UNWEIGHTED_AVERAGE,
#             minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
#             device="cuda",
#         )

#         return feat_tfield

#     def pad_batch(self, coors, feats):
#         """
#         From a list of multi-level features create a list of batched tensors with
#         point features padded to the max number of points/voxels in the batch.
#         For point-level features, create intermediate TensorField:
#             voxel_features -> TensorField -> full scale point features

#         returns:
#             feats: List of batched feature Tensors per feature level (features for each voxel/point)
#             coors: List of batched coordinate Tensors per feature level to later generate pos enc
#             pad_masks: List of batched bool Tensors indicating where the padding was performed
#         """
#         # list [feat_levels+1,bs], where each element is [n_pts,dim]
#         # 3 features levels + full scale = 4
#         # get max number of points/voxels in the batch for each feature level
#         maxs = [max([level.shape[0] for level in batch]) for batch in feats]
#         # pad to max number of points in batch and put each feature level in a single Tensor
#         coors = [
#             torch.stack(
#                 [F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
#             for i, batch in enumerate(coors)
#         ]
#         pad_masks = [
#             torch.stack([
#                 F.pad(torch.zeros_like(f[:, 0]), (0, maxs[i] - f.shape[0]),
#                       value=1).bool() for f in batch
#             ]) for i, batch in enumerate(feats)
#         ]
#         feats = [
#             torch.stack(
#                 [F.pad(f, (0, 0, 0, maxs[i] - f.shape[0])) for f in batch])
#             for i, batch in enumerate(feats)
#         ]
#         return feats, coors, pad_masks


# class BasicConvolutionBlock(nn.Module):

#     def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             ME.MinkowskiConvolution(inc,
#                                     outc,
#                                     kernel_size=ks,
#                                     dilation=dilation,
#                                     stride=stride,
#                                     dimension=D),
#             ME.MinkowskiBatchNorm(outc),
#             ME.MinkowskiLeakyReLU(inplace=True),
#         )

#     def forward(self, x):
#         out = self.net(x)
#         return out


# class BasicDeconvolutionBlock(nn.Module):

#     def __init__(self, inc, outc, ks=3, stride=1, D=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             ME.MinkowskiConvolutionTranspose(inc,
#                                              outc,
#                                              kernel_size=ks,
#                                              stride=stride,
#                                              dimension=D),
#             ME.MinkowskiBatchNorm(outc),
#             ME.MinkowskiLeakyReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.net(x)


# class ResidualBlock(nn.Module):

#     def __init__(self, inc, outc, ks=3, stride=1, dilation=1, D=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             ME.MinkowskiConvolution(inc,
#                                     outc,
#                                     kernel_size=ks,
#                                     dilation=dilation,
#                                     stride=stride,
#                                     dimension=D),
#             ME.MinkowskiBatchNorm(outc),
#             ME.MinkowskiReLU(inplace=True),
#             ME.MinkowskiConvolution(outc,
#                                     outc,
#                                     kernel_size=ks,
#                                     dilation=dilation,
#                                     stride=1,
#                                     dimension=D),
#             ME.MinkowskiBatchNorm(outc),
#         )

#         self.downsample = (nn.Sequential() if
#                            (inc == outc and stride == 1) else nn.Sequential(
#                                ME.MinkowskiConvolution(inc,
#                                                        outc,
#                                                        kernel_size=1,
#                                                        dilation=1,
#                                                        stride=stride,
#                                                        dimension=D),
#                                ME.MinkowskiBatchNorm(outc),
#                            ))

#         out = self.relu(self.net(x) + self.downsample(x))
#         return out


class PointCloudEncoderSplit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointCloudEncoderSplit, self).__init__()
        
        self.latent_size = out_channels # Total size (e.g. 35 = 32 shape + 3 size)
        self.shape_dim = 32
        self.size_dim = 3
        
        # Shared Feature Extractor
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.3)

        # Split Heads
        self.fc_shape = nn.Linear(256, self.shape_dim)
        self.fc_size = nn.Linear(256, self.size_dim)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Branching
        shape_code = self.fc_shape(x) # [B, 32]
        size_check = self.fc_size(x)  # [B, 1]
        
        # Concatenate for compatibility
        return torch.cat([shape_code, size_check], dim=1)
    
    def forward(self, x):
        x = self.encoder(x)
        return x