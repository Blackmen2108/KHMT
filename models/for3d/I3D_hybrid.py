import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import ReplicationPad3d
from experiment_config import config  # adjust path if needed

# ------------------------
# Utilities
# ------------------------
def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


# -------------------- DropPath (Stochastic Depth) --------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        # shape: (batch, 1, 1, 1, 1)
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -------------------- SE Block for 3D --------------------
class SEBlock3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


# -------------------- Baseline Unit3Dpy / MaxPool / Mixed (copied & unchanged) --------------------
class Unit3Dpy(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        activation="relu",
        padding="SAME",
        use_bias=False,
        use_bn=True,
    ):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == "VALID":
            padding_shape = 0
        else:
            raise ValueError("padding should be in [VALID|SAME] but got {}".format(padding))

        if padding == "SAME":
            if not self.simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=pad_size, bias=use_bias
                )
        elif padding == "VALID":
            self.conv3d = torch.nn.Conv3d(
                in_channels, out_channels, kernel_size, padding=padding_shape, stride=stride, bias=use_bias
            )
        else:
            raise ValueError("padding should be in [VALID|SAME] but got {}".format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == "relu":
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == "SAME" and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3dTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding="SAME"):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="SAME")
        branch_3_conv2 = Unit3Dpy(in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out


# -------------------- Enhanced I3D (I3D + SE + DropPath) --------------------
class I3DEnhanced(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels,
        modality="rgb",
        dropout_prob=0,
        name="inception_enhanced",
        pre_trained=True,
        freeze_bn=True,
        se_reduction=16,
        drop_path_prob=0.0,
    ):
        super(I3DEnhanced, self).__init__()
        # keep args
        self.name = name
        self.num_classes = num_classes
        self.freeze_bn = freeze_bn
        self.input_channels = input_channels
        if modality == "rgb":
            in_channels = 3
        elif modality == "flow":
            in_channels = 2
        else:
            raise ValueError("{} not among known modalities [rgb|flow]".format(modality))
        if in_channels != input_channels:
            raise ValueError(
                "Modality "
                + str(modality)
                + " does not correspond to input_channels "
                + str(input_channels)
                + ". input_channels should be: "
                + str(3 if modality == "rgb" else 2)
            )
        self.modality = modality

        # ---- original I3D layers (same as baseline) ----
        conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64, in_channels=in_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding="SAME"
        )
        self.conv3d_1a_7x7 = conv3d_1a_7x7
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME")

        conv3d_2b_1x1 = Unit3Dpy(out_channels=64, in_channels=64, kernel_size=(1, 1, 1), padding="SAME")
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(out_channels=192, in_channels=64, kernel_size=(3, 3, 3), padding="SAME")
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME")

        # Mixed_3b, 3c
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding="SAME")

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding="SAME")

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)

        # The classifier conv (will be replaced if pretrained loaded)
        self.conv3d_0c_1x1 = Unit3Dpy(
            in_channels=1024, out_channels=self.num_classes, kernel_size=(1, 1, 1), activation=None, use_bias=True, use_bn=False
        )

        # ---- enhancements ----
        # SE blocks: apply after mixed_4f and mixed_5c
        self.se_4f = SEBlock3D(channels=832, reduction=se_reduction)  # mixed_4f output channels -> 832
        self.se_5c = SEBlock3D(channels=1024, reduction=se_reduction)  # mixed_5c output channels -> 1024

        # DropPath
        self.drop_path = DropPath(drop_prob=drop_path_prob)

        # If pre-trained weights are requested, load baseline weights (same as baseline)
        if pre_trained:
            # baseline had intermediate conv3d_0c_1x1 set to 400 classes; baseline loads weights from config.MODEL_RGB_I3D
            try:
                self.load_state_dict(torch.load(config.MODEL_RGB_I3D))
            except Exception as e:
                print("Warning: could not load pretrained I3D weights:", e)

        # freeze BN layers if requested
        self.train()
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, torch.nn.BatchNorm3d):
                    m.eval()
                    if hasattr(m, "weight") and m.weight is not None:
                        m.weight.requires_grad = False
                    if hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = False

    def forward(self, inp):
        # If original model expects 3 channels but input has 1, expand
        if self.input_channels == 3 and inp.shape[1] == 1:
            inp = inp.expand(-1, 3, -1, -1, -1)

        # forward through original blocks
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)

        # apply SE on mixed_4f
        out = self.se_4f(out)

        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)

        # apply SE on mixed_5c
        out = self.se_5c(out)

        # optional DropPath before pooling
        out = self.drop_path(out)

        out = torch.nn.AvgPool3d((2, 2, 2), (1, 1, 1))(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)

        # original returns mean over depth: out.mean(2).reshape(out.shape[0])
        out = out.mean(2).reshape(out.shape[0])

        return out