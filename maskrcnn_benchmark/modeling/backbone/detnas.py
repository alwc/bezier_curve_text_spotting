"""
Discrete structure of 4 path

Includes utils to convert continous Auto-DeepLab to discrete ones
"""

import torch
from torch import nn

from maskrcnn_benchmark.nas.blocks import FixBlock
from .hnas_common import conv3x3_bn, conv1x1_bn, conv_bn


class Scaler(nn.Module):
    """Reshape features"""
    def __init__(self, scale, inp, C, relu=True):
        """
        Arguments:
            scale (int) [-2, 2]: scale < 0 for downsample
            inp (int): input channel
            C (int): output channel
            relu (bool): set to False if the modules are pre-relu
        """
        super(Scaler, self).__init__()
        self.scale = scale
        if scale == 1:
            self.scaler = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=False),
                conv1x1_bn(inp, C, 1, relu=relu))
        # official implementation used bilinear for all scalers
        if scale == -1:
            self.scaler = conv_bn(inp, C, 2, kernel_size=3, relu=relu)

    def forward(self, hidden_state):
        if self.scale == 0:
            return hidden_state
        return self.scaler(hidden_state)


class Merger(nn.Module):
    """Concat and conv1x1"""
    def __init__(self, channels):
        """
        Arguments:
            channels (int): output channel
        """
        super(Merger, self).__init__()
        self.conv = conv3x3_bn(channels * 2, channels, 1)

    def forward(self, x0, x1):
        return self.conv(torch.cat([x0, x1], dim=1))


class DetNASNet(nn.Module):
    def __init__(self, cfg):
        super(DetNASNet, self).__init__()

        # load genotype
        geno_file = cfg.MODEL.HNASNET.GENOTYPE
        print("Loading genotype from {}".format(geno_file))
        geno_block, geno_path = torch.load(geno_file)

        # manually define geno_cell temporarily
        block_cfg = cfg.MODEL.HNASNET.BLOCK_CFG
        if block_cfg == 'mn1':
            geno_block = ([("ir_k3", [2, 1])]  # 0
                        + [("ir_k3", [2, 1])] * 2  # 0, 0
                        + [("ir_k3", [2, 2])] * 3  # 1, 2, 1
                        + [("ir_k5", [2, 2])] * 3  # 2, 2, 3
                        + [("ir_k3", [2, 2])] * 3) # 3, 2, 1
        if block_cfg == 'mn2':
            geno_block = ([("ir_k3", [1, 1])] * 3  # 0, 0, 0
                        + [("ir_k3", [1, 2])]  # 1
                        + [("ir_k5", [1, 2])]  # 2
                        + [("ir_k3", [1, 2])]  # 1
                        + [("ir_k5", [1, 2])] * 2  # 2, 2
                        + [("ir_k7", [1, 2])] * 2  # 3, 3
                        + [("ir_k5", [1, 2])]  # 2
                        + [("ir_k3", [1, 2])])  # 1
        if block_cfg == 'mn3':
            geno_block = ([("ir_k3", [1, 1])]  # 0
                        + [("ir_k3", [4, 1])] * 2  # 0, 0
                        + [("ir_k3", [4, 2])] * 3  # 1, 2, 1
                        + [("ir_k3", [4, 2])] * 3  # 2, 2, 3
                        + [("ir_k3", [4, 2])] * 3) # 3, 2, 1
        if block_cfg == 'sn1':
            geno_block = ([("sv2_k3", [1, 1])]  # 0
                        + [("sv2_k3", [1, 1])] * 2  # 0, 0
                        + [("sv2_k3", [2, 3])] * 3  # 1, 2, 1
                        + [("sv2_k3", [2, 4])] * 3  # 2, 2, 3
                        + [("sv2_k3", [4, 3])] * 3) # 3, 2, 1

        self.geno_path = geno_path

        # basic configs
        self.f = cfg.MODEL.HNASNET.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.HNASNET.NUM_LAYERS
        stride_mults = cfg.MODEL.HNASNET.STRIDE_MULTIPLIER
        self.num_strides = len(stride_mults)
        BxF = self.f

        self.stem1 = nn.Sequential(
            conv3x3_bn(3, 32, 2),
            conv3x3_bn(32, 32, 1))
        self.stem2 = conv3x3_bn(32, BxF * stride_mults[0], 2)

        in_channels = 32
        # feature pyramids
        self.bases = nn.ModuleList()
        for s in range(self.num_strides):
            out_channels = BxF * stride_mults[s]
            self.bases.append(conv3x3_bn(in_channels, out_channels, 2))
            in_channels = out_channels

        # create cells
        self.cells = nn.ModuleList()
        self.scalers = nn.ModuleList()
        # self.mergers = nn.ModuleList()

        h_0 = 0  # prev prev hidden index
        for layer, (geno, h) in enumerate(zip(geno_block, geno_path), 1):
            stride = stride_mults[h]
            oup = self.f * stride
            self.cells.append(FixBlock(geno, oup))
            # scalers
            inp0 = BxF * stride_mults[h_0]
            scaler = Scaler(h_0 - h, inp0, oup)
            h_0 = h
            self.scalers.append(scaler)

            # mergers
            # self.mergers.append(Merger(oup))

    def forward(self, x):
        h1 = self.stem1(x)
        h0 = self.stem2(h1)

        # get feature pyramids
        fps = []
        for base in self.bases:
            h1 = base(h1)
            fps.append(h1)

        s_1 = 0
        for i, (cell, s) in enumerate(zip(self.cells, self.geno_path)):
            input_0 = self.scalers[i](h0)
            # update feature pyramid at s_{-1}
            fps[s_1] = h0
            # h0 = self.mergers[i](cell(input_0), fps[s])
            h0 = cell(input_0) + fps[s]
            s_1 = s
        fps[s_1] = h0
        return fps
