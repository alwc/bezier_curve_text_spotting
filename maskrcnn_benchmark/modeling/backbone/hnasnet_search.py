"""
Implements Auto-DeepLab framework
"""

import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.nas.cell import Cell
from maskrcnn_benchmark.nas.genotypes import PRIMITIVES
from .hnas_common import conv3x3_bn, conv1x1_bn, viterbi


class Router(nn.Module):
    """ Propagate hidden states to next layer
    """

    def __init__(self, ind, inp, C, num_strides=4, affine=True):
        """
        Arguments:
            ind (int) [2-5]: index of the cell, which decides output scales
            inp (int): inp size
            C (int): output size of the same scale
            num_strides (int): number of feature pyramids in the network
        """
        super(Router, self).__init__()
        self.ind = ind
        self.num_strides = num_strides

        if ind > 0:
            # upsample
            self.postprocess0 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                conv1x1_bn(inp, C // 2, 1, affine=affine, relu=False))
        self.postprocess1 = conv1x1_bn(inp, C, 1, affine=affine, relu=False)
        if ind < num_strides - 1:
            # downsample
            self.postprocess2 = conv3x3_bn(inp, C * 2, 2, affine=affine, relu=False)

    def forward(self, out):
        """
        Returns:
            h_next ([Tensor]): None for empty
        """
        if self.ind > 0:
            h_next_0 = self.postprocess0(out)
        else:
            h_next_0 = None
        h_next_1 = self.postprocess1(out)
        if self.ind < self.num_strides - 1:
            h_next_2 = self.postprocess2(out)
        else:
            h_next_2 = None
        return h_next_0, h_next_1, h_next_2


class HNASNetSearch(nn.Module):
    """
    Main class for Auto-DeepLab.

    Use one cell per hidden states
    """

    def __init__(self, cfg):
        super(HNASNetSearch, self).__init__()
        self.f = cfg.MODEL.HNASNET.FILTER_MULTIPLIER
        self.num_layers = cfg.MODEL.HNASNET.NUM_LAYERS
        self.num_blocks = cfg.MODEL.HNASNET.NUM_BLOCKS
        self.primitives = PRIMITIVES[cfg.MODEL.HNASNET.PRIMITIVES]
        affine = cfg.MODEL.HNASNET.AFFINE
        self.stem1 = conv3x3_bn(3, 64, 2, affine=affine)
        self.stem2 = conv3x3_bn(64, self.f * self.num_blocks, 2,
                                affine=affine)
        stride_mults = cfg.MODEL.HNASNET.STRIDE_MULTIPLIER
        self.num_strides = len(stride_mults)
        # generates first h_1
        self.reduce1 = conv3x3_bn(64, self.f, 2, affine=affine)

        # upsample module for other strides
        self.upsamplers = nn.ModuleList()
        for i in range(1, self.num_strides):
            self.upsamplers.append(conv3x3_bn(self.f * stride_mults[i-1],
                                              self.f * stride_mults[i],
                                              2, affine=affine))

        self.cells = nn.ModuleList()
        self.routers = nn.ModuleList()
        self.cell_configs = []
        self.tie_cell = cfg.DARTS.TIE_CELL

        for l in range(1, self.num_layers + 1):
            for h in range(min(self.num_strides, l + 1)):
                stride = stride_mults[h]
                C = self.f * stride

                if h < l:
                    self.routers.append(Router(h, C * self.num_blocks,
                                               C, affine=affine))

                self.cell_configs.append(
                    "L{}H{}: {}".format(l, h, C))
                self.cells.append(Cell(self.num_blocks, C,
                                       self.primitives,
                                       affine=affine))
        self.init_alphas()

    def w_parameters(self):
        return [value for key, value in self.named_parameters()
                if 'arch' not in key and value.requires_grad]

    def a_parameters(self):
        a_params = [value for key, value in self.named_parameters() if 'arch' in key]
        return a_params

    def init_alphas(self):
        k = sum(2 + i for i in range(self.num_blocks))
        num_ops = len(self.primitives)
        if self.tie_cell:
            self.arch_alphas = nn.Parameter(torch.ones(k, num_ops))
        else:
            self.arch_alphas = nn.Parameter(torch.ones(len(self.cells), k, num_ops))

        m = sum(min(l + 1, self.num_strides) for l in range(self.num_layers))
        beta_weights = torch.ones(m, 3)
        # mask out
        top_inds = []
        btm_inds = []
        start = 0
        for l in range(self.num_layers):
            top_inds.append(start)
            if l + 1 < self.num_strides:
                start += l + 1
            else:
                start += self.num_strides
                btm_inds.append(start - 1)

        beta_weights[top_inds, 0] = -50
        beta_weights[btm_inds, 2] = -50
        self.arch_betas = nn.Parameter(beta_weights)
        self.score_func = F.softmax

    def scores(self):
        return (self.score_func(self.arch_alphas, dim=-1),
                self.score_func(self.arch_betas, dim=-1))

    def forward(self, x):
        """
        Arguments:
            images (list[Tensor]): images to be processed

        Returns:
            result (list[Tensor]): the output from the model.
        """
        # compute architecture params
        alphas, betas = self.scores()

        # The first layer is different
        features = self.stem1(x)
        inputs_1 = [self.reduce1(features)]
        features = self.stem2(features)

        hidden_states = [features]

        cell_ind = 0
        router_ind = 0
        for l in range(self.num_layers):
            # prepare next inputs
            inputs_0 = [0] * min(l + 2, self.num_strides)
            for i, hs in enumerate(hidden_states):
                # print('router {}: '.format(router_ind), self.cell_configs[router_ind])
                h_0, h_1, h_2 = self.routers[router_ind](hs)
                # print(h_0 is None, h_1 is None, h_2 is None)
                # print(betas[router_ind])
                if i > 0:
                    inputs_0[i - 1] = inputs_0[i - 1] + h_0 * betas[router_ind][0]
                inputs_0[i] = inputs_0[i] + h_1 * betas[router_ind][1]
                if i < self.num_strides - 1:
                    inputs_0[i + 1] = inputs_0[i + 1] + h_2 * betas[router_ind][2]
                router_ind += 1

            # run cells
            old_hs = hidden_states
            if l < 3:
                old_hs.append(0)
            hidden_states = []
            for i, s0 in enumerate(inputs_0):
                # prepare next input
                if i >= len(inputs_1):
                    # print("using upsampler {}.".format(i-1))
                    inputs_1.append(self.upsamplers[i - 1](inputs_1[-1]))
                s1 = inputs_1[i]
                # print('cell: ', self.cell_configs[cell_ind])
                if self.tie_cell:
                    cell_weights = alphas
                else:
                    cell_weights = alphas[cell_ind]
                # add a residual connection
                hidden_states.append(
                    old_hs[i] + self.cells[cell_ind](s0, s1, cell_weights))
                cell_ind += 1

            inputs_1 = inputs_0

        return hidden_states

    def get_path_genotype(self, betas):
        # construct transition matrix
        trans = []
        b_ind = 0
        for l in range(self.num_layers):
            layer = []
            for i in range(self.num_strides):
                if i < l + 1:
                    layer.append(betas[b_ind].detach().cpu().numpy().tolist())
                    b_ind += 1
                else:
                    layer.append([0, 0, 0])
            trans.append(layer)
        return viterbi(trans)

    def genotype(self):
        alphas, betas = self.scores()
        if self.tie_cell:
            gene_cell = self.cells[0].genotype(alphas)
        else:
            gene_cell = []
            for i, cell in enumerate(self.cells):
                gene_cell.append(cell.genotype(alphas[i]))
        gene_path = self.get_path_genotype(betas)
        return gene_cell, gene_path
