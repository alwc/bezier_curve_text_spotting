import operator
import logging

from torch import nn


def conv_bn(inp, oup, stride, kernel_size=3, affine=True, relu=True):
    if not relu:
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(oup, affine=affine))
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
        nn.BatchNorm2d(oup, affine=affine),
        nn.ReLU(inplace=True)
    )


def sep_bn(inp, oup, stride, kernel_size=3, affine=True, relu=True):
    if not relu:
        return nn.Sequential(
            nn.Conv2d(inp, inp, kernel_size, stride, kernel_size // 2, groups=inp, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, affine=affine))
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, kernel_size // 2, groups=inp, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup, affine=affine),
        nn.ReLU(inplace=True),
    )


def conv3x3_bn(inp, oup, stride, affine=True, relu=True):
    if not relu:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup, affine=affine))
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup, affine=affine),
        nn.ReLU(inplace=True)
    )


def conv1x1_bn(inp, oup, stride, affine=True, relu=True):
    if not relu:
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.BatchNorm2d(oup, affine=affine))
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
        nn.BatchNorm2d(oup, affine=affine),
        nn.ReLU(inplace=True))


def sep3x3_bn(inp, oup, rate=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride=1,
                  padding=rate, dilation=rate, groups=inp,
                  bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def viterbi(trans):
    logger = logging.getLogger("maskrcnn_benchmark.hnas")
    """Dynamic programming to find the most likely path.

    Arguments:
        trans (LxSx3 array)"""
    prob = [1, 0, 0, 0]  # keeps the path with highest prob
    probs = [prob]
    paths = []
    for layer in trans:
        prob_next = [0, 0, 0, 0]
        path = [-1, -1, -1, -1]
        for i, stride in enumerate(layer):
            if i > 0:
                prob_up = stride[0] * prob[i]
                if prob_up > prob_next[i-1]:
                    prob_next[i-1] = prob_up
                    path[i-1] = 0
            prob_same = stride[1] * prob[i]
            if prob_same > prob_next[i]:
                prob_next[i] = prob_same
                path[i] = 1
            if i < 3:
                prob_down = stride[2] * prob[i]
                if prob_down > prob_next[i+1]:
                    prob_next[i+1] = prob_down
                    path[i+1] = 2
        prob = prob_next
        probs.append(prob)
        paths.append(path)

    max_ind, max_prob = max(enumerate(probs[-1]), key=operator.itemgetter(1))

    ml_path = [max_ind]
    for i in range(len(paths) - 1, 0, -1):
        path = paths[i]
        ml_path.insert(0, max_ind - path[max_ind] + 1)
        max_ind = max_ind - path[max_ind] + 1
    logger.info(ml_path)

    # check the prob
    ind = 0
    prob = 1
    layer_probs = []
    for i, layer in enumerate(trans):
        next_ind = ml_path[i]
        stride = layer[ind]
        layer_probs.append(layer[ind])
        prob = prob * stride[next_ind-ind+1]
        ind = next_ind
    logger.info(layer_probs)

    assert(max_prob - prob < 0.00001)
    return ml_path
