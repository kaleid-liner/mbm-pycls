#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""AnyNet models."""

from pycls.core.config import cfg
from pycls.models.blocks import (
    activation,
    conv2d,
    conv2d_cx,
    gap2d,
    gap2d_cx,
    init_weights,
    linear,
    linear_cx,
    norm2d,
    norm2d_cx,
    pool2d,
    pool2d_cx,
    SE,
    channel_shuffle,
)
from torch.nn import Module
import torch.nn as nn
import torch


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStem,
        "simple_stem_in": SimpleStem,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
        "res_bottleneck_linear_block": ResBottleneckLinearBlock,
        "inverted_residual": InvertedResidual,
        "mbconv": MBConv,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyHead(Module):
    """AnyNet head: optional conv, AvgPool, 1x1."""

    def __init__(self, w_in, head_width, num_classes):
        super(AnyHead, self).__init__()
        self.head_width = head_width
        if head_width > 0:
            self.conv = conv2d(w_in, head_width, 1)
            self.bn = norm2d(head_width)
            self.af = activation()
            w_in = head_width
        self.avg_pool = gap2d(w_in)
        self.fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.af(self.bn(self.conv(x))) if self.head_width > 0 else x
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @staticmethod
    def complexity(cx, w_in, head_width, num_classes):
        if head_width > 0:
            cx = conv2d_cx(cx, w_in, head_width, 1)
            cx = norm2d_cx(cx, head_width)
            w_in = head_width
        cx = gap2d_cx(cx, w_in)
        cx = linear_cx(cx, w_in, num_classes, bias=True)
        return cx


class VanillaBlock(Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(VanillaBlock, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class BasicTransform(Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, w_in, w_out, stride, _params):
        super(BasicTransform, self).__init__()
        self.a = conv2d(w_in, w_out, 3, stride=stride)
        self.a_bn = norm2d(w_out)
        self.a_af = activation()
        self.b = conv2d(w_out, w_out, 3)
        self.b_bn = norm2d(w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, _params):
        cx = conv2d_cx(cx, w_in, w_out, 3, stride=stride)
        cx = norm2d_cx(cx, w_out)
        cx = conv2d_cx(cx, w_out, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBasicBlock(Module):
    """Residual basic block: x + f(x), f = basic transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBasicBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BasicTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = BasicTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class BottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        self.a = conv2d(w_in, w_b, 1)
        self.a_bn = norm2d(w_b)
        self.a_af = activation()
        self.b = conv2d(w_b, w_b, 3, stride=stride, groups=groups)
        self.b_bn = norm2d(w_b)
        self.b_af = activation()
        self.se = SE(w_b, w_se) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = norm2d(w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_b = int(round(w_out * params["bot_mul"]))
        w_se = int(round(w_in * params["se_r"]))
        groups = w_b // params["group_w"]
        cx = conv2d_cx(cx, w_in, w_b, 1)
        cx = norm2d_cx(cx, w_b)
        cx = conv2d_cx(cx, w_b, w_b, 3, stride=stride, groups=groups)
        cx = norm2d_cx(cx, w_b)
        cx = SE.complexity(cx, w_b, w_se) if w_se else cx
        cx = conv2d_cx(cx, w_b, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or (stride != 1):
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = norm2d(w_out)
        self.f = BottleneckTransform(w_in, w_out, stride, params)
        self.af = activation()

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(x_p + self.f(x))

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        if (w_in != w_out) or (stride != 1):
            h, w = cx["h"], cx["w"]
            cx = conv2d_cx(cx, w_in, w_out, 1, stride=stride)
            cx = norm2d_cx(cx, w_out)
            cx["h"], cx["w"] = h, w
        cx = BottleneckTransform.complexity(cx, w_in, w_out, stride, params)
        return cx


class ResBottleneckLinearBlock(Module):
    """Residual linear bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckLinearBlock, self).__init__()
        self.has_skip = (w_in == w_out) and (stride == 1)
        self.f = BottleneckTransform(w_in, w_out, stride, params)

    def forward(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        return BottleneckTransform.complexity(cx, w_in, w_out, stride, params)


class InvertedResidual(Module):

    def __init__(self, w_in, w_out, stride, params=None):
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = w_out // 2
        if (self.stride == 1) and (w_in != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {w_in} and oup {w_out} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                conv2d(w_in, w_in, 3, stride=self.stride, groups=w_in),
                norm2d(w_in),
                conv2d(w_in, branch_features, 1),
                norm2d(branch_features),
                activation(),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            conv2d(
                w_in if (self.stride > 1) else branch_features,
                branch_features,
                1
            ),
            norm2d(branch_features),
            activation(),
            conv2d(branch_features, branch_features, 3, stride=self.stride, groups=branch_features),
            norm2d(branch_features),
            conv2d(branch_features, branch_features, 1),
            norm2d(branch_features),
            activation(),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class MBConv(Module):

    def __init__(self, w_in, w_out, stride, params=None):
        super().__init__()
        self.exp = None
        w_exp = int(w_in * params["bot_mul"])
        k = params["k"]
        if w_exp != w_in:
            self.exp = conv2d(w_in, w_exp, 1)
            self.exp_bn = norm2d(w_exp)
            self.exp_af = activation()
        self.dwise = conv2d(w_exp, w_exp, k, stride=stride, groups=w_exp)
        self.dwise_bn = norm2d(w_exp)
        self.dwise_af = activation()
        self.se = SE(w_exp, int(w_in * params["se_r"]))
        self.lin_proj = conv2d(w_exp, w_out, 1)
        self.lin_proj_bn = norm2d(w_out)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = self.exp_af(self.exp_bn(self.exp(x))) if self.exp else x
        f_x = self.dwise_af(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            f_x = x + f_x
        return f_x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, params):
        w_exp = int(w_in * params["bot_mul"])
        k = params["k"]
        if w_exp != w_in:
            cx = conv2d_cx(cx, w_in, w_exp, 1)
            cx = norm2d_cx(cx, w_exp)
        cx = conv2d_cx(cx, w_exp, w_exp, k, stride=stride, groups=w_exp)
        cx = norm2d_cx(cx, w_exp)
        cx = SE.complexity(cx, w_exp, int(w_in * params["se_r"]))
        cx = conv2d_cx(cx, w_exp, w_out, 1)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStemCifar(Module):
    """ResNet stem for CIFAR: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, k=3):
        super(ResStemCifar, self).__init__()
        self.conv = conv2d(w_in, w_out, 3)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, k=3):
        cx = conv2d_cx(cx, w_in, w_out, 3)
        cx = norm2d_cx(cx, w_out)
        return cx


class ResStem(Module):
    """ResNet stem for ImageNet: 7x7, BN, AF, MaxPool."""

    def __init__(self, w_in, w_out, k=7):
        super(ResStem, self).__init__()
        self.conv = conv2d(w_in, w_out, k, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()
        self.pool = pool2d(w_out, 3, stride=2)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, k=7):
        cx = conv2d_cx(cx, w_in, w_out, k, stride=2)
        cx = norm2d_cx(cx, w_out)
        cx = pool2d_cx(cx, w_out, 3, stride=2)
        return cx


class SimpleStem(Module):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out, k=3):
        super(SimpleStem, self).__init__()
        self.conv = conv2d(w_in, w_out, k, stride=2)
        self.bn = norm2d(w_out)
        self.af = activation()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, k=3):
        cx = conv2d_cx(cx, w_in, w_out, k, stride=2)
        cx = norm2d_cx(cx, w_out)
        return cx


class AnyStage(Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, params):
        super(AnyStage, self).__init__()
        for i in range(d):
            block = block_fun(w_in, w_out, stride, params)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, block_fun, params):
        for _ in range(d):
            cx = block_fun.complexity(cx, w_in, w_out, stride, params)
            stride, w_in = 1, w_out
        return cx


class BranchStage(Module):
    def __init__(self, branches, ow, w_ins, w_outs=None):
        super().__init__()

        if w_outs is None:
            w_outs = [ow // len(branches)] * len(branches)
        self.branches = nn.ModuleList([
            nn.Sequential(
                branch,
                conv2d(w_in, w_out, 1),
                norm2d(w_out)
            )
            if w_in != w_out
            else branch
            for w_in, w_out, branch in zip(w_ins, w_outs, branches)
        ])

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], 1)


class AnyNet(Module):
    """AnyNet model."""

    @staticmethod
    def get_params():
        nones = [None for _ in cfg.ANYNET.DEPTHS]
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "stem_k": cfg.ANYNET.STEM_K,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "depths": cfg.ANYNET.DEPTHS,
            "widths": cfg.ANYNET.WIDTHS,
            "strides": cfg.ANYNET.STRIDES,
            "kernels": cfg.ANYNET.KERNELS if cfg.ANYNET.KERNELS else nones,
            "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
            "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
            "head_w": cfg.ANYNET.HEAD_W,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
            "devices": cfg.ANYNET.DEVICES,
            "original_widths": cfg.ANYNET.ORIGINAL_WIDTHS
        }

    def __init__(self, params=None):
        super(AnyNet, self).__init__()
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        self.stem = stem_fun(3, p["stem_w"], p["stem_k"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws", "original_widths", "kernels"]

        for i, (ds, ws, ss, b, gs, o, k) in enumerate(zip(*[p[k] for k in keys])):
            stage_branches = []
            if gs is None:
                gs = [None for _ in ds]
            for j, (d, w, s, g) in enumerate(zip(ds, ws, ss, gs)):
                params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"], "k": k}
                stage_branches.append(AnyStage(prev_w, w, s, d, block_fun, params))
            self.add_module("s{}".format(i + 1), BranchStage(stage_branches, o, ws, ws))
            prev_w = o

        self.head = AnyHead(prev_w, p["head_w"], p["num_classes"])
        self.apply(init_weights)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        p = AnyNet.get_params() if not params else params
        stem_fun = get_stem_fun(p["stem_type"])
        block_fun = get_block_fun(p["block_type"])
        cx = stem_fun.complexity(cx, 3, p["stem_w"], p["stem_k"])
        prev_w = p["stem_w"]
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws", "original_widths", "kernels"]
        for i, (ds, ws, ss, b, gs, o, k) in enumerate(zip(*[p[k] for k in keys])):
            if gs is None:
                gs = [None for _ in ds]
            old_h, old_w = cx["h"], cx["w"]
            for j, (d, w, s, g) in enumerate(zip(ds, ws, ss, gs)):
                params = {"bot_mul": b, "group_w": g, "se_r": p["se_r"], "k": k}
                cx["h"], cx["w"] = old_h, old_w
                cx = AnyStage.complexity(cx, prev_w, w, s, d, block_fun, params)
            prev_w = o
        cx = AnyHead.complexity(cx, prev_w, p["head_w"], p["num_classes"])
        return cx
