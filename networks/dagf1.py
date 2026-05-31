# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong.hit@gmail.com
@Software:   PyCharm
@File    :   spn.py
@Time    :   2020/8/6 08:12
@Desc    :
"""
import math

import torch
import numpy as np
from torch import nn
from torch.nn.functional import interpolate, softmax

import networks.common as common
import torch.nn.functional as F
from .wave import *
import networks.depth_encoder as encod


# feature = []


# 测试不同的金字塔网络

def make_model(args): return DAGF(args)

class PyModel(nn.Module):
    def __init__(self, num_features, out_channels=1):
        super(PyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.res_scale = common.Scale(0)

    def forward(self, inputs, add_feature):
        out = self.layer1(inputs)
        if add_feature is not None:
            out = self.layer3(self.res_scale(interpolate(add_feature, scale_factor=2, mode='nearest')) + out)
        return out, self.layer2(out)

# class FuseBlock(nn.Module):
#     def __init__(self, num_feature, act, norm, kernel_size, scale=2):
#         super(FuseBlock, self).__init__()
#
#         self.scale = scale
#         self.num = kernel_size * kernel_size
#
#         self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))
#
#         self.depth_kernel = nn.Sequential(
#             # common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=3, padding=1, act=act, norm=norm),
#             common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size ** 2, kernel_size=1)
#         )
#
#         self.guide_kernel = nn.Sequential(
#             # common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=3, padding=1, act=act, norm=norm),
#             common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size ** 2, kernel_size=1)
#         )
#
#         # self.norm_xca = encod.LayerNorm(kernel_size ** 2, eps=1e-6)
#         #
#         # self.gamma_xca = nn.Parameter(1e-6 * torch.ones(kernel_size ** 2),
#         #                               requires_grad=True) if 1e-6 > 0 else None
#         # self.xca = encod.XCA(dim=kernel_size ** 2, num_heads=9, qkv_bias=True, attn_drop=0, proj_drop=0)
#
#         # self.pix_shf = nn.PixelShuffle(upscale_factor=scale)
#
#         # self.weight_net = nn.Sequential(
#         #     common.ConvBNReLU2D(in_channels=num_feature * 2, out_channels=num_feature, kernel_size=3, padding=1,
#         #                         act=act, norm='Adaptive'),
#         #     # common.TUnet(num_features=num_feature, act=act, norm='Adaptive'),
#         #     common.ConvBNReLU2D(in_channels=num_feature, out_channels=1, kernel_size=3, act=act,
#         #                         padding=1, norm='Adaptive'),
#         # )
#         self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)
#         # self.inputs_conv = nn.Sequential(*[
#         #     common.ResNet(num_features=num_feature, act=act, norm=norm) for _ in range(num_res)])
#
#     def forward(self, depth, guide, inputs, ret_kernel=False):
#         b, c, h, w = inputs.size()
#         # h_, w_ = h * self.scale, w * self.scale
#         # weight_map = self.weight_net(torch.cat((depth, guide), 1))  # wu Softmax
#
#         depth_kernel = self.depth_kernel(depth)
#         guide_kernel = self.guide_kernel(guide)
#
#         # depth_kernel = softmax(depth_kernel.view(b, 9, -1), dim=2)
#         # guide_kernel = softmax(guide_kernel.view(b, 9, -1), dim=2)
#
#
#         # depth_kernel = depth_kernel.view(b,9,h,w)
#         # guide_kernel = guide_kernel.view(b,9,h,w)
#
#         # depth_kernel = softmax(depth_kernel, dim=1)
#         # guide_kernel = softmax(guide_kernel, dim=1)
#
#         # fuse_kernel = weight_map * guide_kernel + (1 - weight_map) * depth_kernel
#         # fuse_kernel = weight_map * guide_kernel - weight_map * depth_kernel
#         # fuse_kernel = guide_kernel - depth_kernel
#         fuse_kernel = softmax(guide_kernel - depth_kernel, dim=1)
#
#         # B, C, H, W = fuse_kernel.shape
#         # fuse_kernel = fuse_kernel.reshape(B, C, H * W).permute(0, 2, 1)
#         #
#         #
#         #
#         # fuse_kernel = fuse_kernel + self.gamma_xca * self.xca(self.norm_xca(fuse_kernel))
#         #
#         # fuse_kernel = fuse_kernel.reshape(B, C, H, W)
#
#         # fuse_kernel = softmax(fuse_kernel, dim=1)
#
#
#         fuse_kernel = torch.tanh(fuse_kernel) / (self.aff_scale_const + 1e-8)
#
#         abs_kernel = torch.abs(fuse_kernel)
#         abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
#
#         abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
#
#         fuse_kernel = fuse_kernel / abs_kernel_sum
#
#         # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
#         unfold_inputs = self.unfold(inputs).view(b, c, -1, h, w)
#         out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs)
#         # if ret_kernel:
#         #     return out, fuse_kernel, weight_map
#         return out + inputs


# class FuseBlock(nn.Module):
#     def __init__(self, num_feature, act, norm, kernel_size, scale=2):
#         super(FuseBlock, self).__init__()
#
#         self.scale = scale
#         self.num = kernel_size * kernel_size
#
#         self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))
#         self.aff_scale_const1 = nn.Parameter(0.5 * self.num * torch.ones(1))
#
#         self.depth_kernel = nn.Sequential(
#             # common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=3, padding=1, act=act, norm=norm),
#             common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size ** 2, kernel_size=1, act=act, norm=norm)
#         )
#
#         self.guide_kernel = nn.Sequential(
#             # common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=3, padding=1, act=act, norm=norm),
#             common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size ** 2, kernel_size=1, act=act, norm=norm)
#         )
#
#         # self.add_feature = nn.Sequential(
#         #     # common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=3, padding=1, act=act, norm=norm),
#         #     common.ConvBNReLU2D(in_channels=num_feature*2, out_channels=num_feature, kernel_size=1, act=act, norm=norm)
#         # )
#
#         # self.norm_xca = encod.LayerNorm(kernel_size ** 2, eps=1e-6)
#         #
#         # self.gamma_xca = nn.Parameter(1e-6 * torch.ones(kernel_size ** 2),
#         #                               requires_grad=True) if 1e-6 > 0 else None
#         # self.xca = encod.XCA(dim=kernel_size ** 2, num_heads=9, qkv_bias=True, attn_drop=0, proj_drop=0)
#
#         # self.pix_shf = nn.PixelShuffle(upscale_factor=scale)
#
#         # self.weight_net = nn.Sequential(
#         #     common.ConvBNReLU2D(in_channels=num_feature * 2, out_channels=num_feature, kernel_size=3, padding=1,
#         #                         act=act, norm='Adaptive'),
#         #     # common.TUnet(num_features=num_feature, act=act, norm='Adaptive'),
#         #     common.ConvBNReLU2D(in_channels=num_feature, out_channels=1, kernel_size=3, act=act,
#         #                         padding=1, norm='Adaptive'),
#         # )
#         self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)
#         self.unfold1 = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)
#         # self.inputs_conv = nn.Sequential(*[
#         #     common.ResNet(num_features=num_feature, act=act, norm=norm) for _ in range(num_res)])
#
#     def forward(self, depth, guide, inputs, ret_kernel=False):
#         b, c, h, w = inputs.size()
#         # h_, w_ = h * self.scale, w * self.scale
#         # weight_map = self.weight_net(torch.cat((depth, guide), 1))  # wu Softmax
#
#         depth_kernel = self.depth_kernel(depth)
#         guide_kernel = self.guide_kernel(guide)
#
#         # depth_kernel = softmax(depth_kernel.view(b, 9, -1), dim=2)
#         # guide_kernel = softmax(guide_kernel.view(b, 9, -1), dim=2)
#
#
#         # depth_kernel = depth_kernel.view(b,9,h,w)
#         # guide_kernel = guide_kernel.view(b,9,h,w)
#
#         depth_kernel = softmax(depth_kernel, dim=1)
#         guide_kernel = softmax(guide_kernel, dim=1)
#
#         # fuse_kernel = weight_map * guide_kernel + (1 - weight_map) * depth_kernel
#         # fuse_kernel = weight_map * guide_kernel - weight_map * depth_kernel
#         # fuse_kernel = guide_kernel - depth_kernel
#         # fuse_kernel = softmax(guide_kernel - depth_kernel, dim=1)
#
#         # B, C, H, W = fuse_kernel.shape
#         # fuse_kernel = fuse_kernel.reshape(B, C, H * W).permute(0, 2, 1)
#         #
#         #
#         #
#         # fuse_kernel = fuse_kernel + self.gamma_xca * self.xca(self.norm_xca(fuse_kernel))
#         #
#         # fuse_kernel = fuse_kernel.reshape(B, C, H, W)
#
#         # fuse_kernel = softmax(fuse_kernel, dim=1)
#
#
#         fuse_kernel = torch.tanh(depth_kernel) / (self.aff_scale_const + 1e-8)
#
#         abs_kernel = torch.abs(fuse_kernel)
#         abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
#
#         abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
#
#         fuse_kernel = fuse_kernel / abs_kernel_sum
#
#         # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
#         unfold_inputs = self.unfold(inputs).view(b, c, -1, h, w)
#         out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs)
#
#         fuse_kernel = torch.tanh(guide_kernel) / (self.aff_scale_const1 + 1e-8)
#
#         abs_kernel = torch.abs(fuse_kernel)
#         abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
#
#         abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
#
#         fuse_kernel = fuse_kernel / abs_kernel_sum
#
#         # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
#         unfold_inputs = self.unfold1(out).view(b, c, -1, h, w)
#         out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs)
#         # out1 = torch.cat((out1, out), 1)
#         # out = self.add_feature(out1)
#         # if ret_kernel:
#         #     return out, fuse_kernel, weight_map
#         return out + inputs


class FuseBlock(nn.Module):
    def __init__(self, num_feature, act, norm, kernel_size, scale=2):
        super(FuseBlock, self).__init__()

        self.scale = scale
        self.num = kernel_size * kernel_size

        self.aff_scale_const = nn.Parameter(0.5 * self.num * torch.ones(1))
        self.aff_scale_const1 = nn.Parameter(0.5 * self.num * torch.ones(1))

        self.depth_kernel = nn.Sequential(
            # common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=3, padding=1, act=act, norm=norm),
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size ** 2, kernel_size=1, act=act, norm=norm)
        )

        self.guide_kernel = nn.Sequential(
            # common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=3, padding=1, act=act, norm=norm),
            common.ConvBNReLU2D(in_channels=num_feature, out_channels=kernel_size ** 2, kernel_size=1, act=act, norm=norm)
        )

        # self.add_feature = nn.Sequential(
        #     # common.ConvBNReLU2D(in_channels=num_feature, out_channels=num_feature, kernel_size=3, padding=1, act=act, norm=norm),
        #     common.ConvBNReLU2D(in_channels=kernel_size ** 2, out_channels=kernel_size ** 2, kernel_size=3, padding=1, stride=2)
        # )

        # self.norm_xca = encod.LayerNorm(kernel_size ** 2, eps=1e-6)
        #
        # self.gamma_xca = nn.Parameter(1e-6 * torch.ones(kernel_size ** 2),
        #                               requires_grad=True) if 1e-6 > 0 else None
        # self.xca = encod.XCA(dim=kernel_size ** 2, num_heads=9, qkv_bias=True, attn_drop=0, proj_drop=0)

        # self.pix_shf = nn.PixelShuffle(upscale_factor=scale)

        # self.weight_net = nn.Sequential(
        #     common.ConvBNReLU2D(in_channels=num_feature * 2, out_channels=num_feature, kernel_size=3, padding=1,
        #                         act=act, norm='Adaptive'),
        #     # common.TUnet(num_features=num_feature, act=act, norm='Adaptive'),
        #     common.ConvBNReLU2D(in_channels=num_feature, out_channels=1, kernel_size=3, act=act,
        #                         padding=1, norm='Adaptive'),
        # )
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)
        self.unfold1 = nn.Unfold(kernel_size=kernel_size, dilation=scale, padding=kernel_size // 2 * scale)
        # self.inputs_conv = nn.Sequential(*[
        #     common.ResNet(num_features=num_feature, act=act, norm=norm) for _ in range(num_res)])

    def forward(self, depth, guide, inputs):
        b, c, h, w = inputs.size()
        # h_, w_ = h * self.scale, w * self.scale
        # weight_map = self.weight_net(torch.cat((depth, guide), 1))  # wu Softmax

        depth_kernel = self.depth_kernel(depth)

        # guide = guide - depth


        # depth_kernel = softmax(depth_kernel.view(b, 9, -1), dim=2)
        # guide_kernel = softmax(guide_kernel.view(b, 9, -1), dim=2)


        # depth_kernel = depth_kernel.view(b,9,h,w)
        # guide_kernel = guide_kernel.view(b,9,h,w)

        depth_kernel = softmax(depth_kernel, dim=1)


        # fuse_kernel = weight_map * guide_kernel + (1 - weight_map) * depth_kernel
        # fuse_kernel = weight_map * guide_kernel - weight_map * depth_kernel
        # fuse_kernel = guide_kernel - depth_kernel
        # guide_kernel = softmax(guide_kernel - depth_kernel, dim=1)

        # B, C, H, W = fuse_kernel.shape
        # fuse_kernel = fuse_kernel.reshape(B, C, H * W).permute(0, 2, 1)
        #
        #
        #
        # fuse_kernel = fuse_kernel + self.gamma_xca * self.xca(self.norm_xca(fuse_kernel))
        #
        # fuse_kernel = fuse_kernel.reshape(B, C, H, W)

        # fuse_kernel = softmax(fuse_kernel, dim=1)


        fuse_kernel = torch.tanh(depth_kernel) / (self.aff_scale_const + 1e-8)

        abs_kernel = torch.abs(fuse_kernel)
        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4

        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0

        fuse_kernel = fuse_kernel / abs_kernel_sum

        # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
        unfold_inputs = self.unfold(guide).view(b, c, -1, h, w)
        out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs)

        guide_kernel = self.guide_kernel(out)

        # if ret_kernel:
        #     kernel = self.add_feature(kernel)
        #     guide_kernel = guide_kernel + kernel

        guide_kernel = softmax(guide_kernel, dim=1)

        fuse_kernel = torch.tanh(guide_kernel) / (self.aff_scale_const1 + 1e-8)

        abs_kernel = torch.abs(fuse_kernel)
        abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4

        abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0

        fuse_kernel = fuse_kernel / abs_kernel_sum

        # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
        unfold_inputs = self.unfold1(inputs).view(b, c, -1, h, w)
        out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs)
        # out = out1-out
        # out1 = torch.cat((out1, out), 1)
        # out = self.add_feature(out1)
        # if ret_kernel:
        #     return out, fuse_kernel, weight_map
        return out

    # def forward(self, depth, guide, inputs, kernel, ret_kernel):
    #     b, c, h, w = inputs.size()
    #     # h_, w_ = h * self.scale, w * self.scale
    #     # weight_map = self.weight_net(torch.cat((depth, guide), 1))  # wu Softmax
    #
    #     depth_kernel = self.depth_kernel(depth)
    #     guide_kernel = self.guide_kernel(guide)
    #
    #
    #     if ret_kernel:
    #         kernel = self.add_feature(kernel)
    #         guide_kernel = guide_kernel + kernel
    #
    #     # depth_kernel = softmax(depth_kernel.view(b, 9, -1), dim=2)
    #     # guide_kernel = softmax(guide_kernel.view(b, 9, -1), dim=2)
    #
    #
    #     # depth_kernel = depth_kernel.view(b,9,h,w)
    #     # guide_kernel = guide_kernel.view(b,9,h,w)
    #
    #     depth_kernel = softmax(depth_kernel, dim=1)
    #     guide_kernel = softmax(guide_kernel, dim=1)
    #
    #     # fuse_kernel = weight_map * guide_kernel + (1 - weight_map) * depth_kernel
    #     # fuse_kernel = weight_map * guide_kernel - weight_map * depth_kernel
    #     # fuse_kernel = guide_kernel - depth_kernel
    #     # guide_kernel = softmax(guide_kernel - depth_kernel, dim=1)
    #
    #     # B, C, H, W = fuse_kernel.shape
    #     # fuse_kernel = fuse_kernel.reshape(B, C, H * W).permute(0, 2, 1)
    #     #
    #     #
    #     #
    #     # fuse_kernel = fuse_kernel + self.gamma_xca * self.xca(self.norm_xca(fuse_kernel))
    #     #
    #     # fuse_kernel = fuse_kernel.reshape(B, C, H, W)
    #
    #     # fuse_kernel = softmax(fuse_kernel, dim=1)
    #
    #
    #     fuse_kernel = torch.tanh(depth_kernel) / (self.aff_scale_const + 1e-8)
    #
    #     abs_kernel = torch.abs(fuse_kernel)
    #     abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
    #
    #     abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
    #
    #     fuse_kernel = fuse_kernel / abs_kernel_sum
    #
    #     # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
    #     unfold_inputs = self.unfold(inputs).view(b, c, -1, h, w)
    #     out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs)
    #
    #     fuse_kernel = torch.tanh(guide_kernel) / (self.aff_scale_const1 + 1e-8)
    #
    #     abs_kernel = torch.abs(fuse_kernel)
    #     abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
    #
    #     abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
    #
    #     fuse_kernel = fuse_kernel / abs_kernel_sum
    #
    #     # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
    #     unfold_inputs = self.unfold1(out).view(b, c, -1, h, w)
    #     out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs) + inputs
    #     # out = out1-out
    #     # out1 = torch.cat((out1, out), 1)
    #     # out = self.add_feature(out1)
    #     # if ret_kernel:
    #     #     return out, fuse_kernel, weight_map
    #     return out, guide_kernel

    # def forward(self, depth, guide, inputs):
    #     b, c, h, w = inputs.size()
    #     # h_, w_ = h * self.scale, w * self.scale
    #     # weight_map = self.weight_net(torch.cat((depth, guide), 1))  # wu Softmax
    #
    #     depth_kernel = self.depth_kernel(inputs)
    #     guide_kernel = self.guide_kernel(guide)
    #
    #     # depth_kernel = softmax(depth_kernel.view(b, 9, -1), dim=2)
    #     # guide_kernel = softmax(guide_kernel.view(b, 9, -1), dim=2)
    #
    #
    #     # depth_kernel = depth_kernel.view(b,9,h,w)
    #     # guide_kernel = guide_kernel.view(b,9,h,w)
    #
    #     depth_kernel = softmax(depth_kernel, dim=1)
    #     guide_kernel = softmax(guide_kernel, dim=1)
    #
    #     # fuse_kernel = weight_map * guide_kernel + (1 - weight_map) * depth_kernel
    #     # fuse_kernel = weight_map * guide_kernel - weight_map * depth_kernel
    #     # fuse_kernel = guide_kernel - depth_kernel
    #     # guide_kernel = softmax(guide_kernel - depth_kernel, dim=1)
    #
    #     # B, C, H, W = fuse_kernel.shape
    #     # fuse_kernel = fuse_kernel.reshape(B, C, H * W).permute(0, 2, 1)
    #     #
    #     #
    #     #
    #     # fuse_kernel = fuse_kernel + self.gamma_xca * self.xca(self.norm_xca(fuse_kernel))
    #     #
    #     # fuse_kernel = fuse_kernel.reshape(B, C, H, W)
    #
    #     # fuse_kernel = softmax(fuse_kernel, dim=1)
    #
    #
    #     fuse_kernel = torch.tanh(depth_kernel) / (self.aff_scale_const + 1e-8)
    #
    #     abs_kernel = torch.abs(fuse_kernel)
    #     abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
    #
    #     abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
    #
    #     fuse_kernel = fuse_kernel / abs_kernel_sum
    #
    #     # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
    #     unfold_inputs = self.unfold(depth).view(b, c, -1, h, w)
    #     out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs)
    #
    #     fuse_kernel = torch.tanh(guide_kernel) / (self.aff_scale_const1 + 1e-8)
    #
    #     abs_kernel = torch.abs(fuse_kernel)
    #     abs_kernel_sum = torch.sum(abs_kernel, dim=1, keepdim=True) + 1e-4
    #
    #     abs_kernel_sum[abs_kernel_sum < 1.0] = 1.0
    #
    #     fuse_kernel = fuse_kernel / abs_kernel_sum
    #
    #     # inputs_up = interpolate(self.inputs_conv(inputs), scale_factor=self.scale, mode='nearest')
    #     unfold_inputs = self.unfold1(guide-out).view(b, c, -1, h, w)
    #     out = torch.einsum('bkhw, bckhw->bchw', fuse_kernel, unfold_inputs)
    #     # out = out1-out
    #     # out1 = torch.cat((out1, out), 1)
    #     # out = self.add_feature(out1)
    #     # if ret_kernel:
    #     #     return out, fuse_kernel, weight_map
    #     return out + inputs


class InitLayer(nn.Module):
    def __init__(self, in_channels, num_features, flag=0):
        super(InitLayer, self).__init__()

        self.flag = flag

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_features, padding=1, kernel_size=3),
            nn.PReLU()
        )


        if flag == 0:
            self.layer2 = nn.Conv2d(in_channels=num_features, out_channels=num_features, padding=1, kernel_size=3)

        else:
            self.layer2 = nn.Conv2d(in_channels=2 * num_features, out_channels=num_features, padding=1, kernel_size=3)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_features, padding=1, kernel_size=3),
            nn.PReLU(),
            nn.Conv2d(in_channels=num_features, out_channels=num_features, padding=1, kernel_size=3)
        )

    def forward(self, inputs, lr):
        out = self.layer1(inputs)
        if self.flag == 0:
            out = self.layer2(out)
        else:
            out = self.layer2(torch.cat((out, lr), dim=1))
        return out

# class DGIF(nn.Module):
#     def __init__(self, num_features, act, norm, filter_size):
#         super(DGIF, self).__init__()
#         # self.args = args
#         # self.num_pyramid = num_pyramid
#         self.head = common.Head(num_features=num_features, act=act, norm=norm)
#         # self.depth_pyramid = nn.ModuleList([
#         #     common.DownSample(num_features=num_features, act=act, norm=norm) for _ in
#         #     range(self.num_pyramid - 1)])
#         #
#         # self.guide_pyramid = nn.ModuleList([
#         #     common.DownSample(num_features=num_features, act=act, norm=norm) for _ in
#         #     range(self.num_pyramid - 1)])
#
#         self.up_sample = FuseBlock(num_feature=num_features, act=act, norm=norm, kernel_size=filter_size, scale=2)
#
#         # self.init_conv = nn.ModuleList(
#         #     [InitLayer(1, num_features=num_features, flag=i) for i in range(num_pyramid)])
#         #
#         # self.p_layers = nn.ModuleList([
#         #     PyModel(num_features, out_channels=1) for _ in range(num_pyramid)
#         # ])
#         #
#         # self.tail_conv = nn.Sequential(
#         #     common.ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1,
#         #                         act=act),
#         #     common.ConvBNReLU2D(in_channels=num_features, out_channels=1, kernel_size=3, padding=1, act=act),
#         # )
#         #
#         # self.res_scale = common.Scale(0)
#
#     def forward(self, lr, rgb):
#
#
#         b, c, h, w = rgb.size()
#         sobel_x = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
#         sobel_y = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]
#
#         sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
#         sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
#         sobel_x = sobel_x.type_as(rgb)
#         sobel_y = sobel_y.type_as(rgb)
#         weight_x = torch.nn.Parameter(data=sobel_x, requires_grad=False)
#         weight_y = torch.nn.Parameter(data=sobel_y, requires_grad=False)
#         Ix = F.conv2d(rgb, weight_x, stride=1, padding=1, groups=c)
#         Iy = F.conv2d(rgb, weight_y, stride=1, padding=1, groups=c)
#
#         Ixy = Ix * Ix + Iy * Iy
#         lr_feature, guide_feature = self.head(Ixy, rgb)
#         # depth_features, guide_features = [lr_feature], [guide_feature]
#         # for num_p in range(self.num_pyramid - 1):
#         #     lr_feature = self.depth_pyramid[num_p](lr_feature)
#         #     depth_features.append(lr_feature)
#         #     guide_feature = self.guide_pyramid[num_p](guide_feature)
#         #     guide_features.append(guide_feature)
#         # 从小到大
#         # depth_features, guide_features = list(reversed(depth_features)), list(reversed(guide_features))
#         # lr_input = None
#         # ret_feature = []
#         # for i in range(self.num_pyramid):
#         #     h, w = depth_features[i].size()[2:]
#
#             # lr_input = self.init_conv[i](interpolate(lr, size=(h // 2, w // 2), mode='nearest'), lr_input)
#         out = self.up_sample(lr_feature, guide_feature, lr)
#
#             # ret_feature.append(lr_input)
#
#         # out = []
#         #
#         # out1, out2 = None, None
#         # for i in range(self.args.num_pyramid):
#         #     out1, out2 = self.p_layers[i](ret_feature[i], out1)
#         #     out2 = interpolate(out2, size=rgb.size()[2:], mode='bilinear', align_corners=False) + lr_up
#         #     out.append(out2)
#
#         return out


class DGIF(nn.Module):
    def __init__(self, num_features, act, norm, filter_size):
        super(DGIF, self).__init__()
        # self.args = args
        # self.num_pyramid = num_pyramid
        self.head = common.Head(num_features=num_features, act=act, norm=norm, guide_channels=6)
        self.head1 = common.Head(num_features=num_features, act=act, norm=norm, in_channels=6)
        self.wave = WavePool(3)
        self.gamma = nn.Parameter(1e-6 * torch.ones((num_features)),
                                  requires_grad=True)
        # self.depth_pyramid = nn.ModuleList([
        #     common.DownSample(num_features=num_features, act=act, norm=norm) for _ in
        #     range(self.num_pyramid - 1)])
        #
        # self.guide_pyramid = nn.ModuleList([
        #     common.DownSample(num_features=num_features, act=act, norm=norm) for _ in
        #     range(self.num_pyramid - 1)])

        self.up_sample = FuseBlock(num_feature=num_features, act=act, norm=norm, kernel_size=filter_size, scale=2)
        self.up_sample1 = FuseBlock(num_feature=num_features, act=act, norm=norm, kernel_size=filter_size, scale=2)

        # self.init_conv = nn.ModuleList(
        #     [InitLayer(1, num_features=num_features, flag=i) for i in range(num_pyramid)])
        #
        # self.p_layers = nn.ModuleList([
        #     PyModel(num_features, out_channels=1) for _ in range(num_pyramid)
        # ])
        #
        # self.tail_conv = nn.Sequential(
        #     common.ConvBNReLU2D(in_channels=num_features, out_channels=num_features, kernel_size=3, padding=1,
        #                         act=act),
        #     common.ConvBNReLU2D(in_channels=num_features, out_channels=1, kernel_size=3, padding=1, act=act),
        # )
        #
        # self.res_scale = common.Scale(0)

    def forward(self, lr, rgb_o):
        rgb = rgb_o[1]
        rgb_l = rgb_o[0]
        b, c, h, w = rgb.size()
        x_LL, x_waves = self.wave(rgb_l)
        sobel_x = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
        sobel_y = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]

        sobel_x = torch.FloatTensor(sobel_x).expand(c, 1, 3, 3)
        sobel_y = torch.FloatTensor(sobel_y).expand(c, 1, 3, 3)
        sobel_x = sobel_x.type_as(rgb)
        sobel_y = sobel_y.type_as(rgb)
        weight_x = torch.nn.Parameter(data=sobel_x, requires_grad=False)
        weight_y = torch.nn.Parameter(data=sobel_y, requires_grad=False)
        Ix = F.conv2d(rgb, weight_x, stride=1, padding=1, groups=c)
        Iy = F.conv2d(rgb, weight_y, stride=1, padding=1, groups=c)

        Ixy = Ix * Ix + Iy * Iy
        lr_feature, guide_feature = self.head(Ixy, x_LL)
        lr_feature1, guide_feature1 = self.head1(x_waves, rgb)
        # depth_features, guide_features = [lr_feature], [guide_feature]
        # for num_p in range(self.num_pyramid - 1):
        #     lr_feature = self.depth_pyramid[num_p](lr_feature)
        #     depth_features.append(lr_feature)
        #     guide_feature = self.guide_pyramid[num_p](guide_feature)
        #     guide_features.append(guide_feature)
        # 从小到大
        # depth_features, guide_features = list(reversed(depth_features)), list(reversed(guide_features))
        # lr_input = None
        # ret_feature = []
        # for i in range(self.num_pyramid):
        #     h, w = depth_features[i].size()[2:]

        # lr_input = self.init_conv[i](interpolate(lr, size=(h // 2, w // 2), mode='nearest'), lr_input)
        out1 = self.up_sample(lr_feature, guide_feature, lr)
        out2 = self.up_sample(lr_feature1, guide_feature1, lr)
        lr = self.gamma * lr.permute(0, 2, 3, 1)
        lr = lr.permute(0, 3, 1, 2)
        out = torch.max(out1, out2) + lr

        # ret_feature.append(lr_input)

        # out = []
        #
        # out1, out2 = None, None
        # for i in range(self.args.num_pyramid):
        #     out1, out2 = self.p_layers[i](ret_feature[i], out1)
        #     out2 = interpolate(out2, size=rgb.size()[2:], mode='bilinear', align_corners=False) + lr_up
        #     out.append(out2)

        return out


class DAGF(nn.Module):
    def __init__(self, args):
        super(DAGF, self).__init__()
        self.args = args
        self.num_pyramid = args.num_pyramid
        self.head = common.Head(num_features=args.num_features, expand_ratio=1, act=args.act,
                                guide_channels=args.guide_channels)
        self.depth_pyramid = nn.ModuleList([
            common.DownSample(num_features=args.num_features, act=args.act, norm=args.norm) for _ in
            range(self.num_pyramid - 1)])

        self.guide_pyramid = nn.ModuleList([
            common.DownSample(num_features=args.num_features, act=args.act, norm=args.norm) for _ in
            range(self.num_pyramid - 1)])

        self.up_sample = nn.ModuleList([
            FuseBlock(num_feature=args.num_features, act=args.act, norm=args.norm, kernel_size=args.filter_size,
                      num_res=args.num_res, scale=2) for _ in range(self.num_pyramid)
        ])

        self.init_conv = nn.ModuleList(
            [InitLayer(1, num_features=args.num_features, flag=i) for i in range(args.num_pyramid)])

        self.p_layers = nn.ModuleList([
            PyModel(args.num_features, out_channels=1) for _ in range(args.num_pyramid)
        ])

        self.tail_conv = nn.Sequential(
            common.ConvBNReLU2D(in_channels=args.num_features, out_channels=args.num_features, kernel_size=3, padding=1,
                                act=args.act),
            common.ConvBNReLU2D(in_channels=args.num_features, out_channels=1, kernel_size=3, padding=1, act=args.act),
        )

        self.res_scale = common.Scale(0)

    def forward(self, lr, rgb, lr_up):
        lr_feature, guide_feature = self.head(lr_up, rgb)
        depth_features, guide_features = [lr_feature], [guide_feature]
        for num_p in range(self.num_pyramid - 1):
            lr_feature = self.depth_pyramid[num_p](lr_feature)
            depth_features.append(lr_feature)
            guide_feature = self.guide_pyramid[num_p](guide_feature)
            guide_features.append(guide_feature)
        # 从小到大
        depth_features, guide_features = list(reversed(depth_features)), list(reversed(guide_features))
        lr_input = None
        ret_feature = []
        for i in range(self.num_pyramid):
            h, w = depth_features[i].size()[2:]

            lr_input = self.init_conv[i](interpolate(lr, size=(h // 2, w // 2), mode='nearest'), lr_input)
            lr_input = self.up_sample[i](depth_features[i], guide_features[i], lr_input)

            ret_feature.append(lr_input)

        out = []

        out1, out2 = None, None
        for i in range(self.args.num_pyramid):
            out1, out2 = self.p_layers[i](ret_feature[i], out1)
            out2 = interpolate(out2, size=rgb.size()[2:], mode='bilinear', align_corners=False) + lr_up
            out.append(out2)

        return out