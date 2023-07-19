import torch 
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
from medpy.io import load

from Functions import generate_grid, save_img, generate_grid_unit, transform_unit_flow_to_flow_cuda
import numpy as np

import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, num_block=5):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        # self.com_transform = CompositionTransform().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)
        # self.input_encoder_lvl2 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)
        # self.input_encoder_lvl3 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, num_block=num_block, bias_opt=bias_opt)
        # self.resblock_group_lvl2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # self.resblock_group_lvl3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        # self.up = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                               padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.output_lvl2 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)
        # self.output_lvl3 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def resblock_seq(self, in_channels, num_block, bias_opt=False):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer


    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, reg_code):

        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        down_y = cat_input_lvl1[:, 1:2, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1)


        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        else:
            return output_disp_e0_v


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl1=None, num_block=5):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1
        # self.model_lvl1 = [model_lvl1[i] for i in range(len(model_lvl1)-1)]
        # self.model_lvl1 = nn.Sequential(*self.model_lvl1)

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        self.com_transform = CompositionTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)
        # self.input_encoder_lvl2 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)
        # self.input_encoder_lvl3 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, num_block=num_block, bias_opt=bias_opt)
        # self.resblock_group_lvl2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # self.resblock_group_lvl3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        # self.output_lvl2 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)
        # self.output_lvl3 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, num_block, bias_opt=False):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, reg_code):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y, reg_code)

        # lvl1_disp, lvl1_warp, lvl1_y, lvl1_v, lvl1_embedding = self.model_lvl1(x, y, reg_code)
        lvl1_disp_up = self.up_tri(lvl1_disp)

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl1_embedding

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0
            # return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0, lvl1_warp, lvl1_y
        else:
            return compose_field_e0_lvl1


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 model_lvl2=None, num_block=5):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        self.com_transform = CompositionTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, num_block=num_block, bias_opt=bias_opt)


        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        # self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, num_block, bias_opt=False):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, reg_code):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(x, y, reg_code)

        # lvl2_disp, lvl2_warp, lvl2_y, lvl2_v, lvl1_v, lvl2_embedding, lvl1_warp, lvl1_y = self.model_lvl2(x, y, reg_code)


        lvl2_disp_up = self.up_tri(lvl2_disp)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_disp, e0
            # return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0, lvl1_warp, lvl1_y, lvl2_warp, lvl2_y
        else:
            return compose_field_e0_lvl1


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, num_block=5, num_con=1):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        # self.com_transform = CompositionTransform().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)
        # self.input_encoder_lvl2 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)
        # self.input_encoder_lvl3 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, num_block=num_block, bias_opt=bias_opt, num_con=num_con)
        # self.resblock_group_lvl2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # self.resblock_group_lvl3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        # self.up = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                               padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.output_lvl2 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)
        # self.output_lvl3 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def resblock_seq(self, in_channels, num_block, bias_opt=False, num_con=1):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt, num_con=num_con))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer


    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, reg_code):

        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        down_y = cat_input_lvl1[:, 2:4, :, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0_v.permute(0, 2, 3, 4, 1), self.grid_1)


        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        else:
            return output_disp_e0_v


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4, model_lvl1=None, num_block=5, num_con=1):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1
        # self.model_lvl1 = [model_lvl1[i] for i in range(len(model_lvl1)-1)]
        # self.model_lvl1 = nn.Sequential(*self.model_lvl1)

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        self.com_transform = CompositionTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)
        # self.input_encoder_lvl2 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)
        # self.input_encoder_lvl3 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, num_block=num_block, bias_opt=bias_opt, num_con=num_con)
        # self.resblock_group_lvl2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # self.resblock_group_lvl3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        # self.output_lvl2 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)
        # self.output_lvl3 = self.outputs(self.start_channel * 4, self.n_classes, kernel_size=5, stride=1, padding=2,
        #                            bias=False)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, num_block, bias_opt=False, num_con=1):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt, num_con=num_con))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, reg_code):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, _, _, lvl1_v, lvl1_embedding = self.model_lvl1(x, y, reg_code)

        # lvl1_disp, lvl1_warp, lvl1_y, lvl1_v, lvl1_embedding = self.model_lvl1(x, y, reg_code)
        lvl1_disp_up = self.up_tri(lvl1_disp)

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl1_embedding

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0
            # return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0, lvl1_warp, lvl1_y
        else:
            return compose_field_e0_lvl1


class Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192, 144), range_flow=0.4,
                 model_lvl2=None, num_block=5, num_con=1):
        super(Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = generate_grid_unit(self.imgshape)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).cuda().float()

        self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().cuda()
        self.com_transform = CompositionTransform_unit().cuda()

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel+3, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, num_block=num_block, bias_opt=bias_opt, num_con=num_con)


        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.up = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        # self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, num_block, bias_opt=False, num_con=1):
        blocks = []
        for i in range(num_block):
            blocks.append(PreActBlock_AdaIn(in_channels, in_channels, bias=bias_opt, num_con=num_con))
            blocks.append(nn.LeakyReLU(0.2))

        layer = nn.ModuleList(blocks)
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, reg_code):
        # compose_field_e0_lvl1, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, lvl1_v, e0
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding = self.model_lvl2(x, y, reg_code)

        # lvl2_disp, lvl2_warp, lvl2_y, lvl2_v, lvl1_v, lvl2_embedding, lvl1_warp, lvl1_y = self.model_lvl2(x, y, reg_code)

        lvl2_disp_up = self.up_tri(lvl2_disp)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 4, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        e0 = e0 + lvl2_embedding

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, reg_code)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        # output_disp_e0 = self.diff_transform(output_disp_e0_v, self.grid_1)
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_disp, e0
            # return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0, lvl1_warp, lvl1_y, lvl2_warp, lvl2_y
        else:
            return compose_field_e0_lvl1


class CentralMappingNetwork(nn.Module):
    def __init__(self, latent_dim=64, mapping_fmaps=64):
        super().__init__()

        self.mapping = nn.Sequential(
            nn.Linear(1, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, latent_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, reg_code):
        return self.mapping(reg_code)


class LocalAdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim=256):
        super().__init__()

        self.gamma = nn.Conv3d(latent_dim, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.beta = nn.Conv3d(latent_dim, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        # self.norm = nn.InstanceNorm3d(in_channel)
        # self.style = nn.Linear(latent_dim, in_channel * 2)
        #
        # self.style.bias.data[:in_channel] = 1
        # self.style.bias.data[in_channel:] = 0

    def forward(self, input, latent_code):
        gamma = self.gamma(latent_code)
        beta = self.beta(latent_code)

        # out = self.norm(input)
        out = input

        out = (1. + gamma) * out + beta

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, latent_dim=256):
        super().__init__()

        # self.norm = nn.InstanceNorm3d(in_channel)

        # self.style = EqualLinear(style_dim, in_channel * 2)

        self.style = nn.Linear(latent_dim, in_channel * 2)

        # self.style.bias.data[:in_channel] = 1
        self.style.bias.data[:in_channel] = 0
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, latent_code):
        # style [batch_size, in_channels*2] => [batch_size, in_channels*2, 1, 1, 1]
        style = self.style(latent_code).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        gamma, beta = style.chunk(2, dim=1)

        # out = self.norm(input)
        out = input

        out = (1. + gamma) * out + beta

        return out


class PreActBlock_AdaIn_central(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False, latent_dim=64):
        super(PreActBlock_AdaIn_central, self).__init__()
        self.ai1 = AdaptiveInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.ai2 = AdaptiveInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x, latent_fea):

        out = F.leaky_relu(self.ai1(x, latent_fea), negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(self.ai2(out, latent_fea), negative_slope=0.2))

        out += shortcut
        return out


class PreActBlock_AdaIn_Local(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False, latent_dim=64, mapping_fmaps=64):
        super(PreActBlock_AdaIn_Local, self).__init__()
        self.ai1 = LocalAdaptiveInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.ai2 = LocalAdaptiveInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        self.mapping = nn.Sequential(
            # nn.Linear(1, mapping_fmaps),
            # nn.LeakyReLU(0.2),
            # nn.Linear(mapping_fmaps, mapping_fmaps),
            # nn.LeakyReLU(0.2),
            # nn.Linear(mapping_fmaps, mapping_fmaps),
            # nn.LeakyReLU(0.2),
            # nn.Linear(mapping_fmaps, latent_dim),
            # nn.LeakyReLU(0.2)
            nn.Conv3d(1, latent_dim, kernel_size=3, stride=stride, padding=1, bias=bias),
            nn.LeakyReLU(0.2)
        )

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x, reg_map):

        # resize conditional map
        reg_map = F.interpolate(reg_map, size=x.size()[2:], mode='trilinear')
        latent_fea = self.mapping(reg_map)

        out = F.leaky_relu(self.ai1(x, latent_fea), negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(self.ai2(out, latent_fea), negative_slope=0.2))

        out += shortcut
        return out


class PreActBlock_AdaIn(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False, latent_dim=64, mapping_fmaps=64, num_con=1):
        super(PreActBlock_AdaIn, self).__init__()
        self.ai1 = AdaptiveInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.ai2 = AdaptiveInstanceNorm(in_planes, latent_dim=latent_dim)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        self.mapping = nn.Sequential(
            nn.Linear(num_con, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, latent_dim),
            nn.LeakyReLU(0.2)
        )

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x, reg_code):

        latent_fea = self.mapping(reg_code)

        out = F.leaky_relu(self.ai1(x, latent_fea), negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(self.ai2(out, latent_fea), negative_slope=0.2))

        out += shortcut
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False):
        super(PreActBlock, self).__init__()
        # self.bn1 = nn.GroupNorm(num_group, in_planes)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        # self.bn2 = nn.GroupNorm(num_group, planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.GroupNorm):
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # out = F.relu(self.bn1(x))
        # out = F.relu(x)
        out = F.leaky_relu(x, negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        # out = self.conv2(F.relu(self.bn2(out)))
        # out = self.conv2(F.relu(out))
        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))


        out += shortcut
        return out


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(False), kernel_size=3, bias=False):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            # nn.ReflectionPad3d(pw),
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=pw, bias=bias),
            activation,
            # nn.ReflectionPad3d(pw),
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=pw, bias=bias)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


class ResNextBlock(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(ResNextBlock, self).__init__()

        bias_opt = True

        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv3d(in_planes, group_width, kernel_size=1, bias=bias_opt)
        # self.bn1 = nn.GroupNorm(cardinality, group_width)
        self.conv2 = nn.Conv3d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=bias_opt)
        # self.bn2 = nn.GroupNorm(cardinality, group_width)
        self.conv3 = nn.Conv3d(group_width, in_planes, kernel_size=1, bias=bias_opt)
        # self.bn3 = nn.GroupNorm(cardinality, in_planes)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*group_width:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv3d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm3d(self.expansion*group_width)
        #     )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        # out = self.bn3(self.conv3(out))

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += x
        out = F.relu(out)
        return out


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class AggregationResBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(False), kernel_size=3, bias=False):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            # nn.ReflectionPad3d(pw),
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=pw, bias=bias),
            activation,
            # nn.ReflectionPad3d(pw),
            nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=pw, bias=bias)
        )

    def forward(self, x, side_input):
        y = self.conv_block(x + side_input)
        out = x + y
        return out


class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (
                    size_tensor[3] - 1) * 2
        sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (
                    size_tensor[2] - 1) * 2
        sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (
                    size_tensor[1] - 1) * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)

        return flow


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        # size_tensor = sample_grid.size()
        # sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
        # sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - (size_tensor[2] / 2)) / size_tensor[2] * 2
        # sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - (size_tensor[1] / 2)) / size_tensor[1] * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)

        return flow


class SpatialTransformNearest(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0,:,:,:,0] = (sample_grid[0,:,:,:,0]-((size_tensor[3]-1)/2))/(size_tensor[3]-1)*2
        sample_grid[0,:,:,:,1] = (sample_grid[0,:,:,:,1]-((size_tensor[2]-1)/2))/(size_tensor[2]-1)*2
        sample_grid[0,:,:,:,2] = (sample_grid[0,:,:,:,2]-((size_tensor[1]-1)/2))/(size_tensor[1]-1)*2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)

        return flow


class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        # size_tensor = sample_grid.size()
        # sample_grid[0, :, :, :, 0] = (sample_grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
        # sample_grid[0, :, :, :, 1] = (sample_grid[0, :, :, :, 1] - (size_tensor[2] / 2)) / size_tensor[2] * 2
        # sample_grid[0, :, :, :, 2] = (sample_grid[0, :, :, :, 2] - (size_tensor[1] / 2)) / size_tensor[1] * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)

        return flow


class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid, range_flow):
        flow = velocity/(2.0**self.time_step)
        size_tensor = sample_grid.size()
        # 0.5 flow
        for _ in range(self.time_step):
            grid = sample_grid + (flow.permute(0,2,3,4,1) * range_flow)
            grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3]-1) / 2)) / (size_tensor[3]-1) * 2
            grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2]-1) / 2)) / (size_tensor[2]-1) * 2
            grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1]-1) / 2)) / (size_tensor[1]-1) * 2
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


class DiffeomorphicTransform_unit(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform_unit, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid):
        flow = velocity/(2.0**self.time_step)
        # size_tensor = sample_grid.size()
        # 0.5 flow
        for _ in range(self.time_step):
            grid = sample_grid + flow.permute(0,2,3,4,1)
            # grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
            # grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - (size_tensor[2] / 2)) / size_tensor[2] * 2
            # grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - (size_tensor[1] / 2)) / size_tensor[1] * 2
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


class CompositionTransform(nn.Module):
    def __init__(self):
        super(CompositionTransform, self).__init__()

    def forward(self, flow_1, flow_2, sample_grid, range_flow):
        size_tensor = sample_grid.size()
        grid = sample_grid + (flow_1.permute(0,2,3,4,1) * range_flow)
        grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
        grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
        grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
        compos_flow = F.grid_sample(flow_2, grid, mode='bilinear', padding_mode="border", align_corners=True) + flow_1
        return compos_flow


class CompositionTransform_unit(nn.Module):
    def __init__(self):
        super(CompositionTransform_unit, self).__init__()

    def forward(self, flow_1, flow_2, sample_grid):
        # size_tensor = sample_grid.size()
        grid = sample_grid + flow_2.permute(0,2,3,4,1)
        # grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
        # grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - (size_tensor[2] / 2)) / size_tensor[2] * 2
        # grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - (size_tensor[1] / 2)) / size_tensor[1] * 2
        compos_flow = F.grid_sample(flow_1, grid, mode='bilinear', padding_mode="border", align_corners=True) + flow_2
        return compos_flow


# def antifoldloss(y_pred):
#     dy = y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :]-1
#     dx = y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :]-1
#     dz = y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:]-1
#
#     dy = F.relu(dy) * torch.abs(dy*dy)
#     dx = F.relu(dx) * torch.abs(dx*dx)
#     dz = F.relu(dz) * torch.abs(dz*dz)
#     return (torch.mean(dx)+torch.mean(dy)+torch.mean(dz))/3.0


def continuous_contrastive_loss(smo_1, smo_2, w_1, w_2):
    return torch.abs((w_1 * smo_1) - (w_2 * smo_2))


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0


def weighted_smoothloss(y_pred, weight):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])

    return (torch.mean(weight[:,:,:, 1:, :]*(dx*dx))+torch.mean(weight[:,:,1:, :, :]*(dy*dy))+torch.mean(weight[:,:,:, :, 1:]*(dz*dz)))/3.0


def multi_resolution_smoothloss(y_pred, num_scale=4):
    total_smooth_loss = 0.0

    scale_y_pred = y_pred
    for i in range(num_scale):
        current_smooth_loss = smoothloss(scale_y_pred)
        total_smooth_loss += current_smooth_loss
        scale_y_pred = nn.functional.interpolate(y_pred, scale_factor=(1.0 / (2 ** (i + 1))))

    return total_smooth_loss


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    # return selected_neg_Jdet
    return torch.mean(selected_neg_Jdet)


def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag))/num_ele

    return diff


def mse_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f-y_pred_f
    mse = torch.mul(diff,diff).mean()   
    return mse


class Normalized_Gradient_Field_weight(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, eps=0.1, channel=1):
        super(Normalized_Gradient_Field_weight, self).__init__()
        self.eps = eps

        # sobel = torch.tensor([
        #     [-1., 0., 1.],
        #     [-2., 0., 2.],
        #     [-1., 0., 1.],
        # ], requires_grad=False).cuda()
        #
        # sobel_z_raw = torch.tensor(
        #     [[[-1., -1., -1.],
        #       [-2., -2., -2.],
        #       [-1., -1., -1.]],
        #      [[0., 0., 0.],
        #       [0., 0., 0.],
        #       [0., 0., 0.]],
        #      [[1., 1., 1.],
        #       [2., 2., 2.],
        #       [1., 1., 1.]]], requires_grad=False).cuda()
        sobel = torch.tensor([
            [-0.5, 0., 0.5],
            [-0.5, 0., 0.5],
            [-0.5, 0., 0.5],
        ], requires_grad=False).cuda()
        sobel_z_raw = torch.tensor(
            [[[-0.5, -0.5, -0.5],
              [-0.5, -0.5, -0.5],
              [-0.5, -0.5, -0.5]],
             [[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]],
             [[0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5]]], requires_grad=False).cuda()

        self.channel = channel
        self.sobel_x = sobel.repeat(channel, 1, 3, 1, 1)
        self.sobel_y = sobel.T.repeat(channel, 1, 3, 1, 1)
        self.sobel_z = sobel_z_raw.repeat(channel, 1, 1, 1, 1)
        self.reflection_pad = nn.ReplicationPad3d(1)

    def forward(self, I, J, mask):
        I_pad = self.reflection_pad(I)
        J_pad = self.reflection_pad(J)
        Ix = F.conv3d(I_pad, self.sobel_x, padding=0, groups=self.channel)
        Iy = F.conv3d(I_pad, self.sobel_y, padding=0, groups=self.channel)
        Iz = F.conv3d(I_pad, self.sobel_z, padding=0, groups=self.channel)
        Jx = F.conv3d(J_pad, self.sobel_x, padding=0, groups=self.channel)
        Jy = F.conv3d(J_pad, self.sobel_y, padding=0, groups=self.channel)
        Jz = F.conv3d(J_pad, self.sobel_z, padding=0, groups=self.channel)

        # Ix = torch.abs(F.conv3d(I_pad, self.sobel_x, padding=0, groups=self.channel))
        # Iy = torch.abs(F.conv3d(I_pad, self.sobel_y, padding=0, groups=self.channel))
        # Iz = torch.abs(F.conv3d(I_pad, self.sobel_z, padding=0, groups=self.channel))
        # Jx = torch.abs(F.conv3d(J_pad, self.sobel_x, padding=0, groups=self.channel))
        # Jy = torch.abs(F.conv3d(J_pad, self.sobel_y, padding=0, groups=self.channel))
        # Jz = torch.abs(F.conv3d(J_pad, self.sobel_z, padding=0, groups=self.channel))

        I_mag = Ix ** 2 + Iy ** 2 + Iz ** 2 + self.eps ** 2
        J_mag = Jx ** 2 + Jy ** 2 + Jz ** 2 + self.eps ** 2
        I_d = torch.cat((Ix, Iy, Iz), dim=1)
        J_d = torch.cat((Jx, Jy, Jz), dim=1)

        ngf = (torch.sum(I_d * J_d, dim=1)) ** 2 / (I_mag * J_mag)
        # ngf_cpu = ngf.cpu().numpy()[0, 0]

        # ngf_2 = (torch.norm(I_d * J_d, dim=1)) ** 2 / (I_mag * J_mag)
        # ngf_2_cpu = ngf_2.cpu().numpy()[0, 0]

        # fig = plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(ngf_cpu[:, :, 81], cmap='gray')
        # plt.title("ngf_sum")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(ngf_2_cpu[:, :, 81], cmap='gray')
        # plt.title("ngf_norm")
        #
        # plt.show()


        # ngf = (torch.sum(I_d*J_d, dim=1))**2/(I_mag * J_mag)
        # temp = (torch.sum((I_d)**2, dim=1) + 1e-10)/(Ix**2 + Iy**2 + Iz**2 + 1e-10) #---->1
        # print(temp.mean(), temp.min(), temp.max())
        # mask = (J > 0).float()
        inner = (1. - ngf)

        # inner_cpu = inner.cpu().numpy()[0, 0]
        # fig = plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(ngf_cpu[:, :, 81], cmap='gray')
        # plt.title("ngf_sum")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(inner_cpu[:, :, 81], cmap='gray')
        # plt.title("inner")
        #
        # plt.show()

        cc = torch.mean(inner*mask)

        return cc


class Normalized_Gradient_Field(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, eps=0.1, channel=1):
        super(Normalized_Gradient_Field, self).__init__()
        self.eps = eps

        # sobel = torch.tensor([
        #     [-1., 0., 1.],
        #     [-2., 0., 2.],
        #     [-1., 0., 1.],
        # ], requires_grad=False).cuda()
        #
        # sobel_z_raw = torch.tensor(
        #     [[[-1., -1., -1.],
        #       [-2., -2., -2.],
        #       [-1., -1., -1.]],
        #      [[0., 0., 0.],
        #       [0., 0., 0.],
        #       [0., 0., 0.]],
        #      [[1., 1., 1.],
        #       [2., 2., 2.],
        #       [1., 1., 1.]]], requires_grad=False).cuda()
        sobel = torch.tensor([
            [-0.5, 0., 0.5],
            [-0.5, 0., 0.5],
            [-0.5, 0., 0.5],
        ], requires_grad=False).cuda()
        sobel_z_raw = torch.tensor(
            [[[-0.5, -0.5, -0.5],
              [-0.5, -0.5, -0.5],
              [-0.5, -0.5, -0.5]],
             [[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]],
             [[0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5]]], requires_grad=False).cuda()

        self.channel = channel
        self.sobel_x = sobel.repeat(channel, 1, 3, 1, 1)
        self.sobel_y = sobel.T.repeat(channel, 1, 3, 1, 1)
        self.sobel_z = sobel_z_raw.repeat(channel, 1, 1, 1, 1)
        self.reflection_pad = nn.ReplicationPad3d(1)

    def forward(self, I, J):
        I_pad = self.reflection_pad(I)
        J_pad = self.reflection_pad(J)
        Ix = F.conv3d(I_pad, self.sobel_x, padding=0, groups=self.channel)
        Iy = F.conv3d(I_pad, self.sobel_y, padding=0, groups=self.channel)
        Iz = F.conv3d(I_pad, self.sobel_z, padding=0, groups=self.channel)
        Jx = F.conv3d(J_pad, self.sobel_x, padding=0, groups=self.channel)
        Jy = F.conv3d(J_pad, self.sobel_y, padding=0, groups=self.channel)
        Jz = F.conv3d(J_pad, self.sobel_z, padding=0, groups=self.channel)

        # Ix = torch.abs(F.conv3d(I_pad, self.sobel_x, padding=0, groups=self.channel))
        # Iy = torch.abs(F.conv3d(I_pad, self.sobel_y, padding=0, groups=self.channel))
        # Iz = torch.abs(F.conv3d(I_pad, self.sobel_z, padding=0, groups=self.channel))
        # Jx = torch.abs(F.conv3d(J_pad, self.sobel_x, padding=0, groups=self.channel))
        # Jy = torch.abs(F.conv3d(J_pad, self.sobel_y, padding=0, groups=self.channel))
        # Jz = torch.abs(F.conv3d(J_pad, self.sobel_z, padding=0, groups=self.channel))

        I_mag = Ix ** 2 + Iy ** 2 + Iz ** 2 + self.eps ** 2
        J_mag = Jx ** 2 + Jy ** 2 + Jz ** 2 + self.eps ** 2
        I_d = torch.cat((Ix, Iy, Iz), dim=1)
        J_d = torch.cat((Jx, Jy, Jz), dim=1)

        ngf = (torch.sum(I_d * J_d, dim=1)) ** 2 / (I_mag * J_mag)
        # ngf_cpu = ngf.cpu().numpy()[0, 0]

        # ngf_2 = (torch.norm(I_d * J_d, dim=1)) ** 2 / (I_mag * J_mag)
        # ngf_2_cpu = ngf_2.cpu().numpy()[0, 0]

        # fig = plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(ngf_cpu[:, :, 81], cmap='gray')
        # plt.title("ngf_sum")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(ngf_2_cpu[:, :, 81], cmap='gray')
        # plt.title("ngf_norm")
        #
        # plt.show()


        # ngf = (torch.sum(I_d*J_d, dim=1))**2/(I_mag * J_mag)
        # temp = (torch.sum((I_d)**2, dim=1) + 1e-10)/(Ix**2 + Iy**2 + Iz**2 + 1e-10) #---->1
        # print(temp.mean(), temp.min(), temp.max())
        # mask = (J > 0).float()
        inner = (1. - ngf)

        # inner_cpu = inner.cpu().numpy()[0, 0]
        # fig = plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(ngf_cpu[:, :, 81], cmap='gray')
        # plt.title("ngf_sum")
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(inner_cpu[:, :, 81], cmap='gray')
        # plt.title("inner")
        #
        # plt.show()

        cc = torch.mean(inner)

        return cc


class multi_resolution_NGF(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, scale=3, eps=0.1):
        super(multi_resolution_NGF, self).__init__()
        self.num_scale = scale
        self.eps = eps
        self.similarity_metric = []
        for i in range(scale):
            self.similarity_metric.append(Normalized_Gradient_Field(eps=self.eps))

    def forward(self, I, J):
        total_NGF = []
        for i in range(self.num_scale):
            current_NGF = self.similarity_metric[0](I, J)
            total_NGF.append(current_NGF/(2**i))
            # print(scale_I.size(), scale_J.size())
            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)
        return sum(total_NGF)


class mindssc_loss(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, delta=1, sigma=0.8):
        super(mindssc_loss, self).__init__()
        self.delta = delta
        self.sigma = sigma

    # learn2reg, delta=3, sigma=3
    def mindssc(self, img, delta=1, sigma=0.8):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        device = img.device

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]], dtype=torch.float, device=device)

        # squared distances
        dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6, device=device), torch.arange(6, device=device))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :].long()
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :].long()
        mshift1 = torch.zeros((12, 1, 3, 3, 3), device=device)
        mshift1.view(-1)[
            torch.arange(12, device=device) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:,
                                                                                                 2]] = 1
        mshift2 = torch.zeros((12, 1, 3, 3, 3), device=device)
        mshift2.view(-1)[
            torch.arange(12, device=device) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:,
                                                                                                 2]] = 1
        rpad = nn.ReplicationPad3d(delta)

        # compute patch-ssd
        ssd = smooth(
            ((F.conv3d(rpad(img), mshift1, dilation=delta) - F.conv3d(rpad(img), mshift2, dilation=delta)) ** 2),
            sigma)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], dtype=torch.long), :, :, :]

        return mind

    def forward(self, x, y):
        return torch.mean((self.mindssc(x, delta=self.delta, sigma=self.sigma) - self.mindssc(y, delta=self.delta, sigma=self.sigma))**2)


class MSE(torch.nn.Module):
    """
    Sigma-weighted mean squared error for image reconstruction.
    """
    def __init__(self, image_sigma=1.0):
        super(MSE, self).__init__()
        self.image_sigma = image_sigma

    def forward(self, y_true, y_pred, weight_map=None):
        if weight_map is None:
            return 1.0 / (self.image_sigma**2) * torch.mean(torch.square(y_true - y_pred))
        else:
            return 1.0 / (self.image_sigma**2) * torch.mean(torch.square(y_true - y_pred) * weight_map)


class multi_resolution_MSE_weight(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, scale=3, image_sigma=1.0):
        super(multi_resolution_MSE_weight, self).__init__()
        self.num_scale = scale
        self.similarity_metric = []
        for i in range(scale):
            self.similarity_metric.append(MSE(image_sigma=image_sigma))

    def forward(self, I, J, weight_map):
        total_MSE = []
        for i in range(self.num_scale):
            current_MSE = self.similarity_metric[i](I, J, weight_map)
            total_MSE.append(current_MSE/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            weight_map = nn.functional.avg_pool3d(weight_map, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_MSE)


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5, channel=1):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win
        self.channel = channel

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((self.channel, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2), groups=self.channel)
        J_sum = conv_fn(J, weight, padding=int(win_size/2), groups=self.channel)
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2), groups=self.channel)
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2), groups=self.channel)
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2), groups=self.channel)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)
        # print(I_var.min(), I_var.max())
        # print(cc.min(), cc.max())

        # return negative cc.
        return -1.0 * torch.mean(cc)


class NCC_weight(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-8, channel=1):
        super(NCC_weight, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win
        self.channel = channel

    def forward(self, I, J, weight_map):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((self.channel, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2), groups=self.channel)
        J_sum = conv_fn(J, weight, padding=int(win_size/2), groups=self.channel)
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2), groups=self.channel)
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2), groups=self.channel)
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2), groups=self.channel)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc*weight_map)


class NCC_weight_allmod(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-8, channel=3):
        super(NCC_weight_allmod, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win
        self.channel = channel

    def forward(self, I, J, weight_map):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((self.channel, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2), groups=self.channel)
        J_sum = conv_fn(J, weight, padding=int(win_size/2), groups=self.channel)
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2), groups=self.channel)
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2), groups=self.channel)
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2), groups=self.channel)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc*weight_map)


class NCC_allmod(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-8):
        super(NCC_allmod, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((3, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2), groups=3)
        J_sum = conv_fn(J, weight, padding=int(win_size/2), groups=3)
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2), groups=3)
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2), groups=3)
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2), groups=3)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class NCC_map(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=None, eps=1e-5):
        super(NCC_map, self).__init__()
        self.win = win
        self.eps = eps

    def forward(self, I, J):
        ndims = 3

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        weight_win_size = 9
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(weight_win_size/2))
        J_sum = conv_fn(J, weight, padding=int(weight_win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(weight_win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(weight_win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(weight_win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return cc


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        # self.similarity_metric = NCC(win=win)

        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J):
        total_NCC = []
        # scale_I = I
        # scale_J = J
        #
        # for i in range(self.num_scale):
        #     current_NCC = similarity_metric(scale_I,scale_J)
        #     # print("Scale ", i, ": ", current_NCC, (2**i))
        #     total_NCC += current_NCC/(2**i)
        #     # print(scale_I.size(), scale_J.size())
        #     # print(current_NCC)
        #     scale_I = nn.functional.interpolate(I, scale_factor=(1.0/(2**(i+1))))
        #     scale_J = nn.functional.interpolate(J, scale_factor=(1.0/(2**(i+1))))

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)


class multi_resolution_NCC_weight(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3, channel=1):
        super(multi_resolution_NCC_weight, self).__init__()
        self.num_scale = scale
        # self.similarity_metric = NCC(win=win)

        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC_weight(win=win - (i*2), channel=channel))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J, weight_map):
        total_NCC = []
        # scale_I = I
        # scale_J = J
        #
        # for i in range(self.num_scale):
        #     current_NCC = similarity_metric(scale_I,scale_J)
        #     # print("Scale ", i, ": ", current_NCC, (2**i))
        #     total_NCC += current_NCC/(2**i)
        #     # print(scale_I.size(), scale_J.size())
        #     # print(current_NCC)
        #     scale_I = nn.functional.interpolate(I, scale_factor=(1.0/(2**(i+1))))
        #     scale_J = nn.functional.interpolate(J, scale_factor=(1.0/(2**(i+1))))

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J, weight_map)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            weight_map = nn.functional.avg_pool3d(weight_map, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)


class multi_resolution_NCC_weight_allmod(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3, channel=3):
        super(multi_resolution_NCC_weight_allmod, self).__init__()
        self.num_scale = scale
        # self.similarity_metric = NCC(win=win)

        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC_weight_allmod(win=win - (i*2), channel=channel))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J, weight_map):
        total_NCC = []
        # scale_I = I
        # scale_J = J
        #
        # for i in range(self.num_scale):
        #     current_NCC = similarity_metric(scale_I,scale_J)
        #     # print("Scale ", i, ": ", current_NCC, (2**i))
        #     total_NCC += current_NCC/(2**i)
        #     # print(scale_I.size(), scale_J.size())
        #     # print(current_NCC)
        #     scale_I = nn.functional.interpolate(I, scale_factor=(1.0/(2**(i+1))))
        #     scale_J = nn.functional.interpolate(J, scale_factor=(1.0/(2**(i+1))))

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J, weight_map)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            weight_map = nn.functional.avg_pool3d(weight_map, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)


class multi_resolution_NCC_allmod(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC_allmod, self).__init__()
        self.num_scale = scale
        # self.similarity_metric = NCC(win=win)

        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC_allmod(win=win - (i*2)))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J):
        total_NCC = []

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)


class multi_resolution_Gradient_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_Gradient_NCC, self).__init__()
        self.num_scale = scale
        # self.similarity_metric = NCC(win=win)

        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(Normalized_Gradient_Field(eps=0.1))

    def forward(self, I, J):
        total_NCC = []
        # scale_I = I
        # scale_J = J
        #
        # for i in range(self.num_scale):
        #     current_NCC = similarity_metric(scale_I,scale_J)
        #     # print("Scale ", i, ": ", current_NCC, (2**i))
        #     total_NCC += current_NCC/(2**i)
        #     # print(scale_I.size(), scale_J.size())
        #     # print(current_NCC)
        #     scale_I = nn.functional.interpolate(I, scale_factor=(1.0/(2**(i+1))))
        #     scale_J = nn.functional.interpolate(J, scale_factor=(1.0/(2**(i+1))))

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[0](I, J)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)


class DemonsOrientation(nn.Module):
    def __init__(self, channel, alpha=1.0):
        super(DemonsOrientation, self).__init__()
        self.alpha = alpha
        sobel = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.],
        ], requires_grad=False).cuda()

        sobel_z_raw = torch.tensor(
            [[[-1., -1., -1.],
              [-2., -2., -2.],
              [-1., -1., -1.]],
             [[0., 0., 0.],
              [0., 0., 0.],
              [0., 0., 0.]],
             [[1., 1., 1.],
              [2., 2., 2.],
              [1., 1., 1.]]], requires_grad=False).cuda()

        self.channel = channel
        self.sobel_x = sobel.repeat(channel, 1, 3, 1, 1)
        self.sobel_y = sobel.T.repeat(channel, 1, 3, 1, 1)
        self.sobel_z = sobel_z_raw.repeat(channel, 1, 1, 1, 1)
        # self.conv0 = nn.Conv3d(3*channel, out_channel, kernel_size=3, padding=1, bias=False, groups=3)
        self.eps = 1e-10

    def forward(self, M, S, flow):
        Idiff = M - S

        Sx = F.conv3d(S, self.sobel_x, padding=1, groups=self.channel)
        Sy = F.conv3d(S, self.sobel_y, padding=1, groups=self.channel)
        Sz = F.conv3d(S, self.sobel_z, padding=1, groups=self.channel)

        Mx = F.conv3d(M, self.sobel_x, padding=1, groups=self.channel)
        My = F.conv3d(M, self.sobel_y, padding=1, groups=self.channel)
        Mz = F.conv3d(M, self.sobel_z, padding=1, groups=self.channel)

        Sxyz_mag = Sx ** 2 + Sy ** 2 + Sz ** 2
        Mxyz_mag = Mx ** 2 + My ** 2 + Mz ** 2

        # Demon force.
        Ux = Idiff * ((Sx / (Sxyz_mag + (self.alpha ** 2) * (Idiff ** 2) + 1e-10)) + (
                    Mx / (Mxyz_mag + (self.alpha ** 2) * (Idiff ** 2) + 1e-10)))
        Uy = Idiff * ((Sy / (Sxyz_mag + (self.alpha ** 2) * (Idiff ** 2) + 1e-10)) + (
                    My / (Mxyz_mag + (self.alpha ** 2) * (Idiff ** 2) + 1e-10)))
        Uz = Idiff * ((Sz / (Sxyz_mag + (self.alpha ** 2) * (Idiff ** 2) + 1e-10)) + (
                    Mz / (Mxyz_mag + (self.alpha ** 2) * (Idiff ** 2) + 1e-10)))

        # demons_force = torch.cat([Ux, Uy, Uz], dim=1)
        # demons_fea = self.conv0(demons_force)
        demons_ori_xz = torch.atan(Ux/(Uz+self.eps))
        demons_ori_yz = torch.atan(Uy/(Uz+self.eps))

        flow_ori_xz = torch.atan(flow[:,0, :, :, :]/(flow[:,2, :, :, :]+self.eps))
        flow_ori_yz = torch.atan(flow[:,1, :, :, :]/(flow[:,2, :, :, :]+self.eps))

        ori_error = torch.mean((flow_ori_xz - demons_ori_xz)**2) + torch.mean((flow_ori_yz - demons_ori_yz)**2)

        return ori_error


class Center_of_mass_initial_pairwise(nn.Module):
    def __init__(self):
        super(Center_of_mass_initial_pairwise, self).__init__()
        self.id = torch.zeros((1, 3, 4)).cuda()
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1

        self.to_center_matrix = torch.zeros((1, 3, 4)).cuda()
        self.to_center_matrix[0, 0, 0] = 1
        self.to_center_matrix[0, 1, 1] = 1
        self.to_center_matrix[0, 2, 2] = 1

    def forward(self, x, y):
        # center of mass of x -> center of mass of y
        id_grid = F.affine_grid(self.id, x.shape, align_corners=True)
        # mask = (x > 0).float()
        # mask_sum = torch.sum(mask)
        x_sum = torch.sum(x)
        x_center_mass_x = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0])/x_sum
        x_center_mass_y = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1])/x_sum
        x_center_mass_z = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2])/x_sum

        y_sum = torch.sum(y)
        y_center_mass_x = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0]) / y_sum
        y_center_mass_y = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1]) / y_sum
        y_center_mass_z = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2]) / y_sum

        self.to_center_matrix[0, 0, 3] = x_center_mass_x - y_center_mass_x
        self.to_center_matrix[0, 1, 3] = x_center_mass_y - y_center_mass_y
        self.to_center_matrix[0, 2, 3] = x_center_mass_z - y_center_mass_z

        grid = F.affine_grid(self.to_center_matrix, x.shape, align_corners=True)
        transformed_image = F.grid_sample(x, grid, align_corners=True)

        # print(affine_para)
        # print(output_affine_m[0:3])

        return transformed_image, grid


class curvature_smoothness(torch.nn.Module):
    """
    2nd order curvature regularization
    """
    def __init__(self):
        super(curvature_smoothness, self).__init__()

        laplacian = torch.tensor(
            [[[0., 0., 0.],
              [0., 1., 0.],
              [0., 0., 0.]],
             [[0., 1., 0.],
              [1., -6., 1.],
              [0., 1., 0.]],
             [[0., 0., 0.],
              [0., 1., 0.],
              [0., 0., 0.]]], requires_grad=False).cuda()

        self.channel = 3
        self.laplacian = laplacian.repeat(self.channel, 1, 1, 1, 1)
        self.reflection_pad = nn.ReplicationPad3d(1)

    def forward(self, y_pred):
        y_pred_pad = self.reflection_pad(y_pred)
        lap = F.conv3d(y_pred_pad, self.laplacian, padding=0, groups=self.channel)

        return torch.mean(torch.pow(lap, 2))


if __name__ == '__main__':
    print()
