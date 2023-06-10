import glob
import os
import random
from argparse import ArgumentParser

import math
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from scipy.ndimage.interpolation import map_coordinates

from Functions import Dataset_bratsreg_bidirection_t1cet2, Validation_Brats_t1cet2, \
    generate_grid_unit
from bratsreg_model_stage import Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl1, \
    Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl2, Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl3, \
    SpatialTransform_unit, smoothloss, \
    multi_resolution_NCC_weight_allmod

parser = ArgumentParser()
parser.add_argument("--modelname", type=str,
                    dest="modelname",
                    default='Brats_NCC_disp_fea6b5_AdaIn64_t1cet2_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_',
                    help="Model name")
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=130001,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.,
                    help="Anti-fold loss: suggested range 1 to 1000000")
parser.add_argument("--occ", type=float,
                    dest="occ", default=0.01,
                    help="Mask loss: suggested range 0.01 to 1")
parser.add_argument("--inv_con", type=float,
                    dest="inv_con", default=0.1,
                    help="Inverse consistency loss: suggested range 1 to 10")
# parser.add_argument("--grad_sim", type=float,
#                     dest="grad_sim", default=0.1,
#                     help="grad_sim loss: suggested range ... to ...")
# parser.add_argument("--smooth", type=float,
#                     dest="smooth", default=12.0,
#                     help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=2000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,  # default:8, 7 for stage
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../Dataset/BraTSReg_self_train',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=3000,
                    help="Number step for freezing the previous level")
parser.add_argument("--num_cblock", type=int,
                    dest="num_cblock", default=5,
                    help="Number of conditional block")


def affine_aug(im, im_label=None):
    # mode = 'bilinear' or 'nearest'
    with torch.no_grad():
        angle_range = 10
        trans_range = 0.05
        scale_range = 0.0
        # scale_range = 0.15

        angle_xyz = (random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180)
        scale_xyz = (random.uniform(-scale_range, scale_range), random.uniform(-scale_range, scale_range),
                     random.uniform(-scale_range, scale_range))
        trans_xyz = (random.uniform(-trans_range, trans_range), random.uniform(-trans_range, trans_range),
                     random.uniform(-trans_range, trans_range))

        rotation_x = torch.tensor([
            [1., 0, 0, 0],
            [0, math.cos(angle_xyz[0]), -math.sin(angle_xyz[0]), 0],
            [0, math.sin(angle_xyz[0]), math.cos(angle_xyz[0]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_y = torch.tensor([
            [math.cos(angle_xyz[1]), 0, math.sin(angle_xyz[1]), 0],
            [0, 1., 0, 0],
            [-math.sin(angle_xyz[1]), 0, math.cos(angle_xyz[1]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_z = torch.tensor([
            [math.cos(angle_xyz[2]), -math.sin(angle_xyz[2]), 0, 0],
            [math.sin(angle_xyz[2]), math.cos(angle_xyz[2]), 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        trans_shear_xyz = torch.tensor([
            [1. + scale_xyz[0], 0, 0, trans_xyz[0]],
            [0, 1. + scale_xyz[1], 0, trans_xyz[1]],
            [0, 0, 1. + scale_xyz[2], trans_xyz[2]],
            [0, 0, 0, 1]
        ], requires_grad=False).unsqueeze(0).cuda()

        theta_final = torch.matmul(rotation_x, rotation_y)
        theta_final = torch.matmul(theta_final, rotation_z)
        theta_final = torch.matmul(theta_final, trans_shear_xyz)

        output_disp_e0_v = F.affine_grid(theta_final[:, 0:3, :], im.shape, align_corners=True)

        im = F.grid_sample(im, output_disp_e0_v, mode='bilinear', padding_mode="border", align_corners=True)

        if im_label is not None:
            im_label = F.grid_sample(im_label, output_disp_e0_v, mode='bilinear', padding_mode="border",
                                     align_corners=True)
            return im, im_label
        else:
            return im


def compute_tre(x, y, spacing=(1, 1, 1)):
    return np.linalg.norm((x - y) * spacing, axis=1)


def train():
    print("Training lvl3...")
    model_lvl1 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl1(4, 3, start_channel, is_train=True,
                                                                          imgshape=imgshape_4,
                                                                          range_flow=range_flow,
                                                                          num_block=num_cblock).cuda()
    model_lvl2 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl2(4, 3, start_channel, is_train=True,
                                                                          imgshape=imgshape_2,
                                                                          range_flow=range_flow, model_lvl1=model_lvl1,
                                                                          num_block=num_cblock).cuda()

    model = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_t1cet2_lvl3(4, 3, start_channel, is_train=True,
                                                                     imgshape=imgshape,
                                                                     range_flow=range_flow, model_lvl2=model_lvl2,
                                                                     num_block=num_cblock).cuda()

    # loss_similarity = mse_loss
    # loss_similarity = NCC()
    # loss_similarity = Edge_enhanced_CC()
    # loss_similarity = CC()
    # loss_similarity = Normalized_Gradient_Field(eps=0.1)
    loss_similarity = multi_resolution_NCC_weight_allmod(win=7, scale=3, channel=2)

    # loss_similarity_grad = Gradient_CC()
    # loss_similarity = NCC()

    # loss_inverse = mse_loss
    # loss_antifold = antifoldloss
    loss_smooth = smoothloss
    # loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().cuda()
    # transform_nearest = SpatialTransformNearest_unit().cuda()
    # diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
    # com_transform = CompositionTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    # names = sorted(glob.glob(datapath + '/*.nii'))[0:255]
    fixed_t1ce_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
    moving_t1ce_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_t1ce.nii.gz"))
    moving_t1ce_list = sorted([path for path in moving_t1ce_list if path not in fixed_t1ce_list])

    # fixed_flair_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_flair.nii.gz"))
    # moving_flair_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_flair.nii.gz"))
    # moving_flair_list = sorted([path for path in moving_flair_list if path not in fixed_flair_list])

    fixed_t2_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_t2.nii.gz"))
    moving_t2_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_t2.nii.gz"))
    moving_t2_list = sorted([path for path in moving_t2_list if path not in fixed_t2_list])

    # # LPBA
    # names = sorted(glob.glob(datapath + '/S*_norm.nii'))[0:30]

    # grid = generate_grid(imgshape)
    # grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    grid_unit = generate_grid_unit(imgshape)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model/' + model_name[0:-1]

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    lossall = np.zeros((4, iteration_lvl3 + 1))

    training_generator = Data.DataLoader(
        Dataset_bratsreg_bidirection_t1cet2(fixed_t1ce_list, moving_t1ce_list, fixed_t2_list, moving_t2_list,
                                            norm=False), batch_size=1,
        shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl3:
        for X, Y in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()

            aug_flag = random.uniform(0, 1)
            if aug_flag > 0.2:
                X = affine_aug(X)

            aug_flag = random.uniform(0, 1)
            if aug_flag > 0.2:
                Y = affine_aug(Y)

            X = F.interpolate(X, size=imgshape, mode='trilinear')
            Y = F.interpolate(Y, size=imgshape, mode='trilinear')

            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            # lap
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

            F_Y_X, Y_X, X_4x, F_yx, F_yx_lvl1, F_yx_lvl2, _ = model(Y, X, reg_code)

            F_X_Y_warpped = transform(F_X_Y, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit)
            F_Y_X_warpped = transform(F_Y_X, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)

            diff_fw = F_X_Y + F_Y_X_warpped  # Y
            diff_bw = F_Y_X + F_X_Y_warpped  # X

            fw_mask = (Y_4x[:, 0:1] > 0).float()
            bw_mask = (X_4x[:, 0:1] > 0).float()

            u_diff_fw = torch.sum(torch.norm(diff_fw * fw_mask, dim=1, keepdim=True)) / torch.sum(fw_mask)
            u_diff_bw = torch.sum(torch.norm(diff_bw * bw_mask, dim=1, keepdim=True)) / torch.sum(bw_mask)

            thresh_fw = (u_diff_fw + 0.015) * torch.ones_like(Y_4x[:, 0:1], device=Y_4x.device)
            thresh_bw = (u_diff_bw + 0.015) * torch.ones_like(X_4x[:, 0:1], device=X_4x.device)

            # smoothing
            norm_diff_fw = torch.norm(diff_fw, dim=1, keepdim=True)
            norm_diff_bw = torch.norm(diff_bw, dim=1, keepdim=True)

            smo_norm_diff_fw = F.avg_pool3d(F.avg_pool3d(norm_diff_fw, kernel_size=5, stride=1, padding=2),
                                            kernel_size=5, stride=1, padding=2)
            smo_norm_diff_bw = F.avg_pool3d(F.avg_pool3d(norm_diff_bw, kernel_size=5, stride=1, padding=2),
                                            kernel_size=5, stride=1, padding=2)

            occ_xy = (smo_norm_diff_fw > thresh_fw).float()  # y mask
            occ_yx = (smo_norm_diff_bw > thresh_bw).float()  # x mask

            # mask occ
            occ_xy = occ_xy * fw_mask
            occ_yx = occ_yx * bw_mask

            occ_xy_l = F.relu(smo_norm_diff_fw - thresh_fw) * 10.
            occ_yx_l = F.relu(smo_norm_diff_bw - thresh_bw) * 10.

            mask_xy = 1. - occ_xy
            mask_yx = 1. - occ_yx

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x, mask_xy) + loss_similarity(Y_X, X_4x, mask_yx)

            loss_inverse = torch.mean(norm_diff_fw * mask_xy) + torch.mean(norm_diff_bw * mask_yx)
            loss_occ = torch.mean(occ_xy_l) + torch.mean(occ_yx_l)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
            # loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = z
            norm_vector[0, 1, 0, 0, 0] = y
            norm_vector[0, 2, 0, 0, 0] = x
            loss_regulation = loss_smooth(F_X_Y * norm_vector) + loss_smooth(F_Y_X * norm_vector)

            # b, c, x, y, z = F_xy.shape
            # scale_flow = torch.Tensor([z, y, x]).cuda().reshape(1, 3, 1, 1, 1)
            # loss_regulation = loss_smooth(F_xy*scale_flow)

            loss = (
                               1. - reg_code) * loss_multiNCC + reg_code * loss_regulation + occ * loss_occ + inv_con * loss_inverse
            # loss = loss_multiNCC + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_inverse.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" -sim_NCC "{2:4f}" -inv "{3:.4f}" -occ "{4:4f}" -smo "{5:.4f} -reg_c "{6:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_inverse.item(), loss_occ.item(),
                    loss_regulation.item(), reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

                # Validation
                val_datapath = '../Dataset/BraTSReg_self_valid'
                start, end = 0, 20
                val_fixed_csv_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_0000_landmarks.csv"))
                val_moving_csv_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_landmarks.csv"))
                val_moving_csv_list = sorted([path for path in val_moving_csv_list if path not in val_fixed_csv_list])

                val_fixed_t1ce_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
                val_moving_t1ce_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_t1ce.nii.gz"))
                val_moving_t1ce_list = sorted(
                    [path for path in val_moving_t1ce_list if path not in val_fixed_t1ce_list])

                val_fixed_t2_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_0000_t2.nii.gz"))
                val_moving_t2_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_t2.nii.gz"))
                val_moving_t2_list = sorted([path for path in val_moving_t2_list if path not in val_fixed_t2_list])

                # assert len(val_fixed_list) == len(val_moving_list)

                valid_generator = Data.DataLoader(
                    Validation_Brats_t1cet2(val_fixed_t1ce_list, val_moving_t1ce_list, val_fixed_t2_list,
                                            val_moving_t2_list, val_fixed_csv_list,
                                            val_moving_csv_list, norm=False), batch_size=1,
                    shuffle=False, num_workers=2)

                use_cuda = True
                device = torch.device("cuda" if use_cuda else "cpu")
                # dice_total = []
                tre_total = []
                print("\nValiding...")
                for batch_idx, data in enumerate(valid_generator):
                    # X, Y, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), \
                    #                          data['move_label'].numpy()[0], data['fixed_label'].numpy()[0]
                    Y, X, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), \
                                             data['move_label'].numpy()[0], data['fixed_label'].numpy()[0]
                    ori_img_shape = X.shape[2:]
                    h, w, d = ori_img_shape

                    X = F.interpolate(X, size=imgshape, mode='trilinear')
                    Y = F.interpolate(Y, size=imgshape, mode='trilinear')

                    with torch.no_grad():
                        reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(dim=0)
                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

                        # X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit).data.cpu().numpy()[0, 0, :, :, :]
                        # Y_label = Y_label.data.cpu().numpy()[0, 0, :, :, :]

                        F_X_Y = F.interpolate(F_X_Y, size=ori_img_shape, mode='trilinear', align_corners=True)

                        full_F_X_Y = torch.zeros(F_X_Y.shape)
                        full_F_X_Y[0, 0] = F_X_Y[0, 2] * (h - 1) / 2
                        full_F_X_Y[0, 1] = F_X_Y[0, 1] * (w - 1) / 2
                        full_F_X_Y[0, 2] = F_X_Y[0, 0] * (d - 1) / 2

                        # TRE
                        full_F_X_Y = full_F_X_Y.cpu().numpy()[0]

                        fixed_keypoints = Y_label
                        moving_keypoints = X_label

                        moving_disp_x = map_coordinates(full_F_X_Y[0], moving_keypoints.transpose())
                        moving_disp_y = map_coordinates(full_F_X_Y[1], moving_keypoints.transpose())
                        moving_disp_z = map_coordinates(full_F_X_Y[2], moving_keypoints.transpose())
                        lms_moving_disp = np.array((moving_disp_x, moving_disp_y, moving_disp_z)).transpose()

                        warped_moving_keypoint = moving_keypoints + lms_moving_disp

                        tre_score = compute_tre(warped_moving_keypoint, fixed_keypoints,
                                                spacing=(1., 1., 1.)).mean()
                        tre_total.append(tre_score)

                tre_total = np.array(tre_total)
                print("TRE mean: ", tre_total.mean())
                with open(log_dir, "a") as log:
                    log.write(str(step) + ":" + str(tre_total.mean()) + "\n")

            # if step == freeze_step:
            #     model.unfreeze_modellvl2()
            #     # num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            #     # print("\nmodel_lvl2_num_param_requires_grad: ", num_param)

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)


if __name__ == '__main__':
    opt = parser.parse_args()

    lr = opt.lr
    start_channel = opt.start_channel
    antifold = opt.antifold
    # grad_sim = opt.grad_sim
    n_checkpoint = opt.checkpoint
    # smooth = opt.smooth
    datapath = opt.datapath
    freeze_step = opt.freeze_step
    num_cblock = opt.num_cblock
    occ = opt.occ
    inv_con = opt.inv_con

    iteration_lvl3 = opt.iteration_lvl3

    model_name = opt.modelname

    # Create and initalize log file
    if not os.path.isdir("../Log"):
        os.mkdir("../Log")

    log_dir = "../Log/" + model_name + ".txt"

    with open(log_dir, "a") as log:
        log.write("Validation TRE log for " + model_name[0:-1] + ":\n")

    img_h, img_w, img_d = 160, 160, 80
    imgshape = (img_h, img_w, img_d)
    imgshape_4 = (img_h // 4, img_w // 4, img_d // 4)
    imgshape_2 = (img_h // 2, img_w // 2, img_d // 2)

    range_flow = 0.4
    print("Training %s ..." % model_name)
    train()
