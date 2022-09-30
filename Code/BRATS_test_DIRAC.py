import glob
import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from scipy.ndimage.interpolation import map_coordinates

from Functions import Validation_Brats, generate_grid_unit, save_img
from bratsreg_model_stage import Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1, \
    Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2, Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3, \
    SpatialTransform_unit

parser = ArgumentParser()
parser.add_argument("--modelname", type=str,
                    dest="modelname",
                    default='../Model/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv5_a0015_aug_mean_fffixed_github_stagelvl3_64000.pth',
                    help="Model name")
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=6,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../Dataset/test',
                    help="data path for training images")
parser.add_argument("--num_cblock", type=int,
                    dest="num_cblock", default=5,
                    help="Number of conditional block")
parser.add_argument("--output_seg", type=bool,
                    dest="output_seg", default=False,
                    help="True: save segmentation map")


def compute_tre(x, y, spacing=(1, 1, 1)):
    return np.linalg.norm((x - y) * spacing, axis=1)


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if i == 0:
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice/num_count


def test():
    print("Training lvl3...")
    model_lvl1 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4,
                                               range_flow=range_flow, num_block=num_cblock).cuda()
    model_lvl2 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2,
                                          range_flow=range_flow, model_lvl1=model_lvl1, num_block=num_cblock).cuda()

    model = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                          range_flow=range_flow, model_lvl2=model_lvl2, num_block=num_cblock).cuda()

    model_path = model_name
    model.load_state_dict(torch.load(model_path))

    transform = SpatialTransform_unit().cuda()
    # transform_nearest = SpatialTransformNearest_unit().cuda()
    # diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
    # com_transform = CompositionTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # Validation
    start, end = 0, 20
    # val_fixed_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))[-20:]
    # val_moving_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_t1ce.nii.gz"))[-40:]
    # val_moving_list = sorted([path for path in val_moving_list if path not in val_fixed_list])
    #
    # val_fixed_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_landmarks.csv"))[-20:]
    # val_moving_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarks.csv"))[-40:]
    # val_moving_csv_list = sorted([path for path in val_moving_csv_list if path not in val_fixed_csv_list])
    # tumor_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_seg.nii.gz"))[-20:]

    val_fixed_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
    val_moving_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_t1ce.nii.gz"))
    val_moving_list = sorted([path for path in val_moving_list if path not in val_fixed_list])

    val_fixed_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_landmarks.csv"))
    val_moving_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarks.csv"))
    val_moving_csv_list = sorted([path for path in val_moving_csv_list if path not in val_fixed_csv_list])

    valid_generator = Data.DataLoader(Validation_Brats(val_fixed_list, val_moving_list, val_fixed_csv_list,
                                                      val_moving_csv_list, norm=True), batch_size=1,
                                      shuffle=False, num_workers=2)

    save_path = '../Result'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    template = nib.load(val_fixed_list[0])
    header, affine = template.header, template.affine

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    # dice_total = []
    tre_total = []
    print("\nValiding...")
    for batch_idx, data in enumerate(valid_generator):
        # X_ori, Y_ori, X_label, Y_label, tumor_mask = data['move'].to(device), data['fixed'].to(device), \
        #                          data['move_label'].numpy()[0], data['fixed_label'].numpy()[0], data['tumor_mask'].to(device)
        Y_ori, X_ori, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), \
                                                     data['move_label'].numpy()[0], data['fixed_label'].numpy()[0]

        ori_img_shape = X_ori.shape[2:]
        h, w, d = ori_img_shape

        X = F.interpolate(X_ori, size=imgshape, mode='trilinear')
        Y = F.interpolate(Y_ori, size=imgshape, mode='trilinear')

        with torch.no_grad():
            reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(dim=0)
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

            F_Y_X, Y_X, X_4x, F_yx, F_yx_lvl1, F_yx_lvl2, _ = model(Y, X, reg_code)

            F_X_Y = F.interpolate(F_X_Y, size=ori_img_shape, mode='trilinear', align_corners=True)
            F_Y_X = F.interpolate(F_Y_X, size=ori_img_shape, mode='trilinear', align_corners=True)

            grid_unit = generate_grid_unit(ori_img_shape)
            grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

            if output_seg:
                F_X_Y_warpped = transform(F_X_Y, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit)
                F_Y_X_warpped = transform(F_Y_X, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)

                diff_fw = F_X_Y + F_Y_X_warpped  # Y
                diff_bw = F_Y_X + F_X_Y_warpped  # X

                fw_mask = (Y_ori > 0).float()
                bw_mask = (X_ori > 0).float()

                u_diff_fw = torch.sum(torch.norm(diff_fw * fw_mask, dim=1, keepdim=True)) / torch.sum(fw_mask)
                u_diff_bw = torch.sum(torch.norm(diff_bw * bw_mask, dim=1, keepdim=True)) / torch.sum(bw_mask)

                thresh_fw = (u_diff_fw + 0.015) * torch.ones_like(Y_ori, device=Y_ori.device)
                thresh_bw = (u_diff_bw + 0.015) * torch.ones_like(X_ori, device=X_ori.device)

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

                save_img(occ_xy.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_xy_seg.nii.gz", header=header,
                         affine=affine)
                save_img(occ_yx.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_yx_seg.nii.gz", header=header,
                         affine=affine)

                save_img(norm_diff_fw.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_diff_fw.nii.gz", header=header,
                         affine=affine)
                save_img(norm_diff_bw.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_diff_bw.nii.gz", header=header,
                         affine=affine)

            X_Y = transform(X_ori, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)
            Y_X = transform(Y_ori, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit)

            save_img(X_Y.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_X_Y.nii.gz", header=header, affine=affine)
            save_img(Y_X.cpu().numpy()[0, 0], f"{save_path}/{batch_idx + 1}_Y_X.nii.gz", header=header, affine=affine)

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

            print(batch_idx, ": TRE: ", tre_score)

    tre_total = np.array(tre_total)
    print("TRE mean: ", tre_total.mean())
    # with open(log_dir, "a") as log:
    #     log.write(str(step)+":"+str(tre_total.mean()) + "\n")

if __name__ == '__main__':
    opt = parser.parse_args()

    lr = opt.lr
    start_channel = opt.start_channel
    datapath = opt.datapath
    num_cblock = opt.num_cblock
    model_name = opt.modelname
    output_seg = opt.output_seg

    img_h, img_w, img_d = 160, 160, 80
    imgshape = (img_h, img_w, img_d)
    imgshape_4 = (img_h//4, img_w//4, img_d//4)
    imgshape_2 = (img_h//2, img_w//2, img_d//2)

    range_flow = 0.4
    print("Testing %s ..." % model_name)
    test()
