# import SimpleITK as sitk
import math
import random

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import torch.nn.functional as F

import glob
import itertools
import csv
import scipy.ndimage


def create_image_pyramid(image, down_sample_factor, device, channel=1):
    """
    :param image: (1, 1, H, W, D)
    :param down_sample_factor: [4, 2]
    :return: list of images
    """
    image_pyramid = []
    for level in down_sample_factor:
        sigma = 1
        kernel_size = [int(2*np.ceil(sigma*2) + 1)] * 3
        kernel = get_gaussian_kernel(kernel_size, [sigma]*3).to(device)
        kernel = kernel.repeat(channel, 1, 1, 1, 1)
        padding = kernel_size[0]//2

        down_image = F.conv3d(image, kernel, stride=level, padding=padding, groups=channel)
        image_pyramid.append(down_image)

    image_pyramid.append(image)
    return image_pyramid


def get_gaussian_kernel(kernel_size, sigma):
    """
    :param kernel_size: tuple, e.g., [5, 5, 5]
    :param sigma: tuple, e.g., [1, 1, 1]
    :return: Gaussian kernel in pytorch weight, e.g., (1, 1, 5, 5, 5)
    """
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )

    kernel = 1
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)
    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    # kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)) # *[1] * (kernel.dim() - 1) -> (1, 1, 1)

    return kernel


def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
                jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                       :, :]) + \
             jacobian[2, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                       :, :])

    return jacdet


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid

# (grid[0, :, :, :, 0] - (size_tensor[3] / 2)) / size_tensor[3] * 2
def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0]-1)/2)) / (imgshape[0]-1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1]-1)/2)) / (imgshape[1]-1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2]-1)/2)) / (imgshape[2]-1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid,0,2)
    grid = np.swapaxes(grid,1,2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)/2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)/2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)/2

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)/2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)/2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)/2

    return flow

def load_4D(name):
    # X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    # X = np.reshape(X, (1,)+ X.shape)
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_5D(name):
    # X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,)+(1,)+ X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)
    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def Norm_Zscore(img):
    img= (img-np.mean(img))/np.std(img) 
    return img


def save_img(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_img_nii(I_img,savename):
    # I2 = sitk.GetImageFromArray(I_img,isVector=False)
    # sitk.WriteImage(I2,savename)
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    # save_path = os.path.join(output_path, savename)
    nib.save(new_img, savename)


def save_flow(I_img,savename,header=None,affine=None):
    # I2 = sitk.GetImageFromArray(I_img,isVector=True)
    # sitk.WriteImage(I2,savename)
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


class Dataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names,iterations,norm=True):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations
  def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names)) [0:2]
        img_A = load_4D(self.names[index_pair[0]])
        img_B = load_4D(self.names[index_pair[1]])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=True):
        'Initialization'
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        moving_img = load_4D(self.index_pair[step][0])
        fixed_img = load_4D(self.index_pair[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float() ,torch.from_numpy(imgnorm(fixed_img)).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float()


class Dataset_epoch_NLST(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_list, mask_list, keypoint_csv, need_label=True):
        'Initialization'
        super(Dataset_epoch_NLST, self).__init__()
        # self.exp_path = exp_path
        self.img_pair = img_list
        self.mask_pair = mask_list
        self.keypoint_csv = keypoint_csv
        self.need_label = need_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img = load_4D(self.img_pair[step][0])
        moving_img = load_4D(self.img_pair[step][1])

        # Windowing
        fixed_img = np.clip(fixed_img, a_min=-1000, a_max=1518)
        moving_img = np.clip(moving_img, a_min=-1000, a_max=1518)
        # fixed_img = np.clip(fixed_img, a_min=-1100, a_max=1518)
        # moving_img = np.clip(moving_img, a_min=-1100, a_max=1518)

        keypoints = []
        with open(self.keypoint_csv[step][0], newline='') as fixed_csvfile, open(self.keypoint_csv[step][1], newline='') as moving_csvfile:
            fixed_rows = csv.reader(fixed_csvfile)
            moving_rows = csv.reader(moving_csvfile)
            for index, (f_row, m_row) in enumerate(zip(fixed_rows, moving_rows)):
                pt = []
                pt.append(float(f_row[0].strip()))
                pt.append(float(f_row[1].strip()))
                pt.append(float(f_row[2].strip()))
                pt.append(float(m_row[0].strip()))
                pt.append(float(m_row[1].strip()))
                pt.append(float(m_row[2].strip()))
                keypoints.append(pt)

        keypoints = torch.from_numpy(np.array(keypoints))

        if self.need_label:
            fixed_mask = load_4D(self.mask_pair[step][0])
            moving_mask = load_4D(self.mask_pair[step][1])
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), torch.from_numpy(moving_mask).float(), torch.from_numpy(fixed_mask).float(), keypoints
        else:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(
                imgnorm(fixed_img)).float(), keypoints


class Dataset_epoch_NLST_nowin(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_list, mask_list, keypoint_csv, need_label=True):
        'Initialization'
        super(Dataset_epoch_NLST_nowin, self).__init__()
        # self.exp_path = exp_path
        self.img_pair = img_list
        self.mask_pair = mask_list
        self.keypoint_csv = keypoint_csv
        self.need_label = need_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img = load_4D(self.img_pair[step][0])
        moving_img = load_4D(self.img_pair[step][1])

        fixed_mask = load_4D(self.mask_pair[step][0])
        moving_mask = load_4D(self.mask_pair[step][1])

        # # Windowing
        # mask_fixed = fixed_img * fixed_mask
        fixed_img = np.clip(fixed_img, a_min=-1000, a_max=1600)
        moving_img = np.clip(moving_img, a_min=-1000, a_max=1600)

        keypoints = []
        with open(self.keypoint_csv[step][0], newline='') as fixed_csvfile, open(self.keypoint_csv[step][1], newline='') as moving_csvfile:
            fixed_rows = csv.reader(fixed_csvfile)
            moving_rows = csv.reader(moving_csvfile)
            for index, (f_row, m_row) in enumerate(zip(fixed_rows, moving_rows)):
                pt = []
                pt.append(float(f_row[0].strip()))
                pt.append(float(f_row[1].strip()))
                pt.append(float(f_row[2].strip()))
                pt.append(float(m_row[0].strip()))
                pt.append(float(m_row[1].strip()))
                pt.append(float(m_row[2].strip()))
                keypoints.append(pt)

        keypoints = torch.from_numpy(np.array(keypoints))

        if self.need_label:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), torch.from_numpy(moving_mask).float(), torch.from_numpy(fixed_mask).float(), keypoints
        else:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(
                imgnorm(fixed_img)).float(), keypoints


class Dataset_epoch_nopermutation(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=True):
        'Initialization'
        self.names = names
        self.norm = norm

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        moving_img = load_4D(self.names[step][0])
        fixed_img = load_4D(self.names[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float()


class Dataset_bratsreg_bidirection(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, fixed_list, moving_list, norm=True):
        'Initialization'
        self.fixed_list = fixed_list + moving_list
        self.moving_list = moving_list + fixed_list
        self.norm = norm

        self.index_pair = list(zip(self.fixed_list, self.moving_list))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img = load_4D(self.index_pair[step][0])
        moving_img = load_4D(self.index_pair[step][1])

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moving_img = np.clip(moving_img, a_min=0, a_max=1500)

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float()


class Dataset_bratsreg_bidirection_all(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, norm=True):
        'Initialization'
        self.fixed_t1ce_list = fixed_t1ce_list + moving_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list + fixed_t1ce_list
        self.fixed_flair_list = fixed_flair_list + moving_flair_list
        self.moving_flair_list = moving_flair_list + fixed_flair_list
        self.fixed_t2_list = fixed_t2_list + moving_t2_list
        self.moving_t2_list = moving_t2_list + fixed_t2_list
        self.norm = norm

        self.index_pair_t1ce = list(zip(self.fixed_t1ce_list, self.moving_t1ce_list))
        self.index_pair_flair = list(zip(self.fixed_flair_list, self.moving_flair_list))
        self.index_pair_t2 = list(zip(self.fixed_t2_list, self.moving_t2_list))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair_t1ce)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][0]))
        moving_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][1]))

        fixed_img_flair = imgnorm(load_4D(self.index_pair_flair[step][0]))
        moving_img_flair = imgnorm(load_4D(self.index_pair_flair[step][1]))

        fixed_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][0]))
        moving_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][1]))

        fixed_img = np.concatenate((fixed_img_t1ce, fixed_img_flair, fixed_img_t2), axis=0)
        moving_img = np.concatenate((moving_img_t1ce, moving_img_flair, moving_img_t2), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moving_img = np.clip(moving_img, a_min=0, a_max=1500)

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float()


class Dataset_bratsreg_bidirection_all4(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, fixed_t1_list, moving_t1_list, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, norm=True):
        'Initialization'
        super(Dataset_bratsreg_bidirection_all4, self).__init__()
        self.fixed_t1_list = fixed_t1_list
        self.moving_t1_list = moving_t1_list
        self.fixed_t1ce_list = fixed_t1ce_list + moving_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list + fixed_t1ce_list
        self.fixed_flair_list = fixed_flair_list + moving_flair_list
        self.moving_flair_list = moving_flair_list + fixed_flair_list
        self.fixed_t2_list = fixed_t2_list + moving_t2_list
        self.moving_t2_list = moving_t2_list + fixed_t2_list
        self.norm = norm

        self.index_pair_t1 = list(zip(self.fixed_t1_list, self.moving_t1_list))
        self.index_pair_t1ce = list(zip(self.fixed_t1ce_list, self.moving_t1ce_list))
        self.index_pair_flair = list(zip(self.fixed_flair_list, self.moving_flair_list))
        self.index_pair_t2 = list(zip(self.fixed_t2_list, self.moving_t2_list))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair_t1ce)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img_t1 = imgnorm(load_4D(self.index_pair_t1[step][0]))
        moving_img_t1 = imgnorm(load_4D(self.index_pair_t1[step][1]))

        fixed_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][0]))
        moving_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][1]))

        fixed_img_flair = imgnorm(load_4D(self.index_pair_flair[step][0]))
        moving_img_flair = imgnorm(load_4D(self.index_pair_flair[step][1]))

        fixed_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][0]))
        moving_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][1]))

        fixed_img = np.concatenate((fixed_img_t1, fixed_img_t1ce, fixed_img_flair, fixed_img_t2), axis=0)
        moving_img = np.concatenate((moving_img_t1, moving_img_t1ce, moving_img_flair, moving_img_t2), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moving_img = np.clip(moving_img, a_min=0, a_max=1500)

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float()


class Dataset_bratsreg_all_withmask(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, tumor_list, norm=True):
        'Initialization'
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_flair_list = fixed_flair_list
        self.moving_flair_list = moving_flair_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list
        self.tumor_list = tumor_list
        self.norm = norm

        self.index_pair_t1ce = list(zip(self.fixed_t1ce_list, self.moving_t1ce_list))
        self.index_pair_flair = list(zip(self.fixed_flair_list, self.moving_flair_list))
        self.index_pair_t2 = list(zip(self.fixed_t2_list, self.moving_t2_list))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair_t1ce)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][0]))
        moving_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][1]))

        fixed_img_flair = imgnorm(load_4D(self.index_pair_flair[step][0]))
        moving_img_flair = imgnorm(load_4D(self.index_pair_flair[step][1]))

        fixed_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][0]))
        moving_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][1]))

        fixed_img = np.concatenate((fixed_img_t1ce, fixed_img_flair, fixed_img_t2), axis=0)
        moving_img = np.concatenate((moving_img_t1ce, moving_img_flair, moving_img_t2), axis=0)

        tumor_img = load_4D(self.tumor_list[step])

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moving_img = np.clip(moving_img, a_min=0, a_max=1500)

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), torch.from_numpy(tumor_img).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float(), torch.from_numpy(tumor_img).float()


class Dataset_bratsreg_bidirection_t1cet2(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_t2_list, moving_t2_list, norm=True):
        'Initialization'
        self.fixed_t1ce_list = fixed_t1ce_list + moving_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list + fixed_t1ce_list
        self.fixed_t2_list = fixed_t2_list + moving_t2_list
        self.moving_t2_list = moving_t2_list + fixed_t2_list
        self.norm = norm

        self.index_pair_t1ce = list(zip(self.fixed_t1ce_list, self.moving_t1ce_list))
        self.index_pair_t2 = list(zip(self.fixed_t2_list, self.moving_t2_list))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair_t1ce)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][0]))
        moving_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][1]))

        fixed_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][0]))
        moving_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][1]))

        fixed_img = np.concatenate((fixed_img_t1ce, fixed_img_t2), axis=0)
        moving_img = np.concatenate((moving_img_t1ce, moving_img_t2), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moving_img = np.clip(moving_img, a_min=0, a_max=1500)

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float()


class Dataset_bratsreg_bidirection_all_id(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, norm=True):
        'Initialization'
        self.fixed_t1ce_list = fixed_t1ce_list + moving_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list + fixed_t1ce_list
        self.fixed_flair_list = fixed_flair_list + moving_flair_list
        self.moving_flair_list = moving_flair_list + fixed_flair_list
        self.fixed_t2_list = fixed_t2_list + moving_t2_list
        self.moving_t2_list = moving_t2_list + fixed_t2_list
        self.norm = norm

        self.index_pair_t1ce = list(zip(self.fixed_t1ce_list, self.moving_t1ce_list))
        self.index_pair_flair = list(zip(self.fixed_flair_list, self.moving_flair_list))
        self.index_pair_t2 = list(zip(self.fixed_t2_list, self.moving_t2_list))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair_t1ce)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][0]))[:, ::-1, ::-1]
        moving_img_t1ce = imgnorm(load_4D(self.index_pair_t1ce[step][1]))[:, ::-1, ::-1]

        fixed_img_flair = imgnorm(load_4D(self.index_pair_flair[step][0]))[:, ::-1, ::-1]
        moving_img_flair = imgnorm(load_4D(self.index_pair_flair[step][1]))[:, ::-1, ::-1]

        fixed_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][0]))[:, ::-1, ::-1]
        moving_img_t2 = imgnorm(load_4D(self.index_pair_t2[step][1]))[:, ::-1, ::-1]

        fixed_img = np.concatenate((fixed_img_t1ce, fixed_img_flair, fixed_img_t2), axis=0)
        moving_img = np.concatenate((moving_img_t1ce, moving_img_flair, moving_img_t2), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moving_img = np.clip(moving_img, a_min=0, a_max=1500)

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float()


class Dataset_bratsreg_bidirection_win(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, fixed_list, moving_list, norm=True):
        'Initialization'
        self.fixed_list = fixed_list + moving_list
        self.moving_list = moving_list + fixed_list
        self.norm = norm

        self.index_pair = list(zip(self.fixed_list, self.moving_list))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img = load_4D(self.index_pair[step][0])
        moving_img = load_4D(self.index_pair[step][1])

        fixed_img = np.clip(fixed_img, a_min=0, a_max=0.6*np.max(fixed_img))
        moving_img = np.clip(moving_img, a_min=0, a_max=0.6*np.max(moving_img))

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float()


class Dataset_bratsreg_withmask(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, fixed_list, moving_list, tumor_list, norm=True):
        'Initialization'
        self.fixed_list = fixed_list
        self.moving_list = moving_list
        self.tumor_list = tumor_list
        self.norm = norm

        self.index_pair = list(zip(self.fixed_list, self.moving_list, self.tumor_list))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        fixed_img = load_4D(self.index_pair[step][0])
        moving_img = load_4D(self.index_pair[step][1])
        tumor_img = load_4D(self.index_pair[step][2])

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moving_img = np.clip(moving_img, a_min=0, a_max=1500)

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), torch.from_numpy(tumor_img).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float(), torch.from_numpy(tumor_img).float()


class Dataset_epoch_local(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, seg_list, norm=True):
        'Initialization'
        self.names = names
        self.seg_list = seg_list
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))
        self.seg_pair = list(itertools.permutations(seg_list, 2))

        self.t_seg_list = [[16], [10, 49], [8, 47], [4, 43], [7, 46], [12, 51], [11, 50], [13, 52], [17, 53], [14],
                           [15], [18, 54], [24], [3, 42]]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        moving_img = load_4D(self.index_pair[step][0])
        fixed_img = load_4D(self.index_pair[step][1])

        moving_seg = nib.load(self.seg_pair[step][0]).get_fdata()
        reg_seg = np.zeros(moving_seg.shape, dtype=moving_seg.dtype)
        reg_seg = reg_seg + np.random.rand(1)
        for t in self.t_seg_list:
            if len(t) == 1:
                mask = (moving_seg == t[0])
                reg_seg[mask] = np.random.rand(1)
            else:
                mask = (moving_seg == t[0]) | (moving_seg == t[1])
                reg_seg[mask] = np.random.rand(1)
        reg_seg = np.reshape(reg_seg, (1,) + reg_seg.shape)

        if self.norm:
            return torch.from_numpy(imgnorm(moving_img)).float(), torch.from_numpy(imgnorm(fixed_img)).float(), torch.from_numpy(reg_seg).float()
        else:
            return torch.from_numpy(moving_img).float(), torch.from_numpy(fixed_img).float(), torch.from_numpy(reg_seg).float()


class Dataset_epoch_DIRLab(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=True):
        'Initialization'
        self.names = names
        self.norm = norm
        # self.index_pair = list(itertools.permutations(names, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        # img_A = load_4D(self.index_pair[step][0])
        # img_B = load_4D(self.index_pair[step][1])

        moving_id = math.floor(step/10)*10 + random.randint(0, 9)

        img_A = load_4D(self.names[step])
        img_B = load_4D(self.names[moving_id])

        # windowing
        img_A = np.clip(img_A, a_min=200, a_max=1500)
        img_B = np.clip(img_B, a_min=200, a_max=1500)

        if self.norm:
            img_A = imgnorm(img_A)
            img_B = imgnorm(img_B)

        patch_size = (256, 256, 64)
        s_x = random.randint(0, img_A.shape[1] - patch_size[0])
        s_y = random.randint(0, img_A.shape[2] - patch_size[1])
        s_z = random.randint(0, img_A.shape[3] - patch_size[2])

        img_A = img_A[:, s_x:s_x+patch_size[0], s_y:s_y+patch_size[1], s_z:s_z+patch_size[2]]
        img_B = img_B[:, s_x:s_x+patch_size[0], s_y:s_y+patch_size[1], s_z:s_z+patch_size[2]]

        return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch_DIRLab_full(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=True):
        'Initialization'
        self.names = names
        self.norm = norm
        # self.index_pair = list(itertools.permutations(names, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        # img_A = load_4D(self.index_pair[step][0])
        # img_B = load_4D(self.index_pair[step][1])

        moving_id = math.floor(step/10)*10 + random.randint(0, 9)

        img_A = load_4D(self.names[step])
        img_B = load_4D(self.names[moving_id])

        # windowing
        img_A = np.clip(img_A, a_min=200, a_max=1500)
        img_B = np.clip(img_B, a_min=200, a_max=1500)

        if self.norm:
            img_A = imgnorm(img_A)
            img_B = imgnorm(img_B)

        # patch_size = (256, 256, 64)
        # s_x = random.randint(0, img_A.shape[1] - patch_size[0])
        # s_y = random.randint(0, img_A.shape[2] - patch_size[1])
        # s_z = random.randint(0, img_A.shape[3] - patch_size[2])
        #
        # img_A = img_A[:, s_x:s_x+patch_size[0], s_y:s_y+patch_size[1], s_z:s_z+patch_size[2]]
        # img_B = img_B[:, s_x:s_x+patch_size[0], s_y:s_y+patch_size[1], s_z:s_z+patch_size[2]]

        return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch_DIRLab_full_newwin(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, names, norm=True):
        'Initialization'
        self.names = names
        self.norm = norm
        # self.index_pair = list(itertools.permutations(names, 2))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.names)

  def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        # img_A = load_4D(self.index_pair[step][0])
        # img_B = load_4D(self.index_pair[step][1])

        moving_id = math.floor(step/10)*10 + random.randint(0, 9)

        img_A = load_4D(self.names[step])
        img_B = load_4D(self.names[moving_id])

        # windowing
        img_A = np.clip(img_A, a_min=100, a_max=1050)
        img_B = np.clip(img_B, a_min=100, a_max=1050)

        if self.norm:
            img_A = imgnorm(img_A)
            img_B = imgnorm(img_B)

        # patch_size = (256, 256, 64)
        # s_x = random.randint(0, img_A.shape[1] - patch_size[0])
        # s_y = random.randint(0, img_A.shape[2] - patch_size[1])
        # s_z = random.randint(0, img_A.shape[3] - patch_size[2])
        #
        # img_A = img_A[:, s_x:s_x+patch_size[0], s_y:s_y+patch_size[1], s_z:s_z+patch_size[2]]
        # img_B = img_B[:, s_x:s_x+patch_size[0], s_y:s_y+patch_size[1], s_z:s_z+patch_size[2]]

        return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Predict_dataset(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=True):
        super(Predict_dataset, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list)
        moved_img = load_4D(self.move_list[index])
        fixed_label = load_4D(self.fixed_label_list)
        moved_label = load_4D(self.move_label_list[index])

        if self.norm:
            fixed_img = Norm_Zscore(imgnorm(fixed_img))
            moved_img = Norm_Zscore(imgnorm(moved_img))

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)
        fixed_label = torch.from_numpy(fixed_label)
        moved_label = torch.from_numpy(moved_label)

        if self.norm:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output
        else:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output


class Validation_Brats(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=True):
        super(Validation_Brats, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list[index])
        moved_img = load_4D(self.move_list[index])

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_all(Data.Dataset):
    def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, fixed_label_list, move_label_list, norm=True):
        super(Validation_Brats_all, self).__init__()
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_flair_list = fixed_flair_list
        self.moving_flair_list = moving_flair_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        # fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))[:, ::-1, ::-1]
        # moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))[:, ::-1, ::-1]
        # fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))[:, ::-1, ::-1]
        # moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))[:, ::-1, ::-1]
        # fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))[:, ::-1, ::-1]
        # moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))[:, ::-1, ::-1]

        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))
        moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1ce_img, fixed_flair_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1ce_img, moved_flair_img, moved_t2_img), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])
            # moved_key_list.append([239.-float(mov_line[1]), -1.*float(mov_line[2]), float(mov_line[3])])
            # fixed_key_list.append([239.-float(fixed_line[1]), -1.*float(fixed_line[2]), float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_all4(Data.Dataset):
    def __init__(self, fixed_t1_list, moving_t1_list, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, fixed_label_list, move_label_list, norm=True):
        super(Validation_Brats_all4, self).__init__()
        self.fixed_t1_list = fixed_t1_list
        self.moving_t1_list = moving_t1_list
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_flair_list = fixed_flair_list
        self.moving_flair_list = moving_flair_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        # fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))[:, ::-1, ::-1]
        # moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))[:, ::-1, ::-1]
        # fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))[:, ::-1, ::-1]
        # moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))[:, ::-1, ::-1]
        # fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))[:, ::-1, ::-1]
        # moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))[:, ::-1, ::-1]

        fixed_t1_img = imgnorm(load_4D(self.fixed_t1_list[index]))
        moved_t1_img = imgnorm(load_4D(self.moving_t1_list[index]))
        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))
        moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1_img, fixed_t1ce_img, fixed_flair_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1_img, moved_t1ce_img, moved_flair_img, moved_t2_img), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])
            # moved_key_list.append([239.-float(mov_line[1]), -1.*float(mov_line[2]), float(mov_line[3])])
            # fixed_key_list.append([239.-float(fixed_line[1]), -1.*float(fixed_line[2]), float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_all4_with_mask(Data.Dataset):
    def __init__(self, fixed_t1_list, moving_t1_list, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, fixed_label_list, move_label_list, tumor_list, norm=True):
        super(Validation_Brats_all4_with_mask, self).__init__()
        self.fixed_t1_list = fixed_t1_list
        self.moving_t1_list = moving_t1_list
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_flair_list = fixed_flair_list
        self.moving_flair_list = moving_flair_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

        self.tumor_list = tumor_list

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        # fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))[:, ::-1, ::-1]
        # moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))[:, ::-1, ::-1]
        # fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))[:, ::-1, ::-1]
        # moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))[:, ::-1, ::-1]
        # fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))[:, ::-1, ::-1]
        # moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))[:, ::-1, ::-1]

        fixed_t1_img = imgnorm(load_4D(self.fixed_t1_list[index]))
        moved_t1_img = imgnorm(load_4D(self.moving_t1_list[index]))
        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))
        moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1_img, fixed_t1ce_img, fixed_flair_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1_img, moved_t1ce_img, moved_flair_img, moved_t2_img), axis=0)

        tumor_img = load_4D(self.tumor_list[index])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])
            # moved_key_list.append([239.-float(mov_line[1]), -1.*float(mov_line[2]), float(mov_line[3])])
            # fixed_key_list.append([239.-float(fixed_line[1]), -1.*float(fixed_line[2]), float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'tumor_mask': tumor_img, 'index': index}
        return output


class Validation_Brats_all4_with_mask_submission(Data.Dataset):
    def __init__(self, fixed_t1_list, moving_t1_list, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, move_label_list, tumor_list, norm=True):
        super(Validation_Brats_all4_with_mask_submission, self).__init__()
        self.fixed_t1_list = fixed_t1_list
        self.moving_t1_list = moving_t1_list
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_flair_list = fixed_flair_list
        self.moving_flair_list = moving_flair_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.move_label_list = move_label_list
        self.norm = norm

        self.tumor_list = tumor_list

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        # fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))[:, ::-1, ::-1]
        # moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))[:, ::-1, ::-1]
        # fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))[:, ::-1, ::-1]
        # moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))[:, ::-1, ::-1]
        # fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))[:, ::-1, ::-1]
        # moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))[:, ::-1, ::-1]

        fixed_t1_img = imgnorm(load_4D(self.fixed_t1_list[index]))
        moved_t1_img = imgnorm(load_4D(self.moving_t1_list[index]))
        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))
        moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1_img, fixed_t1ce_img, fixed_flair_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1_img, moved_t1ce_img, moved_flair_img, moved_t2_img), axis=0)

        tumor_img = load_4D(self.tumor_list[index])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        mov_file = open(self.move_label_list[index])

        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line in moved_reader:
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            # moved_key_list.append([239.-float(mov_line[1]), -1.*float(mov_line[2]), float(mov_line[3])])
            # fixed_key_list.append([239.-float(fixed_line[1]), -1.*float(fixed_line[2]), float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                'move_label': np.array(moved_key_list), 'tumor_mask': tumor_img, 'index': index}
        return output


class Validation_Brats_all4_submission(Data.Dataset):
    def __init__(self, fixed_t1_list, moving_t1_list, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, move_label_list, norm=True):
        super(Validation_Brats_all4_submission, self).__init__()
        self.fixed_t1_list = fixed_t1_list
        self.moving_t1_list = moving_t1_list
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_flair_list = fixed_flair_list
        self.moving_flair_list = moving_flair_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        # fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))[:, ::-1, ::-1]
        # moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))[:, ::-1, ::-1]
        # fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))[:, ::-1, ::-1]
        # moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))[:, ::-1, ::-1]
        # fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))[:, ::-1, ::-1]
        # moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))[:, ::-1, ::-1]

        fixed_t1_img = imgnorm(load_4D(self.fixed_t1_list[index]))
        moved_t1_img = imgnorm(load_4D(self.moving_t1_list[index]))
        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))
        moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1_img, fixed_t1ce_img, fixed_flair_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1_img, moved_t1ce_img, moved_flair_img, moved_t2_img), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        mov_file = open(self.move_label_list[index])
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        moved_key_list = []

        for mov_line in moved_reader:
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_t1cet2(Data.Dataset):
    def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_t2_list, moving_t2_list, fixed_label_list, move_label_list, norm=True):
        super(Validation_Brats_t1cet2, self).__init__()
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1ce_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1ce_img, moved_t2_img), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])
            # moved_key_list.append([239.-float(mov_line[1]), -1.*float(mov_line[2]), float(mov_line[3])])
            # fixed_key_list.append([239.-float(fixed_line[1]), -1.*float(fixed_line[2]), float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_t1ce_submission(Data.Dataset):
    def __init__(self, fixed_t1ce_list, moving_t1ce_list, move_label_list, norm=True):
        super(Validation_Brats_t1ce_submission, self).__init__()
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list

        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        fixed_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_img = imgnorm(load_4D(self.moving_t1ce_list[index]))

        mov_file = open(self.move_label_list[index])

        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        moved_key_list = []

        for mov_line in moved_reader:
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            # moved_key_list.append([239.-float(mov_line[1]), -1.*float(mov_line[2]), float(mov_line[3])])
            # fixed_key_list.append([239.-float(fixed_line[1]), -1.*float(fixed_line[2]), float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_t1cet2_submission(Data.Dataset):
    def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_t2_list, moving_t2_list, move_label_list, norm=True):
        super(Validation_Brats_t1cet2_submission, self).__init__()
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1ce_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1ce_img, moved_t2_img), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        mov_file = open(self.move_label_list[index])

        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        moved_key_list = []

        for mov_line in moved_reader:
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            # moved_key_list.append([239.-float(mov_line[1]), -1.*float(mov_line[2]), float(mov_line[3])])
            # fixed_key_list.append([239.-float(fixed_line[1]), -1.*float(fixed_line[2]), float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_all_submission(Data.Dataset):
    def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, move_label_list, norm=True):
        super(Validation_Brats_all_submission, self).__init__()
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_flair_list = fixed_flair_list
        self.moving_flair_list = moving_flair_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))
        moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1ce_img, fixed_flair_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1ce_img, moved_flair_img, moved_t2_img), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        mov_file = open(self.move_label_list[index])
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line in moved_reader:
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_all_id(Data.Dataset):
    def __init__(self, fixed_t1ce_list, moving_t1ce_list, fixed_flair_list, moving_flair_list, fixed_t2_list, moving_t2_list, fixed_label_list, move_label_list, norm=True):
        super(Validation_Brats_all_id, self).__init__()
        self.fixed_t1ce_list = fixed_t1ce_list
        self.moving_t1ce_list = moving_t1ce_list
        self.fixed_flair_list = fixed_flair_list
        self.moving_flair_list = moving_flair_list
        self.fixed_t2_list = fixed_t2_list
        self.moving_t2_list = moving_t2_list

        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.moving_t1ce_list)

    def __getitem__(self, index):
        fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))[:, ::-1, ::-1]
        moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))[:, ::-1, ::-1]
        fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))[:, ::-1, ::-1]
        moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))[:, ::-1, ::-1]
        fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))[:, ::-1, ::-1]
        moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))[:, ::-1, ::-1]

        # fixed_t1ce_img = imgnorm(load_4D(self.fixed_t1ce_list[index]))
        # moved_t1ce_img = imgnorm(load_4D(self.moving_t1ce_list[index]))
        # fixed_flair_img = imgnorm(load_4D(self.fixed_flair_list[index]))
        # moved_flair_img = imgnorm(load_4D(self.moving_flair_list[index]))
        # fixed_t2_img = imgnorm(load_4D(self.fixed_t2_list[index]))
        # moved_t2_img = imgnorm(load_4D(self.moving_t2_list[index]))

        fixed_img = np.concatenate((fixed_t1ce_img, fixed_flair_img, fixed_t2_img), axis=0)
        moved_img = np.concatenate((moved_t1ce_img, moved_flair_img, moved_t2_img), axis=0)

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            # moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            # fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])
            moved_key_list.append([239.-float(mov_line[1]), -1.*float(mov_line[2]), float(mov_line[3])])
            fixed_key_list.append([239.-float(fixed_line[1]), -1.*float(fixed_line[2]), float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_win(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=True):
        super(Validation_Brats_win, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list[index])
        moved_img = load_4D(self.move_list[index])

        fixed_img = np.clip(fixed_img, a_min=0, a_max=0.6*np.max(fixed_img))
        moved_img = np.clip(moved_img, a_min=0, a_max=0.6*np.max(moved_img))

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'index': index}
        return output


class Validation_Brats_with_mask(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, tumor_list, norm=True):
        super(Validation_Brats_with_mask, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list

        self.tumor_list = tumor_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list[index])
        moved_img = load_4D(self.move_list[index])
        tumor_img = load_4D(self.tumor_list[index])

        # fixed_img = np.clip(fixed_img, a_min=0, a_max=1500)
        # moved_img = np.clip(moved_img, a_min=0, a_max=1500)

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'tumor_mask': torch.from_numpy(tumor_img).float(), 'index': index}
        return output


class Validation_Brats_with_mask_window(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, tumor_list, norm=True):
        super(Validation_Brats_with_mask_window, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list

        self.tumor_list = tumor_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list[index])
        moved_img = load_4D(self.move_list[index])
        tumor_img = load_4D(self.tumor_list[index])

        fixed_img = np.clip(fixed_img, a_min=0, a_max=0.6*np.max(fixed_img))
        moved_img = np.clip(moved_img, a_min=0, a_max=0.6*np.max(moved_img))

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_file = open(self.fixed_label_list[index])
        mov_file = open(self.move_label_list[index])

        fixed_reader = csv.reader(fixed_file)
        next(fixed_reader)
        moved_reader = csv.reader(mov_file)
        next(moved_reader)

        fixed_key_list = []
        moved_key_list = []

        for mov_line, fixed_line in zip(moved_reader, fixed_reader):
            moved_key_list.append([float(mov_line[1]), float(mov_line[2])+239., float(mov_line[3])])
            fixed_key_list.append([float(fixed_line[1]), float(fixed_line[2])+239., float(fixed_line[3])])

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)

        fixed_file.close()
        mov_file.close()

        output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                  'fixed_label': np.array(fixed_key_list), 'move_label': np.array(moved_key_list), 'tumor_mask': torch.from_numpy(tumor_img).float(), 'index': index}
        return output


class Validation_DIRLab(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, move_list, fixed_list, move_keypoint, fixed_keypoint, need_label=True):
        'Initialization'
        super(Validation_DIRLab, self).__init__()
        self.move_list = move_list
        self.fixed_list = fixed_list

        self.move_keypoint = move_keypoint
        self.fixed_keypoint = fixed_keypoint
        self.need_label = need_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        insp_img = load_4D(self.img_pair[step][0])
        exp_img = load_4D(self.img_pair[step][1])

        # Windowing
        insp_img = np.clip(insp_img, a_min=0, a_max=1500)
        exp_img = np.clip(exp_img, a_min=0, a_max=1500)

        keypoints = []
        with open(self.keypoint_csv[step], newline='') as csvfile:
            rows = csv.reader(csvfile)
            for index, row in enumerate(rows):
                pt = []
                pt.append(float(row[0].strip()))
                pt.append(float(row[1].strip()))
                pt.append(float(row[2].strip()))
                pt.append(float(row[3].strip()))
                pt.append(float(row[4].strip()))
                pt.append(float(row[5].strip()))
                keypoints.append(pt)

        keypoints = torch.from_numpy(np.array(keypoints))

        if self.need_label:
            insp_mask = load_4D(self.mask_pair[step][0])
            exp_mask = load_4D(self.mask_pair[step][1])
            return torch.from_numpy(imgnorm(insp_img)).float(), torch.from_numpy(
                imgnorm(exp_img)).float(), torch.from_numpy(insp_mask).float(), torch.from_numpy(exp_mask).float(), keypoints
        else:
            return torch.from_numpy(imgnorm(insp_img)).float(), torch.from_numpy(
                imgnorm(exp_img)).float(), keypoints



if __name__ == '__main__':
    # datapath = '/home/wing/Desktop/registration/miccai2019/data_and_aseg/crop_min_max/norm'
    # names = sorted(glob.glob(datapath + '/*.nii'))[0:255]
    # dataset = Dataset_epoch(names, False)
    # training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
    #                                      shuffle=False, num_workers=2)
    # for X, Y in training_generator:
    #     print("---")
    # # 64770 pairs
    # for i in range(0, dataset.__len__()):
    #     print(i, dataset.__len__())
    #     dataset.__getitem__(i)
    #     print("---")

    # imgshape = (5, 6, 7)
    # x = np.arange(imgshape[0])
    # y = np.arange(imgshape[1])
    # z = np.arange(imgshape[2])
    # grid = np.array(np.meshgrid(z, y, x)) #(3, 6, 7, 5)
    # grid = np.rollaxis(grid, 0, 4)# (6, 7, 5, 3)
    # grid = np.swapaxes(grid, 0, 2)# (5, 7, 6, 3)
    # grid = np.swapaxes(grid, 1, 2)# (5, 6, 7, 3)

    grid = generate_grid_unit((5, 6, 7))

    print(grid[:, :, :, 0].min(), grid[:, :, :, 0].max()) # -1, 1
    print(grid[:, :, :, 1].min(), grid[:, :, :, 1].max()) # -1, 1
    print(grid[:, :, :, 2].min(), grid[:, :, :, 2].max()) # -1, 1

    grid = generate_grid((5, 6, 7))
    # print(grid[:, :, :, 0]) # 0-6
    # print(grid[:, :, :, 1]) # 0-5
    # print(grid[:, :, :, 2]) # 0-4
    print("done")
