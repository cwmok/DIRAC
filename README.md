# Unsupervised Deformable Image Registration with Absent Correspondences in Pre-operative and Post-Recurrence Brain Tumor MRI Scans
This is the official Pytorch implementation of "Unsupervised Deformable Image Registration with Absent Correspondences in Pre-operative and Post-Recurrence Brain Tumor MRI Scans" (MICCAI 2022), written by Tony C. W. Mok and Albert C. S. Chung.

## Prerequisites
- `Python 3.5.2+`
- `Pytorch 1.3.0 - 1.9.1`
- `NumPy`
- `NiBabel`
- `Scipy`

This code has been tested with `Pytorch 1.10.0` and NVIDIA TITAN RTX GPU.

## Inference

Inference for DIRAC:
```
python BRATS_test_DIRAC.py
```

Inference for DIRAC-D:
```
python BRATS_test_DIRAC_D.py
```

## Train your own model
Step 1: Download the BraTS-Reg dataset from https://www.med.upenn.edu/cbica/brats-reg-challenge/

Step 2: Define and split the dataset into training and validation set, i.e., 'Dataset/BraTSReg_self_train' and 'Dataset/BraTSReg_self_valid', respectively.

Step 3: `python BRATS_train_DIRAC.py` to train the DIRAC model or `python BRATS_train_DIRAC_D.py` to train the DIRAC-D model.

## Publication
If you find this repository useful, please cite:
- **Unsupervised Deformable Image Registration with Absent Correspondences in Pre-operative and Post-Recurrence Brain Tumor MRI Scans**  
[Tony C. W. Mok](https://cwmok.github.io/ "Tony C. W. Mok"), Albert C. S. Chung  
MICCAI 2022. [eprint arXiv:2206.03900](https://arxiv.org/abs/2206.03900)

- **Conditional Deformable Image Registration with Convolutional Neural Network**  
[Tony C. W. Mok](https://cwmok.github.io/ "Tony C. W. Mok"), Albert C. S. Chung  
MICCAI 2021. [eprint arXiv:2106.12673](https://arxiv.org/abs/2106.12673)

###### Keywords
Keywords: Absent correspondences, Patient-specific registration, Deformable registration