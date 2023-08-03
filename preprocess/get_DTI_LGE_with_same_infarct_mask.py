import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
import os
import cv2
from scipy import ndimage
import numpy as np
import shutil
import SimpleITK as sitk

AB_path_root = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\AB"
A_path_root = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\A"
B_path_root = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\B"
myo_path_root = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\myo_gt"
infarct_path_root = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\infarct_gt"

AB_paths = glob(os.path.join(AB_path_root, "*.nii.gz"))
myo_paths = glob(os.path.join(myo_path_root, "*.nii.gz"))
infarct_paths = glob(os.path.join(infarct_path_root, "*.nii.gz"))
print(len(AB_paths))

paired_AB = []
paired_myo = []
paired_infarct = []

for path, myo_path, infarct_path in zip(AB_paths, myo_paths, infarct_paths):
    AB_data = nib.load(path).get_fdata()
    AB_name = os.path.basename(path)
    print(AB_name)
    myo_data = nib.load(myo_path).get_fdata()
    myo_name = os.path.basename(myo_path)
    infarct_data = nib.load(infarct_path).get_fdata()
    infarct_name = os.path.basename(infarct_path)
    infarct_data = ndimage.zoom(infarct_data, 0.125, order=0)

    # 检查所有前景像素是否都在图像B中
    all_data = myo_data + infarct_data
    all_data = (all_data != 0).astype(np.uint32)
    myo_data = (myo_data != 0).astype(np.uint32)
    num = all_data - myo_data

    print(np.sum(num))
    frac = np.sum(num) / (np.sum(infarct_data) + 0.05)
    print(frac)
    if frac < 0.3 and np.sum(infarct_data) > 0:
        new_infarct = infarct_data - num
        paired_AB.append(path)
        paired_myo.append(myo_path)
        paired_infarct.append(new_infarct)

dest_AB_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE_paired\AB"
dest_A_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE_paired\A"
dest_B_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE_paired\B"
dest_myo_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE_paired\myo_gt"
dest_infarct_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE_paired\infarct_gt"

for AB, myo, infarct in zip(paired_AB, paired_myo, paired_infarct):
    AB_name = os.path.basename(AB)
    A = os.path.join(A_path_root, AB_name)
    B = os.path.join(B_path_root, AB_name)

    shutil.copy(A, dest_A_path)
    shutil.copy(B, dest_B_path)
    shutil.copy(AB, dest_AB_path)
    shutil.copy(myo, dest_myo_path)
    # shutil.copy(infarct, dest_infarct_path)
    sitk.WriteImage(sitk.GetImageFromArray(infarct), os.path.join(dest_infarct_path, AB_name))
