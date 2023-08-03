import os
import numpy as np
from glob import glob
from skimage.transform import resize
import SimpleITK as sitk
from matplotlib import pylab as plt
import nibabel as nib
from collections import OrderedDict
from skimage.morphology import label
import pickle

skips = ["MI12_acute_002.nii.gz",

         "MI27_chronic_001.nii.gz",

         "MI29_acute_001.nii.gz",

         "MI29_acute_002.nii.gz",

         "MI38_acute_001.nii.gz",

         "MI41_chronic_001.nii.gz",

         "MI41_chronic_002.nii.gz",

         "MI42_acute_001.nii.gz",

         "MI42_acute_002.nii.gz",

         "MI65_chronic_002.nii.gz"]


def read_img(mod_1_path, mod_2_path, src_file_name, mod_1_save_path, mod_2_save_path, mod_12_save_path,
             infarct_gt_file_path, myo_gt_file_path, infarct_gt_save_path, myo_gt_save_path):  # for .mhd .nii .nrrd
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    if not os.path.exists(mod_1_path):
        raise FileNotFoundError
    mod_1_data = nib.load(mod_1_path).get_fdata()
    mod_2_data = nib.load(mod_2_path).get_fdata()

    mod_1_data_norm = mod_1_data  # no case norm for mod_1
    mod_2_data_norm = mod_2_data  # no case norm for mod_2
    gt_file_name = src_file_name
    print(src_file_name, " data_shape:", mod_1_data_norm.shape)
    print(mod_1_data_norm.max())

    infarct_mask_data = nib.load(infarct_gt_file_path).get_fdata()
    myo_mask_data = nib.load(myo_gt_file_path).get_fdata()

    # save_mod_1_path = mkdir(os.path.join(mod_1_save_path, src_file_name))
    # save_mod_2_path = mkdir(os.path.join(mod_2_save_path, src_file_name))
    # save_mod_12_path = mkdir(os.path.join(mod_12_save_path, src_file_name))
    # save_infarct_gt_path = mkdir(os.path.join(infarct_gt_save_path, gt_file_name))
    # save_myo_gt_path = mkdir(os.path.join(myo_gt_save_path, gt_file_name))
    save_mod_1_path = mkdir(os.path.join(mod_1_save_path))
    save_mod_2_path = mkdir(os.path.join(mod_2_save_path))
    save_mod_12_path = mkdir(os.path.join(mod_12_save_path))
    save_infarct_gt_path = mkdir(os.path.join(infarct_gt_save_path))
    save_myo_gt_path = mkdir(os.path.join(myo_gt_save_path))

    mod_1_slice = mod_1_data_norm[:, :]
    mod_2_slice = mod_2_data_norm[:, :]
    print("mod1", mod_1_slice.max())
    print("mod2", mod_2_slice.max())
    # mod_2_slice = 255 * (mod_2_slice - mod_2_slice.min()) / (mod_2_slice.max() - mod_2_slice.min())
    mod_2_slice = mod_2_slice.astype(np.uint8)
    print("mod_1_slice.shape:", mod_1_slice.shape)
    print("mod_2_slice.shape:", mod_2_slice.shape)
    combine_slice = np.concatenate((mod_1_slice, mod_2_slice), axis=0)
    print("combine_slice.shape:", combine_slice.shape)
    # np.save(os.path.join(save_mod_1_path, '{}.npy'.format(src_file_name)), mod_1_slice)
    # np.save(os.path.join(save_mod_2_path, '{}.npy'.format(gt_file_name)), mod_2_slice)
    # np.save(os.path.join(save_mod_12_path, '{}.npy'.format(gt_file_name)), combine_slice)
    # np.save(os.path.join(infarct_gt_save_path, '{}.npy'.format(gt_file_name)), infarct_mask_data)
    # np.save(os.path.join(myo_gt_save_path, '{}.npy'.format(gt_file_name)), myo_mask_data)
    if src_file_name in skips:
        print('skipping ', src_file_name)
        return
    sitk.WriteImage(sitk.GetImageFromArray(mod_1_slice),
                    os.path.join(save_mod_1_path, '{}.nii.gz'.format(src_file_name)))
    sitk.WriteImage(sitk.GetImageFromArray(mod_2_slice),
                    os.path.join(save_mod_2_path, '{}.nii.gz'.format(gt_file_name)))
    sitk.WriteImage(sitk.GetImageFromArray(combine_slice),
                    os.path.join(save_mod_12_path, '{}.nii.gz'.format(gt_file_name)))
    sitk.WriteImage(sitk.GetImageFromArray(infarct_mask_data),
                    os.path.join(save_infarct_gt_path, '{}.nii.gz'.format(gt_file_name)))
    sitk.WriteImage(sitk.GetImageFromArray(myo_mask_data),
                    os.path.join(save_myo_gt_path, '{}.nii.gz'.format(gt_file_name)))


def read_dataset(patients, mod_1_path_root, mod_2_path_root, infarct_gt_file_path_root, myo_gt_file_path_root,
                 mod_1_save_path, mod_2_save_path, mod_12_save_path, infarct_gt_save_path, myo_gt_save_path):
    for idx_data in range(len(patients)):
        print('{} / {}'.format(idx_data + 1, len(patients)))

        mod_1_path = os.path.join(mod_1_path_root, patients[idx_data])
        mod_2_path = os.path.join(mod_2_path_root, patients[idx_data])
        infarct_mask_path = os.path.join(infarct_gt_file_path_root, patients[idx_data])
        myo_mask_path = os.path.join(myo_gt_file_path_root, patients[idx_data])

        print(patients[idx_data])

        read_img(mod_1_path, mod_2_path, patients[idx_data], mod_1_save_path, mod_2_save_path, mod_12_save_path,
                 infarct_mask_path, myo_mask_path, infarct_gt_save_path, myo_gt_save_path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


if __name__ == '__main__':

    # path_raw_dataset_type = r"D:\braTS_t"
    # mod_1_path_save = mkdir(r"D:\style_transfer_test\A")
    # mod_12_path_save = mkdir(r"D:\style_transfer_test\AB")
    # mod_2_path_save = mkdir(r"D:\style_transfer_test\B")
    # gt_path_save = mkdir(r"D:\style_transfer_test\gt")

    LGE_root_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\LGE_post_p\imagesTr"
    DTI_root_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_post_p\imagesTr"
    infarct_gt_file_path_root = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\LGE_post_p\label_segTr"
    myo_gt_file_path_root = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_post_p\myo_mask"

    mod_1_path_save = mkdir(r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\A")
    mod_12_path_save = mkdir(r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\AB")
    mod_2_path_save = mkdir(r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\B")
    infarct_gt_path_save = mkdir(r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\infarct_gt")
    myo_gt_path_save = mkdir(r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_LGE\myo_gt")

    raw_LGE_paths = glob(os.path.join(LGE_root_path, "*.nii.gz"))
    raw_LGE_paths.sort()
    raw_DTI_paths = glob(os.path.join(DTI_root_path, "*.nii.gz"))
    raw_DTI_paths.sort()

    LGE_paths = []
    DTI_paths = []
    for idx_data in range(len(raw_LGE_paths)):
        LGE_path = raw_LGE_paths[idx_data]
        LGE_name = os.path.basename(LGE_path)
        LGE_paths.append(LGE_name)
    for idx_data in range(len(raw_DTI_paths)):
        DTI_path = raw_DTI_paths[idx_data]
        DTI_name = os.path.basename(DTI_path)
        DTI_paths.append(DTI_name)
    # print(LGE_paths)
    # print(DTI_paths)
    intersect = list(set(LGE_paths).intersection(set(DTI_paths)))
    intersect.sort()
    print(len(intersect))
    patients = intersect

    read_dataset(patients, LGE_root_path, DTI_root_path, infarct_gt_file_path_root, myo_gt_file_path_root,
                 mod_1_path_save, mod_2_path_save, mod_12_path_save, infarct_gt_path_save, myo_gt_path_save)
