import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as scio


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_img(src_file_path, src_file_name, src_save_path, gt_save_path, dtype=sitk.sitkFloat32,
             filter=False):  # for .mhd .nii .nrrd
    if not os.path.exists(src_file_path):
        return
    save_src_path = mkdir(os.path.join(src_save_path))
    save_gt_path = mkdir(os.path.join(gt_save_path))
    src_slice_paths = sorted(glob(os.path.join(src_file_path, "*diastole")))
    diffusion_slice_paths = sorted(glob(os.path.join(src_slice_paths[0], "diffusion_images", "*dcm")))
    # print(os.path.join(src_slice_paths[0], "diffusion_images", "*dcm"))
    _, diffusion_slice_name = os.path.split(diffusion_slice_paths[0])

    src_slice_1_path = os.path.join(src_slice_paths[0], "diffusion_images", diffusion_slice_name)
    src_slice_2_path = os.path.join(src_slice_paths[1], "diffusion_images", diffusion_slice_name)
    crop_slice_1_path = os.path.join(src_slice_paths[0], "matlab_data\crop_info.mat")
    crop_slice_2_path = os.path.join(src_slice_paths[1], "matlab_data\crop_info.mat")
    myo_slice_1_path = os.path.join(src_slice_paths[0], "matlab_data\myocardium_roi_slice_01.mat")
    myo_slice_2_path = os.path.join(src_slice_paths[1], "matlab_data\myocardium_roi_slice_01.mat")
    src_slice_paths = [
        src_slice_1_path,
        src_slice_2_path
    ]
    crop_slice_paths = [
        crop_slice_1_path,
        crop_slice_2_path
    ]
    myo_slice_paths = [
        myo_slice_1_path,
        myo_slice_2_path
    ]

    slice_idx = 1
    for src_slice_path, crop_slice_path, myo_slice_path in zip(src_slice_paths, crop_slice_paths, myo_slice_paths):
        image = sitk.ReadImage(src_slice_path)
        image_data = sitk.GetArrayFromImage(image)
        image_data_norm = image_data[0]
        print(image_data_norm.shape)

        myo_data = scio.loadmat(myo_slice_path)
        myo_mask = myo_data['current_mask_myo']


        center_data = myo_data['current_centroid'].reshape(-1)
        center_row = int(center_data[1])
        center_col = int(center_data[0])

        crop_data = scio.loadmat(crop_slice_path)
        all_locs = np.argwhere(crop_data["crop_mask"] == 1)

        crop_row_min = min(all_locs[:, 0])
        crop_col_min = min(all_locs[:, 1])
        crop_row_max = max(all_locs[:, 0])
        crop_col_max = max(all_locs[:, 1])
        # print(min(all_locs[:, 0]))
        # print(max(all_locs[:, 0]))
        # print(min(all_locs[:, 1]))
        # print(max(all_locs[:, 1]))

        crop_r = 32

        if crop_row_min + center_row - crop_r < 0:
            image_data_norm = image_data_norm[0:2*crop_r,
                              crop_col_min + center_col - crop_r:crop_col_min + center_col + crop_r]
        else:
            image_data_norm = image_data_norm[crop_row_min + center_row - crop_r:crop_row_min + center_row + crop_r,
                          crop_col_min + center_col - crop_r:crop_col_min + center_col + crop_r]

        # image_data_norm = np.expand_dims(image_data_norm, 0)
        myo_mask = np.pad(myo_mask, ((128, 128), (128, 128)), 'constant', constant_values=((0, 0), (0, 0)))
        myo_mask = myo_mask[128 + center_row - crop_r:128 + center_row + crop_r,
                   128  + center_col - crop_r:128 + center_col + crop_r]
        myo_mask_norm = myo_mask
        # myo_mask_norm = np.expand_dims(myo_mask, 0)

        # image_slice = image_data_norm[0, :, :]
        image_slice = image_data_norm
        image_slice = image_slice.astype(np.uint16)
        # myo_mask_slice = myo_mask_norm[0, :, :]
        myo_mask_slice = myo_mask_norm
        myo_mask_slice = myo_mask_slice > 0
        myo_mask_slice = myo_mask_slice.astype(np.uint32)
        mask_slice_cliped = np.clip(myo_mask_slice, 0, 1)

        # image_slice_down = zoom(image_slice, 0.125, order=3)
        # new_mask_slice = np.concatenate((np.expand_dims(mask_slice, 0),  np.expand_dims(image_slice, 0)), axis=0)

        myo_mask_slice = sitk.GetImageFromArray(myo_mask_slice)
        image_slice = sitk.GetImageFromArray(image_slice)

        if filter:
            if np.sum(mask_slice_cliped) > 500:
                sitk.WriteImage(image_slice,
                                os.path.join(save_src_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))
                sitk.WriteImage(myo_mask_slice,
                                os.path.join(save_gt_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))

            # np.save(os.path.join(save_src_path, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), image_slice_down)
            # np.save(os.path.join(save_gt_path, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), new_mask_slice)
        else:
            sitk.WriteImage(image_slice,
                            os.path.join(save_src_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))
            sitk.WriteImage(myo_mask_slice,
                            os.path.join(save_gt_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))
            # np.save(os.path.join(save_src_path, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), image_slice)
            # np.save(os.path.join(save_gt_path, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), myo_mask_slice)
        slice_idx += 1


def read_dataset(src_file_paths, src_save_path, gt_save_path, filter=False):
    for idx_data in range(len(src_file_paths)):
        print('{} / {}'.format(idx_data + 1, len(src_file_paths)))
        patient_path = src_file_paths[idx_data]
        patient_nameext, _ = os.path.splitext(patient_path)
        _, patient_name = os.path.split(patient_nameext)
        name_list = patient_name.split(' ')
        number, acute = name_list[1], name_list[2]
        if acute == "follow":
            acute = "chronic"
        patient_name = 'MI' + number + '_' + acute
        print(patient_name)
        read_img(patient_path, patient_name, src_save_path, gt_save_path, filter=filter)


if __name__ == "__main__":
    # src_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse\imagesTr"
    # src_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\imagesTr"
    # gt_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse\labelsTr"
    # gt_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\label_segTr"
    # sr_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\label_srTr"

    acute_src_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI\Ramyah acute STEMI DTI with B0 processed on Mac mini"
    chronic_src_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI\Ramyah chronic STEMI DTI with B0 processed on Mac mini"

    src_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\DTI_post_p\image"
    gt_save_path = r"/data/DTI_post_p/label_sr"

    acute_patient_paths = glob(os.path.join(acute_src_file_paths, '*'))
    chronic_patient_paths = glob(os.path.join(chronic_src_file_paths, '*'))

    patient_paths = acute_patient_paths + chronic_patient_paths
    patient_paths.sort()
    print(patient_paths)

    src_path_save = mkdir(src_save_path)
    gt_path_save = mkdir(gt_save_path)

    # src_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'src'))
    # gt_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'gt'))
    read_dataset(patient_paths, src_path_save, gt_path_save, filter=False)
