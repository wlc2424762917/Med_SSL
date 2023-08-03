import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import os
from glob import glob
import matplotlib.pyplot as plt

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_img(src_file_path, src_file_name, src_save_path, gt_file_path, gt_file_name, gt_save_path, sr_save_path, dtype=sitk.sitkFloat32, filter=False):  # for .mhd .nii .nrrd
    if not os.path.exists(src_file_path):
        raise FileNotFoundError
    image = sitk.ReadImage(src_file_path)
    image_data = sitk.GetArrayFromImage(image)  # N*H*W
    image_data_norm = image_data
    print(src_file_name, " data_shape:", image_data_norm.shape)
    if not os.path.exists(gt_file_path):
        raise FileNotFoundError
    mask = sitk.ReadImage(gt_file_path)
    mask_data = sitk.GetArrayFromImage(mask)  # N*H*W
    mask_data_norm = mask_data  # no case norm
    # print(src_file_name, " data_shape:", image_data_norm.shape)

    save_src_path = mkdir(os.path.join(src_save_path))
    save_gt_path = mkdir(os.path.join(gt_save_path))
    save_sr_path = mkdir(os.path.join(sr_save_path))

    for slice_idx in range(0, mask_data_norm.shape[0]):
        image_slice = image_data_norm[slice_idx, :, :]
        mask_slice = mask_data_norm[slice_idx, :, :]
        mask_slice_cliped = np.clip(mask_slice, 0, 1)

        image_slice_down = zoom(image_slice, 0.5, order=3)
        # new_mask_slice = np.concatenate((np.expand_dims(mask_slice, 0),  np.expand_dims(image_slice, 0)), axis=0)

        image_slice_down = sitk.GetImageFromArray(image_slice_down)
        mask_slice = sitk.GetImageFromArray(mask_slice)
        image_slice = sitk.GetImageFromArray(image_slice)

    if filter:
        if np.sum(mask_slice_cliped) > 500:
            sitk.WriteImage(image_slice_down,
                            os.path.join(save_src_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))
            sitk.WriteImage(mask_slice,
                            os.path.join(save_gt_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))
            sitk.WriteImage(image_slice,
                            os.path.join(save_sr_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))

        # np.save(os.path.join(save_src_path, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), image_slice_down)
            # np.save(os.path.join(save_gt_path, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), new_mask_slice)
    else:
        sitk.WriteImage(image_slice_down,
                        os.path.join(save_src_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))
        sitk.WriteImage(mask_slice,
                        os.path.join(save_gt_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))
        sitk.WriteImage(image_slice,
                        os.path.join(save_sr_path, '{}_{:03d}.nii.gz'.format(src_file_name, slice_idx)))
        # np.save(os.path.join(save_src_path, '{}_{:03d}.npy'.format(src_file_name, slice_idx)), image_slice_down)
        # np.save(os.path.join(save_gt_path, '{}_{:03d}.npy'.format(gt_file_name, slice_idx)), new_mask_slice)


def read_dataset(src_file_paths, src_save_path, gt_file_paths, gt_save_path, sr_save_path, filter=False):
    for idx_data in range(len(src_file_paths)):
        print('{} / {}'.format(idx_data + 1, len(src_file_paths)))
        img_path = src_file_paths[idx_data]
        mask_path = gt_file_paths[idx_data]
        img_nameext, _ = os.path.splitext(img_path)
        img_nameext, _ = os.path.splitext(img_nameext)
        mask_nameext, _ = os.path.splitext(mask_path)
        mask_nameext, _ = os.path.splitext(mask_nameext)
        _, img_name = os.path.split(img_nameext)
        _, mask_name = os.path.split(mask_nameext)
        print(mask_name)
        print(img_name)
        read_img(img_path, img_name, src_save_path, mask_path, mask_name, gt_save_path, sr_save_path, filter=filter)


if __name__ == "__main__":
    # src_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse\imagesTr"
    # src_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\imagesTr"
    # gt_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse\labelsTr"
    # gt_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\label_segTr"
    # sr_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\label_srTr"

    src_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\acdc\imagesTr"
    src_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\acdc_seg_sr\imagesTr"
    gt_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\acdc\labelsTr"
    gt_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\acdc_seg_sr\label_segTr"
    sr_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\acdc_seg_sr\label_srTr"

    image_paths = glob(os.path.join(src_file_paths, '*.nii.gz'))
    mask_paths = glob(os.path.join(gt_file_paths, '*.nii.gz'))

    src_path_save = mkdir(src_save_path)
    gt_path_save = mkdir(gt_save_path)
    sr_path_save = mkdir(sr_save_path)

    # src_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'src'))
    # gt_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'gt'))
    read_dataset(image_paths, src_path_save, mask_paths, gt_path_save, sr_path_save, filter=False)
