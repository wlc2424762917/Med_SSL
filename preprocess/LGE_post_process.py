import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from skimage import measure, morphology
from scipy import ndimage as ndi


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_img(src_file_path, src_file_name, src_save_path, gt_save_path, sr_save_path, dtype=sitk.sitkFloat32, filter=False):  # for .mhd .nii .nrrd
    if not os.path.exists(src_file_path):
        return
    save_src_path = mkdir(os.path.join(src_save_path))
    save_gt_path = mkdir(os.path.join(gt_save_path))
    save_sr_path = mkdir(os.path.join(sr_save_path))

    src_slice_1_path = os.path.join(src_file_path, "LGE_slice_1.png")
    src_slice_2_path = os.path.join(src_file_path, "LGE_slice_2.png")
    seg_slice_1_path = os.path.join(src_file_path, "LGE_slice_1_with_label.png")
    seg_slice_2_path = os.path.join(src_file_path, "LGE_slice_2_with_label.png")
    src_slice_paths = [
        src_slice_1_path,
        src_slice_2_path
    ]
    seg_slice_paths = [
        seg_slice_1_path,
        seg_slice_2_path
    ]

    slice_idx = 1
    for src_slice_path, seg_slice_path in zip(src_slice_paths, seg_slice_paths):
        image = Image.open(src_slice_path).convert('L')
        image_data = np.array(image)
        image_data_norm = image_data
        H, W = image_data_norm.shape
        image_data_norm = image_data_norm[H//2-256:H//2+256, W//2-256:W//2+256]
        image_data_norm = np.expand_dims(image_data_norm, 0)

        print(src_slice_path, " data_shape:", image_data_norm.shape)
        image_with_mask = Image.open(seg_slice_path).convert('L')
        image_with_mask_norm = np.array(image_with_mask)
        image_with_mask_norm = image_with_mask_norm[H//2-256:H//2+256, W//2-256:W//2+256]
        image_with_mask_norm = np.expand_dims(image_with_mask_norm, 0)
        mask_data = image_with_mask_norm - image_data_norm
        mask_data_norm = mask_data  # no case norm

        image_slice = image_data_norm[0, :, :]
        image_slice = image_slice.astype(np.uint16)
        mask_slice = mask_data_norm[0, :, :]
        mask_slice = mask_slice > 0
        mask_slice = mask_slice.astype(np.uint32)

        # 进行连通性分析
        labels = measure.label(mask_slice, connectivity=2)
        props = measure.regionprops(labels)

        # 获取每个连通域的面积
        areas = np.bincount(labels.flat)

        # 定义最小连通域面积
        min_area = 225

        # 自定义结构元素
        selem = np.array([[1]])

        # 检查每个连通域是否有空洞，并进行补全
        for i, prop in enumerate(props):
            mask = labels == i + 1
            if not prop.filled_area == np.sum(mask):
                print(f"连通域{i + 1}有空洞，进行补全")
                mask_dilated = ndi.binary_dilation(mask, structure=selem)
                mask_flood = morphology.flood_fill(mask_dilated, (0, 0), 1)
                mask_slice[mask_flood == 0] = 1

        # 筛选掉小连通域
        for i in range(1, len(areas)):
            if areas[i] < min_area:
                mask_slice[labels == i] = 0

        mask_slice_cliped = np.clip(mask_slice, 0, 1)
        image_slice_down = zoom(image_slice, 0.125, order=3)
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
            slice_idx += 1


def read_dataset(src_file_paths, src_save_path, gt_save_path, sr_save_path, filter=False):
    for idx_data in range(len(src_file_paths)):
        print('{} / {}'.format(idx_data + 1, len(src_file_paths)))
        patient_path = src_file_paths[idx_data]
        acute_path = os.path.join(patient_path, 'acute')
        chronic_path = os.path.join(patient_path, 'chronic')
        print(acute_path)
        print(chronic_path)
        patient_nameext, _ = os.path.splitext(patient_path)
        _, patient_name = os.path.split(patient_nameext)
        acute_name = patient_name + '_acute'
        chronic_name = patient_name + '_chronic'

        print(patient_name)
        read_img(acute_path, acute_name, src_save_path, gt_save_path, sr_save_path, filter=filter)
        read_img(chronic_path, chronic_name, src_save_path, gt_save_path, sr_save_path, filter=filter)


if __name__ == "__main__":
    # src_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse\imagesTr"
    # src_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\imagesTr"
    # gt_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse\labelsTr"
    # gt_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\label_segTr"
    # sr_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\synapse_seg_sr\label_srTr"

    src_file_paths = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\LGE_post"
    src_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\LGE_post_m\imageTr"
    gt_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\LGE_post_m\label_segTr"
    sr_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\LGE_post_m\label_srTr"

    patient_paths = glob(os.path.join(src_file_paths, '*'))
    src_path_save = mkdir(src_save_path)
    gt_path_save = mkdir(gt_save_path)
    sr_path_save = mkdir(sr_save_path)

    # src_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'src'))
    # gt_path_save = mkdir(os.path.join(path_raw, f'{dataset_type}set', 'gt'))
    read_dataset(patient_paths, src_path_save, gt_path_save, sr_path_save, filter=False)
