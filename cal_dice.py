import os
import nibabel as nib
from skimage import measure, morphology
from utils.utils import dice, psnr, ssim
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# folder_path = r"D:\fromLab8\dice\inference_outputs\swinTrans_sr_seg_net_with_skip_from_meta_ssl_new_divide\seg"
# save_path = r"D:\fromLab8\dice\inference_outputs\post_processed_swinTrans_sr_seg_net_with_skip_from_meta_ssl_new_divide"
# target_path = r"D:\fromLab8\dice\inference_outputs\gt"

folder_path = r"D:\fromLab8\dice\inference_outputs\swinTrans_seg_sr_skip_LGE_style_DTI_from_meta_ssl\seg"
save_path = r"D:\fromLab8\dice\inference_outputs\post_processed_swinTrans_sr_seg_net_with_skip_from_meta_ssl_new_divide"
target_path = r"D:\fromLab8\dice\inference_outputs\gt"

mkdir(save_path)

dice_list = []
for file_name, target_name in zip(sorted(os.listdir(folder_path)), sorted(os.listdir(target_path))):
    if file_name.endswith(".nii.gz"):
        # 读取NII.gz文件
        file_path = os.path.join(folder_path, file_name)
        img = nib.load(file_path).get_fdata()
        label = nib.load(os.path.join(target_path, target_name)).get_fdata()

        # 将数组转换为二进制图像
        img = img > 0

        # 使用label函数标记连通区域
        labels = measure.label(img)

        # 计算每个标记区域的面积
        regions = measure.regionprops(labels)
        areas = [region.area for region in regions]

        # 计算最大面积
        max_area = max(areas)

        # 根据最大面积筛选连通区域
        selected_labels = []
        for i, region in enumerate(regions):
            if region.area >= max_area/100:
                selected_labels.append(i+1)

        # 使用morphology.remove_small_objects函数删除小的连通区域
        mask = morphology.remove_small_objects(labels, min_size=max_area/100)

        # 将删除后的连通区域转换为二进制图像
        result = mask > 0

        # 保存结果到新的NII.gz文件
        save_file_name = "processed_" + file_name
        save_file_path = os.path.join(save_path, save_file_name)
        nib.save(nib.Nifti1Image(result.astype(int), affine=None), save_file_path)

        dice_score = dice(result.astype(int), label==1)
        print("dice_score:", dice_score)
        # 将dicescore保存
        with open(os.path.join(save_path, "dice_score.txt"), "a") as f:
            f.write(file_name + ": " + str(dice_score) + "\n")
        if dice_score > 0:
            dice_list.append(dice_score)

    # mean dice and std
print("mean dice:", np.mean(dice_list))
print("std dice:", np.std(dice_list))
