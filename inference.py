# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader
from utils.utils import dice, psnr, ssim
from PIL import Image
import statistics

import monai
from models import create_model
import cv2

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
# parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
# parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
# parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
# parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
# parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
# parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--in_shape_x", default=224, type=int, help="roi size in x direction")
parser.add_argument("--in_shape_y", default=224, type=int, help="roi size in y direction")
parser.add_argument("--out_shape_x", default=448, type=int, help="roi size in x direction")
parser.add_argument("--out_shape_y", default=448, type=int, help="roi size in y direction")

parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

parser.add_argument("--seg", action="store_true", default=False, help="if it is segmentation task")
parser.add_argument("--sr", action="store_true", default=False, help="if it is super resolution task")
parser.add_argument("--seg_sr", action="store_true", help="if it is multitask seg and sr task")
parser.add_argument("--sr_loss", default="1*MSE+1*L1", type=str, help="loss function for super resolution task")
parser.add_argument("--model", default="unet", type=str, help="choose the model")
parser.add_argument("--srloss2segloss", default=0.1, type=float, help="sr loss weight to seg loss weight ratio")
parser.add_argument("--segloss2srloss", default=1.0, type=float, help="seg loss weight to sr loss weight ratio")
parser.add_argument("--scale", default=2, type=int, help="upsample scale")
parser.add_argument("--val_metric", default="dice", type=str, help="validation metric to decide best model")
parser.add_argument("--test_mode", default=True, type=bool, help="test mode")


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./inference_outputs/" + args.exp_name
    seg_output_directory = os.path.join(output_directory, "seg")
    sr_output_directory = os.path.join(output_directory, "sr")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if args.seg_sr:
        if not os.path.exists(seg_output_directory):
            os.makedirs(seg_output_directory)
        if not os.path.exists(sr_output_directory):
            os.makedirs(sr_output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    # --------- init model -------- #
    model = create_model(args)
    # ---------------------------------------- #

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    print("Loaded pretrained weights from {}".format(pretrained_pth))
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        psnr_list_case = []
        ssim_list_case = []
        for i, batch in enumerate(val_loader):
            print(i)
            if args.seg:
                val_inputs, val_labels = (batch["image"][:, :, :, :].cuda(), batch["label_seg"][:, :, :, :].cuda())
                original_affine = batch["image_meta_dict"]["affine"][0].numpy()
                _, _, h, w = val_labels.shape
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                print("Inference on case {}".format(img_name))
                val_outputs = model(val_inputs)
                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                val_labels = val_labels.cpu().numpy()[0, 0, :, :]
                h_out, w_out = val_outputs.shape
                if h_out != h or w_out != w:
                    val_outputs = cv2.resize(val_outputs, (w, h), interpolation=cv2.INTER_NEAREST)
                dice_list_sub = []
                for i in range(1, args.out_channels):
                    organ_Dice = dice(val_outputs == i, val_labels == i)
                    dice_list_sub.append(organ_Dice)
                mean_dice = np.mean(dice_list_sub)
                print("Mean Organ Dice: {}".format(mean_dice))
                dice_list_case.append(mean_dice)
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
                )
                # write the dice score of each organ to a txt file
                with open(os.path.join(output_directory, "dice_score.txt"), "a") as f:
                    f.write(img_name + "\n")
                    for i in range(1, args.out_channels):
                        f.write("Organ {}: {}\n".format(i, dice_list_sub[i - 1]))
                    f.write("Mean Organ Dice: {}\n".format(mean_dice))
                    f.write("\n")

            elif args.sr:
                val_inputs, val_labels = (batch["image"][:, :, :, :, 0].cuda(), batch["label_sr"][:, :, :, :, 0].cuda())
                original_affine = batch["image_meta_dict"]["affine"][0].numpy()
                _, _, h, w = val_labels.shape
                target_shape = (h, w)
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                print("Inference on case {}".format(img_name))
                val_outputs = model(val_inputs)
                print(val_outputs.shape)
                val_outputs = val_outputs[0, 0, :, :]
                val_labels = val_labels.cpu().numpy()
                psnr_score = psnr(val_outputs, val_labels, 1)
                ssim_score = ssim(val_outputs, val_labels, 1)
                psnr_list_case.append(psnr_score)
                ssim_list_case.append(ssim_score)
                print("PSNR: {}".format(psnr_score))
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                    os.path.join(output_directory, img_name)
                )
                # write the psnr and ssim score of each organ to a txt file
                with open(os.path.join(output_directory, "psnr_ssim_score.txt"), "a") as f:
                    f.write(img_name + "\n")
                    f.write("PSNR: {}\n".format(psnr_score))
                    f.write("SSIM: {}\n".format(ssim_score))
                    f.write("\n")

            elif args.seg_sr:
                val_inputs, seg_labels, sr_labels = (batch["image"][:, :, :, :].cuda(), batch["label_seg"][:, :, :, :].cuda(), batch["label_sr"][:, :, :, :].cuda())
                original_affine = batch["image_meta_dict"]["affine"][0].numpy()
                _, _, h, w = seg_labels.shape
                target_shape = (h, w)
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                print("Inference on case {}".format(img_name))
                seg_outputs, sr_outputs =model(val_inputs)
                seg_outputs = torch.softmax(seg_outputs, 1).cpu().numpy()
                seg_outputs = np.argmax(seg_outputs, axis=1).astype(np.uint8)[0]
                seg_labels = seg_labels.cpu().numpy()[0, 0, :, :]
                h_out, w_out = seg_outputs.shape
                if h_out != h or w_out != w:
                    seg_outputs = cv2.resize(seg_outputs, (w, h), interpolation=cv2.INTER_NEAREST)
                dice_list_sub = []
                for i in range(1, args.out_channels):
                    organ_Dice = dice(seg_outputs == i, seg_labels == i)
                    dice_list_sub.append(organ_Dice)
                mean_dice = np.mean(dice_list_sub)
                print("Mean Class Dice: {}".format(mean_dice))
                if mean_dice != 0:
                    dice_list_case.append(mean_dice)

                psnr_score = psnr(sr_outputs, sr_labels, 1)
                ssim_score = ssim(sr_outputs, sr_labels, 1)
                psnr_list_case.append(psnr_score)
                ssim_list_case.append(ssim_score)
                print("PSNR: {}".format(psnr))

                nib.save(
                    nib.Nifti1Image(seg_outputs.astype(np.uint8), original_affine), os.path.join(seg_output_directory, img_name)
                )
                nib.save(
                    nib.Nifti1Image(sr_outputs.astype(np.float32), original_affine),
                    os.path.join(sr_output_directory, img_name)
                )

                # write the dice score of each organ to a txt file
                with open(os.path.join(seg_output_directory, "dice_score.txt"), "a") as f:
                    f.write(img_name + "\n")
                    for i in range(1, args.out_channels):
                        f.write("class {}: {}\n".format(i, dice_list_sub[i - 1]))
                    f.write("Mean  Dice: {}\n".format(mean_dice))
                    f.write("\n")

        if args.seg:
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
            print("overall Dice variance: {}".format(np.var(dice_list_case)))
        elif args.sr:
            print("Overall PSNR: {}".format(np.mean(psnr_list_case)))
            print("Overall SSIM: {}".format(np.mean(ssim_list_case)))
        elif args.seg_sr:
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
            print("Overall PSNR: {}".format(np.mean(psnr_list_case)))
            print("Overall SSIM: {}".format(np.mean(ssim_list_case)))
            print("overall Dice variance: {}".format(np.var(dice_list_case)))
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
