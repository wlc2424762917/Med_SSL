# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from glob import glob

import numpy as np
import torch

import monai
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    Resized,
    Zoomd
)

from monai.config import print_config
from monai.metrics import DiceMetric
# from monai.networks.nets import SwinUNETR

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)


import torch
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
from models.SwinUNet import SwinTransformerSys

IMAGE_FOLDER = os.path.join(".", "data")
# writer will create this folder if it doesn't exist.
OUTPUT_FOLDER = os.path.join(".", "output")

os.environ["MONAI_DATA_DIRECTORY"] = "./BTCV_chckpoints"
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory


def main():
    images = sorted(glob(os.path.join(IMAGE_FOLDER, "case*.nii.gz")))
    train_files = [{"img": img} for img in images]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 1

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["image"],
                spatial_size=[224, 224,-1],
                mode='trilinear'
            ),
            Resized(
                keys=["label"],
                spatial_size=[224, 224,-1],
                mode='nearest'
            ),
            # Zoomd(
            #     keys=["image"],
            #     zoom=[224, 224, 1],
            #     # mode='nearest'
            # ),
            # Zoomd(
            #     keys=["label"],
            #     zoom=[224, 224, 1],
            #     mode='nearest'
            # ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0,
                a_max=255,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(
                keys=["image"],
                spatial_size=[224, 224, -1],
                mode='trilinear'
            ),
            Resized(
                keys=["label"],
                spatial_size=[224, 224, -1],
                mode='nearest'
            ),
            # Zoomd(
            #     keys=["image"],
            #     zoom=[224, 224, 1],
            #     # mode='nearest'
            # ),
            # Zoomd(
            #     keys=["label"],
            #     zoom=[224, 224, 1],
            #     mode='nearest'
            # ),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    data_dir = "./data/synapse/"

    # mini_set
    # split_json = "dataset_mini.json"

    # full
    split_json = "dataset.json"

    datasets = data_dir + split_json
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=16, shuffle=True)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    # for data in train_loader:
    #     import matplotlib.pyplot as plot
    #     print(data['image'].shape)
    #     plot.imshow(data['image'].detach().cpu().numpy()[0,0,:,:,0])
    #     plt.show()
    #     print(data['label'].shape)
    #     plot.imshow(data['label'].detach().cpu().numpy()[0,0,:,:,0])
    #     plt.show()

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    # post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinTransformerSys(img_size=224,
                               patch_size=4,
                               in_chans=1,
                               num_classes=14,
                               embed_dim=96,
                               depths=[2, 2, 6, 2],
                               num_heads=[3, 6, 12, 24],
                               window_size=7,
                               mlp_ratio=4,
                               qkv_bias=True,
                               qk_scale=None,
                               drop_rate=0.,
                               drop_path_rate=0.1,
                               ape=False,
                               patch_norm=True,
                               use_checkpoint=False).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    # optimizer and loss function
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    max_iterations = 30000
    eval_num = 100
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    writer = SummaryWriter()
    epoch_len = len(train_ds) // train_loader.batch_size

    while global_step < max_iterations:
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            x = x[:, :, :, :, 0]  # B, C, H, W
            y = y[:, :, :, :, 0]
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
            writer.add_scalar("train_loss", loss.item(), epoch_len * global_step + step)
            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                model.eval()
                with torch.no_grad():
                    for batch in epoch_iterator_val:
                        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                        val_inputs = val_inputs[:, :, :, :, 0]
                        val_labels = val_labels[:, :, :, :, 0]
                        with torch.cuda.amp.autocast():
                            val_outputs = sliding_window_inference(val_inputs, (224, 224), 1, model)
                        # val_outputs = SimpleInferer(val_inputs, model)
                        val_labels_list = decollate_batch(val_labels)
                        val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                        val_outputs_list = decollate_batch(val_outputs)
                        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                        # epoch_iterator_val.set_description("Validate (%d Steps)" % (global_step))
                    mean_dice_val = dice_metric.aggregate().item()
                    dice_metric.reset()
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(mean_dice_val)
                writer.add_scalar("val_mean_dice", mean_dice_val, global_step + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_inputs, global_step + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, global_step + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, global_step + 1, writer, index=0, tag="output")
                if mean_dice_val > dice_val_best:
                    dice_val_best = mean_dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, mean_dice_val)
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, mean_dice_val
                        )
                    )
            global_step += 1
    writer.close()


if __name__ == "__main__":
    main()