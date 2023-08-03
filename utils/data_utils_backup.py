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

import json
import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

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


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training"):

    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    # train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
    datasets = data_dir + datalist_json
    if args.seg:
        train_files = load_decathlon_datalist(datasets, True, "training")
        validation_files = load_decathlon_datalist(datasets, True, "validation")
        train_transform = Compose(
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

        val_transform = Compose(
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
                )
            ]
        )

        test_transform = Compose(
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
                )
            ]
        )
    else:
        train_files = load_decathlon_datalist(datasets, True, "training")
        validation_files = load_decathlon_datalist(datasets, True, "validation")
        train_transform = Compose(
            [
                LoadImaged(keys=["image", "label_seg", "label_sr"], ensure_channel_first=True),
                ScaleIntensityRanged(
                    keys=["image", "label_sr"],
                    a_min=-175,
                    a_max=250,
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
                    keys=["label_seg"],
                    spatial_size=[448, 448, -1],
                    mode='nearest'
                ),
                Resized(
                    keys=["label_sr"],
                    spatial_size=[448, 448, -1],
                    mode='trilinear'
                ),
                RandFlipd(
                    keys=["image", "label_seg", "label_sr"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label_seg", "label_sr"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandRotate90d(
                    keys=["image", "label_seg", "label_sr"],
                    prob=0.10,
                    max_k=3,
                )
            ]
        )

        val_transform = Compose(
            [
                LoadImaged(keys=["image", "label_seg", "label_sr"], ensure_channel_first=True),
                ScaleIntensityRanged(
                    keys=["image", "label_sr"],
                    a_min=-175,
                    a_max=250,
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
                    keys=["label_seg"],
                    spatial_size=[448, 448, -1],
                    mode='nearest'
                ),
                Resized(
                    keys=["label_sr"],
                    spatial_size=[448, 448, -1],
                    mode='trilinear'
                )
            ]
        )

        test_transform = Compose(
            [
                LoadImaged(keys=["image", "label_seg", "label_sr"], ensure_channel_first=True),
                ScaleIntensityRanged(
                    keys=["image", "label_sr"],
                    a_min=-175,
                    a_max=250,
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
                    keys=["label_seg"],
                    spatial_size=[448, 448, -1],
                    mode='nearest'
                ),
                Resized(
                    keys=["label_sr"],
                    spatial_size=[448, 448, -1],
                    mode='trilinear'
                )
            ]
        )

    if args.test_mode:

        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)

        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
