# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def build_dataset(is_train, args):
    
    transform = build_transform(is_train, args)
    csv_path = os.path.join(args.data_path, is_train)
    dataset = RETFound_loader(csv_path, transform=transform, data_path=args.data_path)

    return dataset



class RETFound_loader(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, csv_path, transform, data_path):
        'Initialization'
        self.csv_path = csv_path
        print(self.csv_path)
        df = pd.read_csv(self.csv_path)[['Processed_Image','Age']]
        self.list_IDs = df.values.tolist()
        self.transform = transform
        self.data_path = data_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates samples of data'
        
        img_name, age = self.list_IDs[index]
        image = Image.open('{}/Images_preprocessed/{}'.format(self.data_path, img_name))
        image = image.convert("RGB")
        
        if self.transform is not None:
            image_processed = self.transform(image)
 
        return img_name, image_processed.type(torch.FloatTensor), torch.tensor(age).type(torch.float)
        


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train.csv':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
    
