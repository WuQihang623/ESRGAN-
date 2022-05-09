import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import random
import numpy as np

# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
#
#
# def denormalize(tensors):
#     """ Denormalizes image tensors using mean and std """
#     for c in range(3):
#         tensors[:,c] = tensors[:,c] * std[c] + mean[c]
#     return torch.clamp(tensors, 0, 1)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_shape// 4, hr_shape // 4), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_shape, hr_shape), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    @staticmethod
    def random_horizontal_flip(lr,hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[2])
            hr = torch.flip(hr, dims=[2])
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[1])
            hr = torch.flip(hr, dims=[1])
        return lr, hr

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        img_lr,img_hr = self.random_vertical_flip(img_lr,img_hr)
        img_lr,img_hr = self.random_horizontal_flip(img_lr,img_hr)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)