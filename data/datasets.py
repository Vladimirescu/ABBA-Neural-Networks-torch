import torch
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import medmnist
import numpy as np
import os


global_path = os.getenv("TORCH_DATASETS_PATH")


class Cutout(object):
    """
    From: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py#L5

    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img_np = np.array(img)

        h, w, c = img_np.shape

        mask = np.ones((h, w, c), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        return img_np * mask


def mnist(bs=256, dense=False, bounds=(0, 1)):

    if bounds == (0, 1):
        range_tr = lambda x: x
    elif bounds == (-1, 1):
        range_tr = lambda x: x * 2.0 - 1.0

    if dense:
        transfrms = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           range_tr,
                                           lambda x: torch.flatten(x, 1)
                                       ])
    else:
        transfrms = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           range_tr
                                       ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(global_path, train=True, download=True,
                                   transform=transfrms,
                                   ), batch_size=bs, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(global_path, train=False, download=True,
                                   transform=transfrms),
                                   batch_size=bs, shuffle=False)

    return train_loader, test_loader


def fmnist(bs=256, dense=False, bounds=(0, 1)):

    if bounds == (0, 1):
        range_tr = lambda x: x
    elif bounds == (-1, 1):
        range_tr = lambda x: x * 2.0 - 1.0

    if dense:
        transfrms = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           range_tr,
                                           lambda x: torch.flatten(x, 1)
                                       ])
    else:
        transfrms = torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           range_tr
                                       ])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(global_path, train=True, download=True,
                                   transform=transfrms,
                                   ), batch_size=bs, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(global_path, train=False, download=True,
                                   transform=transfrms),
                                   batch_size=bs, shuffle=False)

    return train_loader, test_loader


def rps(bs=256):
    in_shape = (150, 150, 3)
    data_dir = os.getenv("RPS_DATASET_PATH")

    transform = transforms.Compose([
        transforms.Resize((in_shape[0], in_shape[1])),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    
    full_ds = datasets.ImageFolder(data_dir, transform=transform)
    
    train_size = int(0.8 * len(full_ds)) 
    val_size = len(full_ds) - train_size  
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(2))

    train_loader = DataLoader(
        train_ds, 
        batch_size=bs, 
        shuffle=True, 
        num_workers=4,  
        pin_memory=True  
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=bs, 
        shuffle=False, 
        num_workers=4,  
        pin_memory=True 
    )

    return train_loader, val_loader


def celeba(bs=32):
    in_shape = (128, 128, 3)
    data_dir = os.getenv("CELEBA_DATASET_PATH")
    
    transform = transforms.Compose([
        transforms.Resize((in_shape[0], in_shape[1])), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])
    
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"), 
        transform=transform
    )
    test_ds = datasets.ImageFolder(
        os.path.join(data_dir, "test"), 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=bs, 
        shuffle=True, 
        num_workers=4,  
        pin_memory=True  
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=bs, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True  
    )

    return train_loader, test_loader


def cifar10(bs=128, bounds=(0, 1)):
    if bounds == (0, 1):
        train_transform = transforms.Compose([
            Cutout(n_holes=1, length=16),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        train_transform = transforms.Compose([
            Cutout(n_holes=1, length=16),
            transforms.ToTensor(), lambda x: x * 2.0 - 1.0,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 2.0 - 1.0
        ])

    trainset = torchvision.datasets.CIFAR10(root=global_path, train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=global_path, train=False,
                                           download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


def blood_mnist(bs=128, bounds=(-1, 1), return_names=False):
    if bounds == (0, 1):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif bounds == (-1, 1):
        train_transform = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 2.0 - 1.0,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 2.0 - 1.0
        ])

    ds_train = medmnist.BloodMNIST(split="train", 
                                   transform=train_transform, 
                                   target_transform=lambda x: np.squeeze(x),
                                   download=True, as_rgb=True, size=64)
    ds_val = medmnist.BloodMNIST(split="test", 
                                 transform=test_transform, 
                                 target_transform=lambda x: np.squeeze(x),
                                 download=True, as_rgb=True, size=64)
    
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(ds_val, batch_size=bs, shuffle=True)

    if return_names:
        return trainloader, testloader, ds_train.info["label"]
    else:
        return trainloader, testloader


def pnreumo_mnist(bs=128, bounds=(-1, 1), return_names=False, binary=False):
    if bounds == (0, 1):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    elif bounds == (-1, 1):
        train_transform = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 2.0 - 1.0,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 2.0 - 1.0
        ])

    if not binary:
        target_transform = lambda x: np.squeeze(x)
    else:
        target_transform = lambda x: x.astype(int)

    ds_train = medmnist.PneumoniaMNIST(split="train", 
                                   transform=train_transform, 
                                   target_transform=target_transform,
                                   download=True, as_rgb=False, size=64)
    ds_val = medmnist.PneumoniaMNIST(split="test", 
                                 transform=test_transform, 
                                 target_transform=target_transform,
                                 download=True, as_rgb=False, size=64)
    
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(ds_val, batch_size=bs, shuffle=True)

    if return_names:
        return trainloader, testloader, ds_train.info["label"]
    else:
        return trainloader, testloader


def derma_mnist(bs=128, bounds=(-1, 1), return_names=False):
    if bounds == (0, 1):
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif bounds == (-1, 1):
        train_transform = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 2.0 - 1.0,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(), lambda x: x * 2.0 - 1.0
        ])

    ds_train = medmnist.DermaMNIST(split="train", 
                                   transform=train_transform, 
                                   target_transform=lambda x: np.squeeze(x),
                                   download=True, as_rgb=True, size=64)
    ds_val = medmnist.DermaMNIST(split="test", 
                                 transform=test_transform, 
                                 target_transform=lambda x: np.squeeze(x),
                                 download=True, as_rgb=True, size=64)
    
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=bs, shuffle=True)
    testloader = torch.utils.data.DataLoader(ds_val, batch_size=bs, shuffle=True)

    if return_names:
        return trainloader, testloader, ds_train.info["label"]
    else:
        return trainloader, testloader