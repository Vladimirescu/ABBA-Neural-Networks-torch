import torch
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
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
    data_dir = "/scratch/aneacsu/rps-cv-images/"

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            image_size=(in_shape[0], in_shape[1]),
            batch_size=64,
            label_mode='int',
            color_mode='rgb',
            validation_split=0.2,
            seed=2,
            subset='training')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        image_size=(in_shape[0], in_shape[1]),
        batch_size=64,
        seed=2,
        label_mode='int',
        color_mode='rgb',
        validation_split=0.2,
        subset='validation')
    
    x_train = None
    y_train = None
    for i, (x, y) in enumerate(train_ds):
        if x_train is None:
            x_train = x
            y_train = y
        else:
            x_train = np.concatenate((x_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    x_test = None
    y_test = None
    for i, (x, y) in enumerate(test_ds):
        if x_test is None:
            x_test = x
            y_test = y
        else:
            x_test = np.concatenate((x_test, x), axis=0)
            y_test = np.concatenate((y_test, y), axis=0)

    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train / 255.0), torch.tensor(y_train, dtype=int)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_test / 255.0), torch.tensor(y_test, dtype=int)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return train_loader, test_loader


def celeba(bs=32):
    in_shape = (128, 128, 3)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "/home/opis/aneacsu/conv2d_abbattack/CelebA_smaller/train/", 
        label_mode="int", 
        color_mode="rgb",
        image_size=(in_shape[0], in_shape[1]),
        batch_size=128)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "/home/opis/aneacsu/conv2d_abbattack/CelebA_smaller/test/", 
        label_mode="int", 
        color_mode="rgb",
        image_size=(in_shape[0], in_shape[1]),
        batch_size=128,
        seed=2) # put a seed in order to be able to align images in the same order when plotting results 

    x_train = None
    y_train = None
    for i, (x, y) in enumerate(train_ds):
        if x_train is None:
            x_train = x
            y_train = y
        else:
            x_train = np.concatenate((x_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    x_test = None
    y_test = None
    for i, (x, y) in enumerate(test_ds):
        if x_test is None:
            x_test = x
            y_test = y
        else:
            x_test = np.concatenate((x_test, x), axis=0)
            y_test = np.concatenate((y_test, y), axis=0)

    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train / 255.0), torch.tensor(y_train, dtype=int)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_test / 255.0), torch.tensor(y_test, dtype=int)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

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