import torchvision
import torch
import torchvision.transforms as tfs
import models
import os
import util
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset

from torch.utils.data import Subset
from torchvision import datasets
from data import DataSet


class SimpleDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format

        if self.transform:
            image = self.transform(image)

        return image, 0, idx  # Returning dummy label and idx


def return_model_loader(args, return_loader=True):
    outs = [args.ncl]*args.hc
    assert args.arch in ['alexnet','resnetv2','resnetv1']
    if args.arch == 'alexnet':
        model = models.__dict__[args.arch](num_classes=outs)
    elif args.arch == 'resnetv2':  # resnet
        model = models.__dict__[args.arch](num_classes=outs, nlayers=50, expansion=1)
    else:
        model = models.__dict__[args.arch](num_classes=outs)
    if not return_loader:
        return model
    train_loader = get_aug_dataloader(image_dir=args.dataset_path,
                                      batch_size=args.batch_size,
                                      num_workers=args.workers,
                                      augs=int(args.augs))

    return model, train_loader

def get_aug_dataloader(image_dir,
                       batch_size=256, image_size=256, crop_size=224,
                       mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                       num_workers=8,
                       augs=2, shuffle=True):

    print(image_dir)
    if image_dir is None:
        return None

    print("imagesize: ", image_size, "cropsize: ", crop_size)
    normalize = tfs.Normalize(mean=mean, std=std)
    if augs == 0:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.CenterCrop(crop_size),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 1:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.CenterCrop(crop_size),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 2:
        _transforms = tfs.Compose([
                                    tfs.Resize(image_size),
                                    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])
    elif augs == 3:
        _transforms = tfs.Compose([
                                    tfs.RandomResizedCrop(crop_size),
                                    tfs.RandomGrayscale(p=0.2),
                                    tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                    tfs.RandomHorizontalFlip(),
                                    tfs.ToTensor(),
                                    normalize
                                ])

    train_data = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=_transforms
    )

    subset_train = Subset(train_data, indices=range(100))


    #dataset = SimpleDataset(image_dir, _transforms)
    dataset = DataSet(subset_train)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return loader




