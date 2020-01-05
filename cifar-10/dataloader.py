import torch.nn as nn
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import os
from draw_err import ErrorPlot
import argparse
from sklearn.metrics import accuracy_score

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



def my_data_loader(batch_size):
    global_mean = [0.4914, 0.4822, 0.4465]
    global_std = [0.2023, 0.1994, 0.2010]
    transforml = transforms.Compose([
        transforms.Resize((40, 40)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=global_mean,
                             std=global_std)
    ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=global_mean,std=global_std)
                                         ])

    train_ds = torchvision.datasets.ImageFolder(root='./train_images',transform=transforml)
    ## 需要获取默认的 classes 与 index 的对应关系

    valid_ds = torchvision.datasets.ImageFolder(root='./valid_images',transform=transforml)
    
    test_ds = ImageFolderWithPaths(root='./test',transform=transform_test)
    # test_ds = torchvision.datasets.ImageFolder(root='./test',transform=transform_test)
    train_data_loader = torch.utils.data.DataLoader(train_ds,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=4)

    valid_data_loader = torch.utils.data.DataLoader(valid_ds,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test_ds,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    # num_workers=4
                                                    )
    return train_data_loader,valid_data_loader,test_data_loader,train_ds.class_to_idx
