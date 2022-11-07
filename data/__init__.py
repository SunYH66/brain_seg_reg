# --coding:utf-8--
import torch
import random
import importlib
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

def find_dataset_using_name(dataset_name):
    """
    TODO:Import the module "data/[dataset_name]_dataset.py"
    :param dataset_name:
    :return:
    importlib reference: https://hatboy.github.io/2017/12/21/Python-importlib%E8%AE%B2%E8%A7%A3/
    """
    dataset_filename = 'data.' + dataset_name.lower() + '_dataset'
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():

        if name.lower() == target_dataset_name.lower():
            dataset = cls
    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name"
                                  " that matches %s in lower case." %(dataset_filename, target_dataset_name))
    return dataset


class CustomDatasetDataLoader:
    def __init__(self, opt):
        dataset_class = find_dataset_using_name(opt.datamode)
        self.dataset = dataset_class(opt)
        if opt.phase == 'train':
            print('dataset [%s] was created' % type(self.dataset).__name__)
            train_size = int(0.9 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=2,
                drop_last=True
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=2,
                drop_last=True
            )
        elif opt.phase == 'test':
            print('dataset [%s] was created' % type(self.dataset).__name__)
            self.test_loader = DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=2,
                drop_last=True
            )

    def __len__(self):
        return len(self.dataset)

def create_dataset(opt):
    """TODO: Create specific dataset and dataloader."""
    dataloader = CustomDatasetDataLoader(opt)
    return dataloader


def random_sample(img, patch_size):
    """
    Functions to extract patches from whole image randomly
    :param img: medical image (array data)
    :param patch_size: crop size of extracted patch
    :return: random cropped image patch with exact size
    """
    # transer tensor datatype to ndarray
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # get maximum index and random get selected data
    max_pos = img.shape[2:]

    # select center point from height, width and depth
    select_pos = [random.choice(range(max_pos[0])), random.choice(range(max_pos[1])),
                  random.choice(range(max_pos[2]))]

    gap = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]
    start_pos = [select_pos[0] - gap[0], select_pos[1] - gap[1], select_pos[2] - gap[2]]
    # determine whether start position out of range
    start_pos = [max(0, f) for f in start_pos]
    # determine whether end position out of range
    for idx in range(len(start_pos)):
        if start_pos[idx] + patch_size[idx] > max_pos[idx]:
            start_pos[idx] = max_pos[idx] - patch_size[idx]

    # cropped image by start position and patch size
    img = img[:, :, start_pos[0]:start_pos[0] + patch_size[0], start_pos[1]:start_pos[1] + patch_size[1],
          start_pos[2]:start_pos[2] + patch_size[2]]

    return img


def mask_sample(img, patch_size):
    """
    Functions to extract patches from whole image determined by corresponding mask
    :param img: medical image (array data)
    :param patch_size: crop size of extracted patch
    :return: cropped image by mask and exact size
    """
    # transer tensor datatype to ndarray
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    max_pos = img.shape[2:]

    # get potential exists center point within mask area
    index = np.where(img[3:, ...] != 0)[2:]
    index = list(map(list, zip(*index)))

    # select center point randomly from mask region
    select_pos = random.choice(index)
    gap = [patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]

    # create start position according to the center point and patch size and determine whether the start position out of range
    start_pos = [select_pos[0] - gap[0], select_pos[1] - gap[1], select_pos[2] - gap[2]]
    start_pos = [max(0, f) for f in start_pos]

    # create end position according to start point and patch size, adn determine whether the end point out of range
    for idx in range(len(start_pos)):
        if start_pos[idx] + patch_size[idx] > max_pos[idx]:
            start_pos[idx] = max_pos[idx] - patch_size[idx]

    # cropped image patch
    img = img[:, :, start_pos[0]:start_pos[0] + patch_size[0], start_pos[1]:start_pos[1] + patch_size[1],
          start_pos[2]:start_pos[2] + patch_size[2]]

    return img

if __name__ == '__main__':
    a = mask_sample(torch.rand((6, 1, 256, 256, 256)), patch_size=[128, 128, 96])
    print('aaa')