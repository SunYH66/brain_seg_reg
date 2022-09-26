# --coding:utf-8--
import os
import random

import pandas as pd
import numpy as np
import SimpleITK as sitk
from data import random_sample, mask_sample
from config.config import cfg
from monai import transforms


def get_transform(opt, convert=True):
    """
    conduct data augmentation
    :param opt:
    :param convert:
    :return:
    transforms.Lambda(lambda img: __crop(img, pos, size))
    reference: https://www.cnblogs.com/wanghui-garcia/p/11248416.html
    """
    if opt.phase == 'test':
        opt.preprocess = ''
    transform_list = []
    if 'resize' in opt.preprocess:
        transform_list.append(transforms.Resize(opt.load_size))
    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandSpatialCrop(opt.crop_size, random_center=True, random_size=False))
    if 'rotate' in opt.preprocess:
        transform_list.append(transforms.RandRotate(range_x=opt.angle, range_y=opt.angle, range_z=opt.angle, keep_size=True))
    if 'resample' in opt.preprocess:
        transform_list.append(transforms.Resample())
    if 'flip' in opt.preprocess:
        transform_list.append(transforms.RandFlip(prob=0.5))
    if convert:
        transform_list.append((transforms.ToTensor()))
    return transforms.Compose(transform_list)

def normalize_z_score(data, clip=True):
    """
    funtions to normalize data to standard distribution using (data - data.mean()) / data.std()
    or (data - data.min()) / (data.max() - data.min())
    :param data: numpy array
    :param clip: whether using upper and lower clip
    :return: normalized data by using z-score
    """
    if clip:
        bounds = np.percentile(data, q=[0, 100])
        data[data <= bounds[0]] = bounds[0]
        data[data >= bounds[1]] = bounds[1]

    return (data - data.min()) / (data.max() - data.min())

class IBISDataset:
    """"""
    def __init__(self, opt):
        self.opt = opt
        self.data_root = opt.data_root

        file_list = pd.read_csv(opt.csv_root)
        if opt.phase == 'train':
            file_list = file_list.loc[file_list['Fold'] != opt.fold, 'ID']
        elif opt.phase == 'test':
            file_list = file_list.loc[file_list['Fold'] == opt.fold, 'ID']

        self.img_list = file_list.values.tolist()


    def __getitem__(self, idx):
        self.transforms = get_transform(self.opt)
        #======================================simultaneous seg and reg===========================================
        # Read images
        img_path_06 = os.path.join(self.data_root, self.img_list[idx], '06mo', 'intensity_mask_out.nii.gz')
        img_path_12 = os.path.join(self.data_root, self.img_list[idx], '12mo', 'warped_ori_to_06_affine.nii.gz')
        img_path_24 = os.path.join(self.data_root, self.img_list[idx], '24mo', 'warped_ori_to_06_affine.nii.gz')

        seg_path_06 = os.path.join(self.data_root, self.img_list[idx], '06mo', 'segment.nii.gz')
        seg_path_12 = os.path.join(self.data_root, self.img_list[idx], '12mo', 'warped_seg_to_06_affine.nii.gz')
        seg_path_24 = os.path.join(self.data_root, self.img_list[idx], '24mo', 'warped_seg_to_06_affine.nii.gz')

        img_06 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_06)).transpose((2, 1, 0)))
        img_12 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_12)).transpose((2, 1, 0)))
        img_24 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_24)).transpose((2, 1, 0)))

        seg_06 = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_06)).transpose((2, 1, 0))
        seg_12 = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_12)).transpose((2, 1, 0))
        seg_24 = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_24)).transpose((2, 1, 0))

        # get transformed images (transform input: channel, height, width, deepth)
        img = np.stack((img_06, img_12, img_24, seg_06, seg_12, seg_24), axis=0)
        img = img[:, np.newaxis, ...]
        # img = self.transforms(img)[:, np.newaxis, ...]
        if self.opt.phase == 'train' or self.opt.phase == 'val':
            if random.random() > 0.2:
                img = mask_sample(img, self.opt.crop_size)
            else:
                img = random_sample(img, self.opt.crop_size)

        return {'img_06': img[0, ...], 'img_12': img[1, ...], 'img_24': img[2, ...], 'seg_06': img[3, ...],
                'seg_12': img[4, ...], 'seg_24': img[5, ...], 'img_06_path': os.path.dirname(img_path_06),
                'img_12_path': os.path.dirname(img_path_12), 'img_24_path': os.path.dirname(img_path_24)}

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    ibis_dataset = IBISDataset(cfg)
    ibis_dataset.__getitem__(0)
