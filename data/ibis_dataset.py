# --coding:utf-8--
import os
import random

import pandas as pd
import numpy as np
import SimpleITK as sitk
from brain_reg_seg.data import random_sample, mask_sample
from brain_reg_seg.config.config import cfg
from monai import transforms

def get_transform(opt, convert=True):
    """
    conduct data augmentation
    :param opt:
    :param convert:
    :return:
    transforms.Lambda(lambda img: __crop(img, pos, size))
    reference: https://www.cnblogs.com/wanghui-garcia/p/11248416.htmltest
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
        self.mo_list = opt.mo_list

        file_list = pd.read_csv(opt.csv_root)
        if 'test' not in opt.test_folder:
            file_list = file_list.loc[file_list['Fold'] != opt.fold, 'ID']
        elif 'test' in opt.test_folder:
            file_list = file_list.loc[file_list['Fold'] == opt.fold, 'ID']

        self.img_list = file_list.values.tolist()


    def __getitem__(self, idx):
        if self.opt.model == 'reg':
            # Read images
            img_path_fix   = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[0]), 'intensity.nii.gz')
            img_path_mov_1 = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[1]), 'warped_ori_to_{:02d}.nii.gz'.format(self.mo_list[0]))
            img_path_mov_2 = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[2]), 'warped_ori_to_{:02d}.nii.gz'.format(self.mo_list[0]))

            seg_path_fix   = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[0]), 'tissue.nii.gz')
            seg_path_mov_1 = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[1]), 'warped_seg_to_{:02d}.nii.gz'.format(self.mo_list[0]))
            seg_path_mov_2 = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[2]), 'warped_seg_to_{:02d}.nii.gz'.format(self.mo_list[0]))

            img_fix   = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_fix)).transpose((2, 1, 0)))
            img_mov_1 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_mov_1)).transpose((2, 1, 0)))
            img_mov_2 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_mov_2)).transpose((2, 1, 0)))

            seg_fix   = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_fix)).transpose((2, 1, 0))
            seg_mov_1 = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_mov_1)).transpose((2, 1, 0))
            seg_mov_2 = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_mov_2)).transpose((2, 1, 0))


            # get transformed images (transform input: channel, height, width, deepth)
            img = np.stack((img_fix, img_mov_1, img_mov_2, seg_fix, seg_mov_1, seg_mov_2), axis=0)
            img = img[:, np.newaxis, ...]

            if self.opt.phase == 'train':
                if random.random() > 0.2:
                    img = mask_sample(img, self.opt.crop_size)
                else:
                    img = random_sample(img, self.opt.crop_size)
            img_dict = {'img_fix': img[0, ...], 'img_mov_1': img[1, ...], 'img_mov_2': img[2, ...],
                        'seg_fix': img[3, ...], 'seg_mov_1': img[4, ...], 'seg_mov_2': img[5, ...],
                        'img_path_fix': os.path.dirname(img_path_fix),
                        'img_path_mov_1': os.path.dirname(img_path_mov_1),
                        'img_path_mov_2': os.path.dirname(img_path_mov_2)}
            """
            # read images for reg
            warped_ori_06_12_path = os.path.join(self.data_root, self.img_list[idx], '06mo', 'warped_ori_to_12.nii.gz')
            warped_ori_06_24_path = os.path.join(self.data_root, self.img_list[idx], '06mo', 'warped_ori_to_24.nii.gz')
            warped_ori_12_06_path = os.path.join(self.data_root, self.img_list[idx], '12mo', 'warped_ori_to_06.nii.gz')
            warped_ori_12_24_path = os.path.join(self.data_root, self.img_list[idx], '12mo', 'warped_ori_to_24.nii.gz')
            warped_ori_24_06_path = os.path.join(self.data_root, self.img_list[idx], '24mo', 'warped_ori_to_06.nii.gz')
            warped_ori_24_12_path = os.path.join(self.data_root, self.img_list[idx], '24mo', 'warped_ori_to_12.nii.gz')

            warped_seg_06_12_path = os.path.join(self.data_root, self.img_list[idx], '06mo', 'warped_seg_to_12.nii.gz')
            warped_seg_06_24_path = os.path.join(self.data_root, self.img_list[idx], '06mo', 'warped_seg_to_24.nii.gz')
            warped_seg_12_06_path = os.path.join(self.data_root, self.img_list[idx], '12mo', 'warped_seg_to_06.nii.gz')
            warped_seg_12_24_path = os.path.join(self.data_root, self.img_list[idx], '12mo', 'warped_seg_to_24.nii.gz')
            warped_seg_24_06_path = os.path.join(self.data_root, self.img_list[idx], '24mo', 'warped_seg_to_06.nii.gz')
            warped_seg_24_12_path = os.path.join(self.data_root, self.img_list[idx], '24mo', 'warped_seg_to_12.nii.gz')

            warped_ori_06_12 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_06_12_path)).transpose((2, 1, 0)))
            warped_ori_06_24 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_06_24_path)).transpose((2, 1, 0)))
            warped_ori_12_06 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_12_06_path)).transpose((2, 1, 0)))
            warped_ori_12_24 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_12_24_path)).transpose((2, 1, 0)))
            warped_ori_24_06 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_24_06_path)).transpose((2, 1, 0)))
            warped_ori_24_12 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_24_12_path)).transpose((2, 1, 0)))

            warped_seg_06_12 = sitk.GetArrayFromImage(sitk.ReadImage(warped_seg_06_12_path)).transpose((2, 1, 0))
            warped_seg_06_24 = sitk.GetArrayFromImage(sitk.ReadImage(warped_seg_06_24_path)).transpose((2, 1, 0))
            warped_seg_12_06 = sitk.GetArrayFromImage(sitk.ReadImage(warped_seg_12_06_path)).transpose((2, 1, 0))
            warped_seg_12_24 = sitk.GetArrayFromImage(sitk.ReadImage(warped_seg_12_24_path)).transpose((2, 1, 0))
            warped_seg_24_06 = sitk.GetArrayFromImage(sitk.ReadImage(warped_seg_24_06_path)).transpose((2, 1, 0))
            warped_seg_24_12 = sitk.GetArrayFromImage(sitk.ReadImage(warped_seg_24_12_path)).transpose((2, 1, 0))

            # get transformed images (transform input: channel, height, width, deepth)
            img = np.stack((img_06, img_12, img_24, seg_06, seg_12, seg_24,
                            warped_ori_06_12, warped_ori_06_24, warped_ori_12_06,
                            warped_ori_12_24, warped_ori_24_06, warped_ori_24_12,
                            warped_seg_06_12, warped_seg_06_24, warped_seg_12_06,
                            warped_seg_12_24, warped_seg_24_06, warped_seg_24_12), axis=0)
            img = img[:, np.newaxis, ...]

            if self.opt.phase == 'train':
                if random.random() > 0.2:
                    img = mask_sample(img, self.opt.crop_size)
                else:
                    img = random_sample(img, self.opt.crop_size)
            img_dict = {'img_06': img[0, ...], 'img_12': img[1, ...], 'img_24': img[2, ...],
                        'seg_06': img[3, ...], 'seg_12': img[4, ...], 'seg_24': img[5, ...],
                        'warped_ori_06_12': img[6, ...], 'warped_ori_06_24': img[7, ...], 'warped_ori_12_06': img[8, ...],
                        'warped_ori_12_24': img[9, ...], 'warped_ori_24_06': img[10, ...],'warped_ori_24_12': img[11, ...],
                        'warped_seg_06_12': img[12, ...], 'warped_seg_06_24': img[13, ...], 'warped_seg_12_06': img[14, ...],
                        'warped_seg_12_24': img[15, ...], 'warped_seg_24_06': img[16, ...], 'warped_seg_24_12': img[17, ...],
                        'img_path_06': os.path.dirname(img_path_06), 'img_path_12': os.path.dirname(img_path_12), 'img_path_24': os.path.dirname(img_path_24)}
            """
        elif self.opt.model == 'proj':
            img_path_main = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[0]), 'intensity.nii.gz')

            seg_path_main = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[0]), 'tissue.nii.gz')

            img_main = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_main)).transpose((2, 1, 0)))

            seg_main = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_main)).transpose((2, 1, 0))

            warped_ori_path_help_1 = os.path.join(self.data_root, self.img_list[idx],
                                              '{:02d}mo'.format(self.mo_list[1]), 'warped_ori_to_{:02d}_syn_ori.nii.gz'.format(self.mo_list[0]))
            warped_ori_path_help_2 = os.path.join(self.data_root, self.img_list[idx],
                                              '{:02d}mo'.format(self.mo_list[2]), 'warped_ori_to_{:02d}_syn_ori.nii.gz'.format(self.mo_list[0]))
            warped_seg_path_help_1 = os.path.join(self.data_root, self.img_list[idx],
                                              '{:02d}mo'.format(self.mo_list[1]), 'warped_seg_to_{:02d}_syn_ori.nii.gz'.format(self.mo_list[0]))
            warped_seg_path_help_2 = os.path.join(self.data_root, self.img_list[idx],
                                              '{:02d}mo'.format(self.mo_list[2]), 'warped_seg_to_{:02d}_syn_ori.nii.gz'.format(self.mo_list[0]))

            warped_ori_help_1 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_path_help_1)).transpose((2, 1, 0)))
            warped_ori_help_2 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_path_help_2)).transpose((2, 1, 0)))
            warped_seg_help_1 = sitk.GetArrayFromImage(sitk.ReadImage(warped_seg_path_help_1)).transpose((2, 1, 0))
            warped_seg_help_2 = sitk.GetArrayFromImage(sitk.ReadImage(warped_seg_path_help_2)).transpose((2, 1, 0))

            # get transformed images (transform input: channel, height, width, deepth)
            img = np.stack((img_main, seg_main,
                            warped_ori_help_1, warped_ori_help_2,
                            warped_seg_help_1, warped_seg_help_2), axis=0)
            img = img[:, np.newaxis, ...]

            if self.opt.phase == 'train':
                if random.random() > 0.2:
                    img = mask_sample(img, self.opt.crop_size)
                else:
                    img = random_sample(img, self.opt.crop_size)

            img_dict = {'img_main': img[0, ...], 'seg_main': img[1, ...],
                        'warped_ori_help_1': img[2, ...], 'warped_ori_help_2': img[3, ...],
                        'warped_seg_help_1': img[4, ...], 'warped_seg_help_2': img[5, ...],
                        'img_path_main': os.path.dirname(img_path_main),
                        'warped_ori_path_help_1': os.path.dirname(warped_ori_path_help_1),
                        'warped_ori_path_help_2': os.path.dirname(warped_ori_path_help_2)}

        elif self.opt.model == 'seg':
            # Read images
            img_path = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[0]), 'intensity.nii.gz')

            seg_path = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[0]), 'tissue.nii.gz')

            img = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).transpose((2, 1, 0)))

            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).transpose((2, 1, 0))


            # get transformed images (transform input: channel, height, width, deepth)
            img = np.stack((img, seg), axis=0)
            img = img[:, np.newaxis, ...]
            # img = self.transforms(img)[:, np.newaxis, ...]
            if self.opt.phase == 'train':
                if random.random() > 0.2:
                    img = mask_sample(img, self.opt.crop_size)
                else:
                    img = random_sample(img, self.opt.crop_size)

            img_dict = {'img': img[0, ...],
                        'seg': img[1, ...],
                        'img_path': os.path.dirname(img_path)}

        elif self.opt.model == 'segwarp':
            img_path = os.path.join(self.data_root, self.img_list[idx], #cfg.checkpoint_root, cfg.name, cfg.reg_folder, cfg.test_folder,
                                    '{:02d}mo'.format(self.mo_list[0]), 'warped_ori_to_{:02d}_syn_ori.nii.gz'.format(self.mo_list[1]))

            seg_path = os.path.join(self.data_root, self.img_list[idx],
                                    '{:02d}mo'.format(self.mo_list[0]), 'warped_seg_to_{:02d}_syn_ori.nii.gz'.format(self.mo_list[1]))

            img = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).transpose((2, 1, 0)))
            # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).transpose((2, 1, 0))

            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).transpose((2, 1, 0))

            # get transformed images (transform input: channel, height, width, deepth)
            img = np.stack((img, seg), axis=0)
            img = img[:, np.newaxis, ...]

            if self.opt.phase == 'train':
                if random.random() > 0.2:
                    img = mask_sample(img, self.opt.crop_size)
                else:
                    img = random_sample(img, self.opt.crop_size)

            img_dict = {'img': img[0, ...],
                        'seg': img[1, ...],
                        'img_path': os.path.dirname(img_path)}

            """
            # Read images
            img_path_06 = os.path.join(self.data_root, self.img_list[idx], '06mo', 'intensity_mask_out.nii.gz')
            img_path_12 = os.path.join(self.data_root, self.img_list[idx], '12mo', 'intensity_mask_out.nii.gz')
            img_path_24 = os.path.join(self.data_root, self.img_list[idx], '24mo', 'intensity_mask_out.nii.gz')

            seg_path_06 = os.path.join(self.data_root, self.img_list[idx], '06mo', 'segment.nii.gz')
            seg_path_12 = os.path.join(self.data_root, self.img_list[idx], '12mo', 'segment.nii.gz')
            seg_path_24 = os.path.join(self.data_root, self.img_list[idx], '24mo', 'segment.nii.gz')

            img_06 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_06)).transpose((2, 1, 0)))
            img_12 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_12)).transpose((2, 1, 0)))
            img_24 = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_path_24)).transpose((2, 1, 0)))

            seg_06 = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_06)).transpose((2, 1, 0))
            seg_12 = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_12)).transpose((2, 1, 0))
            seg_24 = sitk.GetArrayFromImage(sitk.ReadImage(seg_path_24)).transpose((2, 1, 0))

            # read images for seg
            warped_ori_06_12_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.reg_folder, cfg.test_folder, self.img_list[idx], '06mo', 'warped_ori_to_12.nii.gz')
            warped_ori_06_24_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.reg_folder, cfg.test_folder, self.img_list[idx], '06mo', 'warped_ori_to_24.nii.gz')
            warped_ori_12_06_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.reg_folder, cfg.test_folder, self.img_list[idx], '12mo', 'warped_ori_to_06.nii.gz')
            warped_ori_12_24_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.reg_folder, cfg.test_folder, self.img_list[idx], '12mo', 'warped_ori_to_24.nii.gz')
            warped_ori_24_06_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.reg_folder, cfg.test_folder, self.img_list[idx], '24mo', 'warped_ori_to_06.nii.gz')
            warped_ori_24_12_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.reg_folder, cfg.test_folder, self.img_list[idx], '24mo', 'warped_ori_to_12.nii.gz')

            warped_ori_06_12 = sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_06_12_path)).transpose((2, 1, 0))
            warped_ori_06_24 = sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_06_24_path)).transpose((2, 1, 0))
            warped_ori_12_06 = sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_12_06_path)).transpose((2, 1, 0))
            warped_ori_12_24 = sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_12_24_path)).transpose((2, 1, 0))
            warped_ori_24_06 = sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_24_06_path)).transpose((2, 1, 0))
            warped_ori_24_12 = sitk.GetArrayFromImage(sitk.ReadImage(warped_ori_24_12_path)).transpose((2, 1, 0))

            # get transformed images (transform input: channel, height, width, deepth)
            img = np.stack((img_06, img_12, img_24, seg_06, seg_12, seg_24,
                            warped_ori_06_12, warped_ori_06_24, warped_ori_12_06,
                            warped_ori_12_24, warped_ori_24_06, warped_ori_24_12), axis=0)
            img = img[:, np.newaxis, ...]

            if self.opt.phase == 'train':
                if random.random() > 0.2:
                    img = mask_sample(img, self.opt.crop_size)
                else:
                    img = random_sample(img, self.opt.crop_size)

            img_dict = {'img_06': img[0, ...], 'img_12': img[1, ...], 'img_24': img[2, ...],
                        'seg_06': img[3, ...], 'seg_12': img[4, ...], 'seg_24': img[5, ...],
                        'warped_ori_06_12': img[6, ...], 'warped_ori_06_24': img[7, ...],
                        'warped_ori_12_06': img[8, ...], 'warped_ori_12_24': img[9, ...],
                        'warped_ori_24_06': img[10, ...], 'warped_ori_24_12': img[11, ...],
                        'img_path_06': os.path.dirname(img_path_06), 'img_path_12': os.path.dirname(img_path_12), 'img_path_24': os.path.dirname(img_path_24)}
            """
        elif self.opt.model == 'cyclegan':
            # Read images
            img_domainA_path = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[0]), 'intensity.nii.gz')

            img_domainB_path = os.path.join(self.data_root, self.img_list[idx], '{:02d}mo'.format(self.mo_list[2]), 'warped_ori_to_{:02d}_syn_ori.nii.gz'.format(self.mo_list[0]))

            img_domainA = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_domainA_path)).transpose((2, 1, 0)))

            img_domainB = normalize_z_score(sitk.GetArrayFromImage(sitk.ReadImage(img_domainB_path)).transpose((2, 1, 0)))


            # get transformed images (transform input: channel, height, width, deepth)
            img = np.stack((img_domainA, img_domainB), axis=0)
            img = img[:, np.newaxis, ...]
            # img = self.transforms(img)[:, np.newaxis, ...]
            if self.opt.phase == 'train':
                if random.random() > 0.2:
                    img = mask_sample(img, self.opt.crop_size)
                else:
                    img = random_sample(img, self.opt.crop_size)

            img_dict = {'img_domainA': img[0, ...],
                        'img_domainB': img[1, ...],
                        'img_path': os.path.dirname(img_domainA_path)}
        else:
            raise NotImplementedError('training model must be one of the following: [reg | seg | proj | segwarp | cyclegan]')

        return img_dict

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    ibis_dataset = IBISDataset(cfg)
    ibis_dataset.__getitem__(0)
