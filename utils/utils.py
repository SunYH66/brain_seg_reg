# --coding:utf-8--
"""This module contains simple helper functions"""
import os
import torch
import shutil
import random
import numpy as np
import pandas as pd
import SimpleITK as sitk

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


def tensor2numpy(input_image, imtype=np.float32):
    """Convert the tensor image array to a numpy image array.

    :param input_image (tensor) -- the input image tensor array
    :param imtype (array type) -- the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data #?????
        else:
            return input_image
        image_numpy = image_tensor[0].squeeze().cpu().float().numpy()
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2im(input_image, imtype=np.uint8):
    """Convert the tensor image array to RGB numpy image type.

    :param input_image (tensor) -- the input image tensor array
    :param imtype (array type) -- the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
        # print('Deleting non-empty fold {} and rebuild it.'.format(path))
        # print('------------------------------------------------------------------')
        # shutil.rmtree(path)
        # os.makedirs(path)

def check_val_folder(cfg):
    if os.path.exists(os.path.join(cfg.checkpoint_root, cfg.name,
                                   '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val')):
        shutil.rmtree(os.path.join(cfg.checkpoint_root, cfg.name,
                                   '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val'))

def save_resluts(image_obj, model, cfg):
    rec_dice = list()
    csv_root = os.path.join(cfg.checkpoint_root, cfg.name,
                            '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val')
    if not os.path.exists(csv_root):
        os.makedirs(csv_root)

    if cfg.phase == 'val':
        for label, img in image_obj.items():
            if img.size()[0] > cfg.batch_size:
                if 'seg_output' in label:
                    for i in range(0, int(img.shape[0] / 3)):
                        dice_csf, dice_gm, dice_wm = multi_layer_dice_coefficient(img[i, ...].cpu().numpy().squeeze(),
                                                                                  model.GT[i, ...].cpu().numpy().squeeze())
                        img_ID = model.get_image_path()
                        img_ID = img_ID['img_path_main'][i].split('/')[-2]
                        rec_dice.append([img_ID, dice_csf, dice_gm, dice_wm])

                    if rec_dice is not None:
                        if not os.path.exists(os.path.join(csv_root, 'rec_dice.csv')):
                            df = pd.DataFrame(rec_dice, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm'])
                            df.to_csv(os.path.join(csv_root, 'rec_dice.csv'), index=False)
                        else:
                            df = pd.DataFrame(rec_dice, columns=None)
                            df.to_csv(os.path.join(csv_root, 'rec_dice.csv'), index=False, header=False, mode='a')
            else: # only seg 06
                if 'seg_output' in label:
                    for i in range(0, img.size()[0]):
                        dice_csf, dice_gm, dice_wm = multi_layer_dice_coefficient(img[i, ...].cpu().numpy().squeeze(),
                                                                                  model.GT[i, ...].cpu().numpy().squeeze())
                        img_ID = model.get_image_path()
                        img_ID = img_ID['img_path_main'][i].split('/')[-2]
                        rec_dice.append([img_ID, dice_csf, dice_gm, dice_wm])

                    if rec_dice is not None:
                        if not os.path.exists(os.path.join(csv_root, 'rec_dice.csv')):
                            df = pd.DataFrame(rec_dice, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm'])
                            df.to_csv(os.path.join(csv_root, 'rec_dice.csv'), index=False)
                        else:
                            df = pd.DataFrame(rec_dice, columns=None)
                            df.to_csv(os.path.join(csv_root, 'rec_dice.csv'), index=False, header=False, mode='a')


def save_best_val(cfg, current_best_dice, model, epoch):
    csv_root = os.path.join(cfg.checkpoint_root, cfg.name,
                            '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val')
    df = pd.read_csv(os.path.join(csv_root, 'rec_dice.csv'))
    dice_csf_total = df['Dice_csf'].sum()
    dice_gm_total = df['Dice_gm'].sum()
    dice_wm_total = df['Dice_wm'].sum()
    dice_average = (dice_csf_total + dice_gm_total + dice_wm_total) / (3 * df.shape[0])

    if current_best_dice <= dice_average:
        current_best_dice = dice_average

        if os.path.exists(os.path.join(cfg.checkpoint_root, cfg.name,
                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val_best')):
            shutil.rmtree(os.path.join(cfg.checkpoint_root, cfg.name,
                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val_best'))

        shutil.copytree(os.path.join(cfg.checkpoint_root, cfg.name,
                            '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val'),
                        os.path.join(cfg.checkpoint_root, cfg.name,
                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val_best'))

        model.save_model(epoch, aff_tag='best_val')

    return current_best_dice

def save_image(image_obj, cfg, img_path, label=''):
    """Save a numpy image to the disk

    Parameters:
        image_obj (dict or np.array)  -- input numpy array
        cfg (easyDict)                -- configure dict
        img_path (dict)               -- dict of image paths
        label (str)                   -- image label (registraion or segmentation)
    """

    if cfg.phase == 'val':
        if isinstance(image_obj, dict): # validation results
            for label, img in image_obj.items():
                if img.size()[0] > cfg.batch_size: # simultaneous seg and reg results (or only reg)
                    if label == 'seg_output':
                        for i in range(0, int(img.size()[0] / 3)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['img_path_main'][i].split('/')[-2],
                                                     img_path['img_path_main'][i].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))
                        for j in range(int(img.size()[0] / 3), 2 * int(img.size()[0] / 3)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['warped_ori_path_help_1'][j - cfg.batch_size].split('/')[-2],
                                                     img_path['warped_ori_path_help_1'][j - cfg.batch_size].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[j, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))
                        for k in range(2 * int(img.size()[0] / 3), 3 * int(img.size()[0] / 3)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['warped_ori_path_help_2'][k - cfg.batch_size * 2].split('/')[-2],
                                                     img_path['warped_ori_path_help_2'][k - cfg.batch_size * 2].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[k, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))
                    elif 'warped' in label:
                        for i in range(0, int(img.size()[0] / 2)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['mov_path_1'][i].split('/')[-2],
                                                     img_path['mov_path_1'][i].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))
                        for j in range(int(img.size()[0] / 2), 2 * int(img.size()[0] / 2)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['mov_path_2'][j - cfg.batch_size].split('/')[-2],
                                                     img_path['mov_path_2'][j - cfg.batch_size].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[j, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))
                    elif 'fix' in label:
                        for i in range(0, int(img.size()[0] / 2)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['fix_path'][i].split('/')[-2],
                                                     img_path['fix_path'][i].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))
                    elif 'flow' in label:
                        for i in range(0, int(img.size()[0] / 2)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['mov_path_1'][i].split('/')[-2],
                                                     img_path['mov_path_1'][i].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))
                        for j in range(int(img.size()[0] / 2), 2 * int(img.size()[0] / 2)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['mov_path_2'][j - cfg.batch_size].split('/')[-2],
                                                     img_path['mov_path_2'][j - cfg.batch_size].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[j, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))

                else: # only seg (or only reg)
                    if 'seg_output' in label:
                        for i in range(0, img.size()[0]):
                            image_numpy = img.detach().cpu().numpy()
                            # TODO: Change image paths for different months
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['img_path_main'][i].split('/')[-2],
                                                     img_path['img_path_main'][i].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))
                    else:
                        for i in range(0, img.size()[0]):
                            image_numpy = img.detach().cpu().numpy()
                            # TODO: Change image paths for different months
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                     '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'val',
                                                     img_path['img_path_main'][i].split('/')[-2],
                                                     img_path['img_path_main'][i].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                            image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                            sitk.WriteImage(image_save,
                                            os.path.join(save_path, label + '.nii.gz'))

    if cfg.phase == 'test':
        if isinstance(image_obj, torch.Tensor): # test results
            if image_obj.size()[0] > cfg.batch_size: # simultaneous seg and reg results (or only reg)
                if label == 'seg_output':
                    for i in range(0, int(image_obj.size()[0] / 3)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), cfg.test_folder,
                                                 img_path['img_path_main'][i].split('/')[-2],
                                                 img_path['img_path_main'][i].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                            os.path.join(save_path, label + '.nii.gz'))
                    for j in range(int(image_obj.size()[0] / 3), 2 * int(image_obj.size()[0] / 3)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), cfg.test_folder,
                                                 img_path['warped_ori_path_help_1'][j - cfg.batch_size].split('/')[-2],
                                                 img_path['warped_ori_path_help_1'][j - cfg.batch_size].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[j, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                            os.path.join(save_path, label + '.nii.gz'))
                    for k in range(2 * int(image_obj.size()[0] / 3), 3 * int(image_obj.size()[0] / 3)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), cfg.test_folder,
                                                 img_path['warped_ori_path_help_2'][k - cfg.batch_size * 2].split('/')[-2],
                                                 img_path['warped_ori_path_help_2'][k - cfg.batch_size * 2].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[k, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                            os.path.join(save_path, label + '.nii.gz'))
                elif 'warped' in label:
                    for i in range(0, int(image_obj.size()[0] / 2)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), cfg.test_folder,
                                                 img_path['mov_path_1'][i].split('/')[-2],
                                                 img_path['mov_path_1'][i].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                                        os.path.join(save_path, label + '.nii.gz'))
                    for j in range(int(image_obj.size()[0] / 2), 2 * int(image_obj.size()[0] / 2)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), cfg.test_folder,
                                                 img_path['mov_path_2'][j - cfg.batch_size].split('/')[-2],
                                                 img_path['mov_path_2'][j - cfg.batch_size].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[j, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                                        os.path.join(save_path, label + '.nii.gz'))
                elif 'fix' in label:
                    for i in range(0, image_obj.size()[0]):
                        image_numpy = image_obj.detach().cpu().numpy()
                        # TODO: Change image paths for different months
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(
                                                     cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder),
                                                 cfg.test_folder,
                                                 img_path['fix_path'][0].split('/')[-2],
                                                 img_path['fix_path'][0].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                                        os.path.join(save_path, label + '.nii.gz'))

            else: # only seg (or only reg)
                if 'warped' in label:
                    for i in range(0, image_obj.size()[0]):
                        image_numpy = image_obj.detach().cpu().numpy()
                        # TODO: Change image paths for different months
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), cfg.test_folder,
                                                 # img_path['fix_path'][i].split('/')[-2],
                                                 img_path['mov_path_1'][i].split('/')[-2],
                                                 img_path['mov_path_1'][i].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                            os.path.join(save_path, label + '.nii.gz'))
                elif 'fix' in label:
                    for i in range(0, image_obj.size()[0]):
                        image_numpy = image_obj.detach().cpu().numpy()
                        # TODO: Change image paths for different months
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), cfg.test_folder,
                                                 img_path['fix_path'][0].split('/')[-2],
                                                 img_path['fix_path'][0].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                            os.path.join(save_path, label + '.nii.gz'))
                elif 'seg_output' in label:
                    for i in range(0, image_obj.size()[0]):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name,
                                                 '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), cfg.test_folder,
                                                 img_path['img_path_main'][0].split('/')[-2],
                                                 img_path['img_path_main'][0].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        image_save = sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0)))
                        image_save.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        sitk.WriteImage(image_save,
                            os.path.join(save_path, label + '.nii.gz'))


def multi_layer_dice_coefficient(source, target, ep=1e-8):
    """TODO: functions to calculate dice coefficient of multi class
    :param source: numpy array (Prediction)
    :param target: numpy array (Ground-Truth)
    :param ep: smooth item
    :return: vector of dice coefficient
    """
    class_num = int(target.max()+1)
    source = source.astype(int)
    source = np.eye(class_num)[source]
    source = source[:,:,:,1:]
    source = source.reshape((-1, class_num-1))

    target = target.astype(int)
    target = np.eye(class_num)[target]
    target = target[:,:,:,1:]
    target = target.reshape(-1, class_num-1)

    intersection = 2 * np.sum(source * target, axis=0) + ep
    union = np.sum(source, axis=0) + np.sum(target, axis=0) + ep

    return intersection / union
