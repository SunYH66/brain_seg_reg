# --coding:utf-8--
"""This module contains simple helper functions"""
import os
import torch
import shutil
import numpy as np
import SimpleITK as sitk

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
        print('Deleting non-empty fold {} and rebuild it.'.format(path))
        print('------------------------------------------------------------------')
        shutil.rmtree(path)
        os.makedirs(path)


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
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name, 'val',
                                                     img_path['img_06_path'][i].split('/')[-2],
                                                     img_path['img_06_path'][i].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            sitk.WriteImage(sitk.GetImageFromArray(
                                            image_numpy[i, ...].squeeze().transpose((2, 1, 0))),
                                            os.path.join(save_path, label + '.nii.gz'))
                        for j in range(int(img.size()[0] / 3), 2 * int(img.size()[0] / 3)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name, 'val',
                                                     img_path['img_12_path'][j - cfg.batch_size].split('/')[-2],
                                                     img_path['img_12_path'][j - cfg.batch_size].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            sitk.WriteImage(sitk.GetImageFromArray(
                                            image_numpy[j, ...].squeeze().transpose((2, 1, 0))),
                                            os.path.join(save_path, label + '.nii.gz'))
                        for k in range(2 * int(img.size()[0] / 3), 3 * int(img.size()[0] / 3)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name, 'val',
                                                     img_path['img_24_path'][k - cfg.batch_size * 2].split('/')[-2],
                                                     img_path['img_24_path'][k - cfg.batch_size * 2].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            sitk.WriteImage(sitk.GetImageFromArray(
                                            image_numpy[k, ...].squeeze().transpose((2, 1, 0))),
                                            os.path.join(save_path, label + '.nii.gz'))
                    elif label == 'warped_seg' or label == 'warped_ori':
                        for i in range(0, int(img.size()[0] / 2)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name, 'val',
                                                     img_path['img_12_path'][i].split('/')[-2],
                                                     img_path['img_12_path'][i].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            sitk.WriteImage(sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0))),
                                            os.path.join(save_path, label + '.nii.gz'))
                        for j in range(int(img.size()[0] / 2), 2 * int(img.size()[0] / 2)):
                            image_numpy = img.detach().cpu().numpy()
                            save_path = os.path.join(cfg.checkpoint_root, cfg.name, 'val',
                                                     img_path['img_24_path'][j - cfg.batch_size].split('/')[-2],
                                                     img_path['img_24_path'][j - cfg.batch_size].split('/')[-1])
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            sitk.WriteImage(sitk.GetImageFromArray(image_numpy[j, ...].squeeze().transpose((2, 1, 0))),
                                            os.path.join(save_path, label + '.nii.gz'))
                else: # only seg (or only reg)
                    for i in range(0, img.size()[0]):
                        image_numpy = img.detach().cpu().numpy()
                        # TODO: Change image paths for different months
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name, 'val',
                                                 img_path['img_12_path'][i].split('/')[-2],
                                                 img_path['img_12_path'][i].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        sitk.WriteImage(sitk.GetImageFromArray(image_numpy[i, ...].squeeze().transpose((2, 1, 0))),
                                        os.path.join(save_path, label + '.nii.gz'))

    if cfg.phase == 'test':
        if isinstance(image_obj, torch.Tensor): # test results
            if image_obj.size()[0] > cfg.batch_size: # simultaneous seg and reg results (or only reg)
                if label == 'seg_output':
                    for i in range(0, int(image_obj.size()[0] / 3)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.test_folder,
                                                 img_path['img_06_path'][i].split('/')[-2],
                                                 img_path['img_06_path'][i].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        sitk.WriteImage(sitk.GetImageFromArray(
                            image_numpy[i:i+1, ...].squeeze().transpose((2, 1, 0))),
                            os.path.join(save_path, label + '.nii.gz'))
                    for j in range(int(image_obj.size()[0] / 3), 2 * int(image_obj.size()[0] / 3)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.test_folder,
                                                 img_path['img_12_path'][j - cfg.batch_size].split('/')[-2],
                                                 img_path['img_12_path'][j - cfg.batch_size].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        sitk.WriteImage(sitk.GetImageFromArray(
                            image_numpy[j:j+1, ...].squeeze().transpose((2, 1, 0))),
                            os.path.join(save_path, label + '.nii.gz'))
                    for k in range(2 * int(image_obj.size()[0] / 3), 3 * int(image_obj.size()[0] / 3)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.test_folder,
                                                 img_path['img_24_path'][k - cfg.batch_size * 2].split('/')[-2],
                                                 img_path['img_24_path'][k - cfg.batch_size * 2].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        sitk.WriteImage(sitk.GetImageFromArray(
                            image_numpy[k:k+1, ...].squeeze().transpose((2, 1, 0))),
                            os.path.join(save_path, label + '.nii.gz'))
                elif label == 'warped_seg' or label == 'warped_ori':
                    for i in range(0, int(image_obj.size()[0] / 2)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.test_folder,
                                                 img_path['img_12_path'][i].split('/')[-2],
                                                 img_path['img_12_path'][i].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        sitk.WriteImage(sitk.GetImageFromArray(image_numpy[i:i+1, ...].squeeze().transpose((2, 1, 0))),
                                        os.path.join(save_path, label + '.nii.gz'))
                    for j in range(int(image_obj.size()[0] / 2), 2 * int(image_obj.size()[0] / 2)):
                        image_numpy = image_obj.detach().cpu().numpy()
                        save_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.test_folder,
                                                 img_path['img_24_path'][j - cfg.batch_size].split('/')[-2],
                                                 img_path['img_24_path'][j - cfg.batch_size].split('/')[-1])
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        sitk.WriteImage(sitk.GetImageFromArray(image_numpy[j:j+1, ...].squeeze().transpose((2, 1, 0))),
                                        os.path.join(save_path, label + '.nii.gz'))
            else: # only seg (or only reg)
                for i in range(0, image_obj.size()[0]):
                    image_numpy = image_obj.detach().cpu().numpy()
                    # TODO: Change image paths for different months
                    save_path = os.path.join(cfg.checkpoint_root, cfg.name, cfg.test_folder,
                                             img_path['img_12_path'][i].split('/')[-2],
                                             img_path['img_12_path'][i].split('/')[-1])
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    sitk.WriteImage(sitk.GetImageFromArray(
                        image_numpy[i:i+1, ...].squeeze().transpose((2, 1, 0))),
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
