# --coding:utf-8--
import os
import glob
import shutil

import cv2
import imageio
from PIL import Image
import ants
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.ndimage import zoom
from brain_reg_seg.config.config import cfg
from sklearn.model_selection import KFold
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

def resample_image_spacing(new_sapcing=(2,2,2)):
    """TODO: Resample image"""
    img_path = '/data/infant_brain_seg_reg/copyDataOut/NDAR_V06-1_IBIS0015/intensity.nii.gz'
    img = sitk.ReadImage(img_path)
    spacing = img.GetSpacing()
    print('Old image shape:', img.GetSize())
    resample_ratio = np.array(spacing) / np.array(new_sapcing)
    resample_img = zoom(sitk.GetArrayFromImage(img), resample_ratio)
    print('New image shape:', resample_img.shape)

def resample_image_size(new_size=(192,224,192)):
    """TODO: Resample image"""
    img_path = '/data/brain_reg_seg/IBIS0015/06mo/intensity.nii.gz'
    img = sitk.ReadImage(img_path)
    print('Old image shape:', img.GetSize())
    resample_ratio = np.array(new_size) / np.array(img.GetSize())
    resample_img = zoom(sitk.GetArrayFromImage(img), resample_ratio)
    new_spacing = np.array(img.GetSize()) * np.array(img.GetSpacing()) / np.array(new_size)
    print('New image shape:', resample_img.shape)
    resample_img = sitk.GetImageFromArray(resample_img)
    resample_img.SetSpacing(new_spacing)

    sitk.WriteImage(resample_img, img_path)

def padding_zero():
    img_path = sorted(glob.glob('/data/dataset/brain_reg_seg/*/*/dk-struct.nii.gz'))
    # seg_path = sorted(glob.glob('/data/brain_reg_seg/*/*/tissue.nii.gz'))
    for i in range(len(img_path)):
        img = np.zeros((192, 224, 192))
        # seg = np.zeros((192, 224, 192))

        img[5:-5, 3:-3, 5:-5] = sitk.GetArrayFromImage(sitk.ReadImage(img_path[i])).transpose((2, 1, 0))
        # seg[5:-5, 3:-3, 5:-5] = sitk.GetArrayFromImage(sitk.ReadImage(seg_path[i])).transpose((2, 1, 0))
        img = sitk.GetImageFromArray(img.transpose((2,1,0)))
        img.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        img.SetOrigin((-90, 126, -72))
        # seg = sitk.ReadImage(seg_path[i])
        # seg.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(img, img_path[i])
        # sitk.WriteImage(seg, seg_path[i])
        print('aaa')

def set_direction():
    img_path = sorted(glob.glob('/data/dataset/brain_reg_seg/*/*/intensity.nii.gz'))
    for i in range(len(img_path)):
        img = sitk.ReadImage(img_path[i])
        img.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        img.SetOrigin((-90, 126, -72))
        sitk.WriteImage(img, img_path[i])

def draw_cubic():
    import turtle

    # turtle.goto(200, 0)
    # turtle.goto(200, 200)
    # turtle.goto(0, 200)
    # turtle.goto(0, 0)
    #
    # turtle.penup()
    # turtle.goto(100, 100)
    # turtle.pendown()
    #
    # turtle.goto(100, -100)
    # turtle.goto(-100, -100)
    # turtle.goto(-100, 100)
    # turtle.goto(100, 100)
    #
    # turtle.goto(200, 200)
    #
    # turtle.penup()
    # turtle.goto(100, -100)
    # turtle.pendown()
    # turtle.goto(200, 0)
    #
    # turtle.penup()
    # turtle.goto(100, -100)
    # turtle.pendown()
    # turtle.goto(200, 0)

    # turtle.done()

    import turtle as t
    p=200
    n=50

    t.goto(p, 0)
    t.goto(p+n, n)
    t.goto(n, n)
    t.goto(0, 0)
    t.goto(0, p)
    t.goto(n, p+n)
    t.goto(p+n, p+n)
    t.goto(p+n, n)
    t.goto(n, n)
    t.goto(n, p+n)
    t.goto(0, p)
    t.goto(p, p)
    t.goto(p+n, p+n)
    t.goto(p, p)
    t.goto(p, 0)

    p=50
    n=25
    pos_1 = 80
    pos_2 = 80
    t.penup()
    t.goto(pos_1, pos_2)
    t.pendown()
    t.goto(p+pos_1, pos_2)
    t.goto(p+n+pos_1, n+pos_2)
    t.goto(n+pos_1, n+pos_2)
    t.goto(pos_1, pos_2)
    t.goto(pos_1, p+pos_2)
    t.goto(n+pos_1, p+n+pos_2)
    t.goto(p+n+pos_1, p+n+pos_2)
    t.goto(p+n+pos_1, n+pos_2)
    t.goto(n+pos_1, n+pos_2)
    t.goto(n+pos_1, p+n+pos_2)
    t.goto(pos_1, p+pos_2)
    t.goto(p+pos_1, p+pos_2)
    t.goto(p+n+pos_1, p+n+pos_2)
    t.goto(p+pos_1, p+pos_2)
    t.goto(p+pos_1, pos_2)
    turtle.hideturtle()
    t.done()

def delete_min_vale():
    pass

def download_csv():
    import requests

    download_url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
    target_csv_path = "/data/dataset/nba_all_elo.csv"

    response = requests.get(download_url)
    response.raise_for_status()  # Check that the request was successful
    with open(target_csv_path, "wb") as f:
        f.write(response.content)
    print("Download ready.")

def settle_dataset_dirs():
    """TODO:Settle dataset dirs into good format."""
    """
    # Move dirs to upper folder and delete useless folder
    file_list = sorted(glob.glob('/data/infant_brain_seg_reg/ND*'))
    for file in file_list:
        shutil.move(file, '/data/infant_brain_seg_reg')
    os.rmdir('/data/infant_brain_seg_reg/copyDataOut')
    print(len(file_list))
    """
    file_list = sorted(glob.glob('/data/brain_reg_seg/*/*/mr_brain'))
    print(len(file_list))
    print(file_list[0])
    for file in file_list:
        os.removedirs(file)

    # # Build new folder
    # file_list = sorted(glob.glob('/data/infant_brain_seg_reg/*'))
    # valued_file_num = 0
    # for path in file_list:
    #     print(path[-8:])
    #     if os.path.isdir(path):
    #         if 'V06' in path:
    #             month06_path = os.path.join('/data/infant_brain_seg_reg/', path[-8:], '06mo')
    #             shutil.copytree(path, month06_path)
    #             valued_file_num += 1
    #         elif 'V12' in path:
    #             month12_path = os.path.join('/data/infant_brain_seg_reg/', path[-8:], '12mo')
    #             shutil.copytree(path, month12_path)
    #             valued_file_num += 1
    #         elif 'V24' in path:
    #             month24_path = os.path.join('/data/infant_brain_seg_reg/', path[-8:], '24mo')
    #             shutil.copytree(path, month24_path)
    #             valued_file_num += 1
    #         else:
    #             raise ValueError('Unexpected file path...')
    #         print(valued_file_num)

    """
    file_id_all_month = sorted(glob.glob('/data/infant_brain_seg_reg/*_' + file_id[-8:]))
    if len(file_id_all_month) == 3:
        month06_path = os.path.join('/data/infant_brain_seg_reg/', file_id[-8:], '06mo')
        month12_path = os.path.join('/data/infant_brain_seg_reg/', file_id[-8:], '12mo')
        month24_path = os.path.join('/data/infant_brain_seg_reg/', file_id[-8:], '24mo')
        # os.makedirs(os.path.join('/data/infant_brain_seg_reg/', file_id[-8:]), exist_ok=True)
        shutil.copytree(file_id_all_month[0], month06_path)
        shutil.copytree(file_id_all_month[1], month12_path)
        shutil.copytree(file_id_all_month[2], month24_path)
        valued_file_num += 1
        print(valued_file_num)
    """

def write_06_12_mo_to_csv():
    """TODO: find img ID that contains both 06 and 12 month images and write them to a csv file."""
    file_list = sorted(glob.glob('/data/infant_brain_seg_reg/*'))

    dir_name_list = []
    for path in file_list:
        month_folder = os.listdir(path)
        print(month_folder)
        if '06mo' in month_folder and '12mo' in month_folder:
            dir_name = os.path.basename(path)
            dir_name_list.append(dir_name)
    print('all folder:', len(file_list))
    print('06_12_folder:', len(dir_name_list))
    df = pd.DataFrame(dir_name_list, columns=['ID'])
    df.to_csv('../csvfile/filelist_06_12.csv')


def write_all_3_mo_to_csv():
    """TODO: find img ID that contains 06, 12 and 24 month images and write them to a csv file."""
    file_list = sorted(glob.glob('/data/infant_brain_seg_reg/*'))

    dir_name_list = []
    for path in file_list:
        month_folder = os.listdir(path)
        print(month_folder)
        if '06mo' in month_folder and '12mo' in month_folder and '24mo' in month_folder:
            dir_name = os.path.basename(path)
            dir_name_list.append(dir_name)
    print('all folder:', len(file_list))
    print('06_12_24_folder:', len(dir_name_list))
    df = pd.DataFrame(dir_name_list, columns=['ID'])
    df.to_csv('../csvfile/filelist_06_12_24.csv')

def write_any_2_mo_to_csv():
    """TODO: find img ID that contains equal or more than two month images (any two) and write them to a csv file."""
    file_list = sorted(glob.glob('/data/infant_brain_seg_reg/*'))

    dir_name_list = []
    for path in file_list:
        month_folder = os.listdir(path)
        print(month_folder)
        if len(month_folder) >= 2:
            dir_name = os.path.basename(path)
            dir_name_list.append(dir_name)
    print('all folder:', len(file_list))
    print('any_2_mo_folder:', len(dir_name_list))
    df = pd.DataFrame(dir_name_list, columns=['ID'])
    df.to_csv('../csvfile/filelist_any_2_mo.csv')


def write_06_any_mo_to_csv():
    """TODO: find img ID that contains 06, (12 "or" 24) month images and write them to a csv file."""
    file_list = sorted(glob.glob('/data/infant_brain_seg_reg/*'))

    dir_name_list = []
    for path in file_list:
        month_folder = os.listdir(path)
        print(month_folder)
        if '06mo' in month_folder and len(month_folder) >= 2:
            dir_name = os.path.basename(path)
            dir_name_list.append(dir_name)
    print('all folder:', len(file_list))
    print('06_any_folder:', len(dir_name_list))
    df = pd.DataFrame(dir_name_list, columns=['ID'])
    df.to_csv('../csvfile/filelist_06_any.csv')

def jiameng_csv():
    """TODO: find img ID that contains both 06 and 12 month images and write them to a csv file."""
    file_list = sorted(glob.glob('/public_bme/home/liujm/Project/InfantSyn/data/*'))

    dir_name_list = []
    for i in file_list:
        for j in file_list:
            dir_name_list.append([os.path.basename(i), os.path.basename(j)])
    df = pd.DataFrame(dir_name_list, columns=['ID_1', 'ID_2'])
    df.to_csv('../csvfile/jiameng.csv')

def jiameng_csv_2():
    """TODO: find img ID that contains both 06 and 12 month images and write them to a csv file."""
    file_list = sorted(glob.glob('/public_bme/home/liujm/Project/InfantSyn/data/*'))

    dir_name_list = []
    for path in file_list:
        month_folder = os.listdir(path)
        print(month_folder)
        if '6mo' in month_folder and len(month_folder) >= 2:
            dir_name = os.path.basename(path)
            dir_name_list.append(dir_name)
    print('all folder:', len(file_list))
    print('06_any_folder:', len(dir_name_list))
    df = pd.DataFrame(dir_name_list, columns=['ID'])
    df.to_csv('../csvfile/jiameng2.csv')

def write_valued_fileIDs_to_csv():
    """
    TODO:Select and write correct file_name to the csv file.
    If the fold doesn't contain images of 6 month, it will not be selected.
    """
    dir_list = sorted(glob.glob('/data/infant_brain_seg_reg/*'))
    dir_name_list = []
    for dir_path in dir_list:
        dir_name = os.path.basename(dir_path)
        lower_dir =  os.listdir(dir_path)
        if not '06mo' in lower_dir:
            print(dir_name)
        else:
            dir_name_list.append(dir_name)
    df = pd.DataFrame(dir_name_list, columns=['ID'])
    df.to_csv('../csvfile/filelist.csv')


def mask_out_brain():
    """Mask out the brain."""
    intensity_path = sorted(glob.glob('/data/brain_reg_seg/*/*/T1.nii.gz'))
    # segment_path = sorted(glob.glob('/data/brain_reg_seg/*/*/tissue.nii.gz'))
    strip_path = sorted(glob.glob('/data/brain_reg_seg/*/*/skull-strip.nii.gz'))
    for i in range(len(intensity_path)):
        intensity = sitk.GetArrayFromImage(sitk.ReadImage(intensity_path[i]))
        # segment = sitk.GetArrayFromImage(sitk.ReadImage(segment_path[i]))
        strip = sitk.GetArrayFromImage(sitk.ReadImage(strip_path[i]))
        intensity_mask_out = intensity * strip
        # segment_mask_out = segment * strip
        intensity_img = sitk.GetImageFromArray(intensity_mask_out)
        intensity_img.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        sitk.WriteImage(intensity_img,
                        os.path.join(os.path.dirname(intensity_path[i]), 'intensity.nii.gz'))
        # sitk.WriteImage(sitk.GetImageFromArray(segment_mask_out),
        #                 os.path.join(os.path.dirname(intensity_path[i]), 'segment_mask_out.nii.gz'))


def rotate_image():
    seg_joint_path = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022_12_02/joint_help_seg/test/IBIS1818/06mo/seg_output.nii.gz'))
    seg_single_path = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022_12_02/single_month_seg/test/IBIS1818/06mo/seg_output.nii.gz'))
    seg_joint = sitk.ReadImage(seg_joint_path[0])
    seg_single = sitk.ReadImage(seg_single_path[0])
    seg_joint.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    seg_single.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(seg_joint, seg_joint_path[0])
    sitk.WriteImage(seg_single, seg_single_path[0])

def generate_slice_image():
    ori_path = sorted(glob.glob('/data/brain_reg_seg/IBIS0407/06mo/intensity.nii.gz'))
    GT_path = sorted(glob.glob('/data/brain_reg_seg/IBIS0407/06mo/tissue.nii.gz'))
    seg_output_path_06 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/our/IBIS0407/06mo/seg_output.nii.gz'))
    seg_output_path_12 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/trans/IBIS0407/06mo/seg_output.nii.gz'))
    seg_output_path_24 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/unet/IBIS0407/06mo/seg_output.nii.gz'))
    seg_output_path_48 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/hyper/IBIS0407/06mo/seg_output.nii.gz'))
    seg_output_path_96 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/dense/IBIS0407/06mo/seg_output.nii.gz'))
    ori = sitk.GetArrayFromImage(sitk.ReadImage(ori_path[0])).transpose((2,1,0))[..., 105]
    ori = ori / np.max(ori) * 255
    GT = sitk.GetArrayFromImage(sitk.ReadImage(GT_path[0])).transpose((2,1,0))[..., 105]
    GT = GT / np.max(GT) * 255
    seg_output_06 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_06[0])).transpose((2,1,0))[...,105]
    seg_output_06 = seg_output_06 / np.max(seg_output_06) * 255
    seg_output_12 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_12[0])).transpose((2,1,0))[...,105]
    seg_output_12 = seg_output_12 / np.max(seg_output_12) * 255
    seg_output_24 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_24[0])).transpose((2,1,0))[...,105]
    seg_output_24 = seg_output_24 / np.max(seg_output_24) * 255
    seg_output_48 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_48[0])).transpose((2,1,0))[...,105]
    seg_output_48 = seg_output_48 / np.max(seg_output_48) * 255
    seg_output_96 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_96[0])).transpose((2,1,0))[...,105]
    seg_output_96 = seg_output_96 / np.max(seg_output_96) * 255

    Image.fromarray(ori).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/our/IBIS0407/06mo/intensity_06_0407_106.png')
    Image.fromarray(GT).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/our/IBIS0407/06mo/tissue_06_0407_106.png')
    Image.fromarray(seg_output_06).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/our/IBIS0407/06mo/our_06_0407_106.png')
    Image.fromarray(seg_output_12).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/trans/IBIS0407/06mo/trans_06_0407_106.png')
    Image.fromarray(seg_output_24).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/unet/IBIS0407/06mo/unet_06_0407_106.png')
    Image.fromarray(seg_output_48).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/hyper/IBIS0407/06mo/hyper_06_0407_106.png')
    Image.fromarray(seg_output_96).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_3/dense/IBIS0407/06mo/dense_06_0407_106.png')

    # ori_path = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/06mo/intensity.nii.gz'))
    # seg_output_path_06 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/06mo/tissue.nii.gz'))
    # seg_output_path_12 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/12mo/warped_ori_to_06_syn_ori.nii.gz'))
    # seg_output_path_24 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/24mo/warped_ori_to_06_syn_ori.nii.gz'))
    # ori = sitk.GetArrayFromImage(sitk.ReadImage(ori_path[0])).transpose((2,1,0))[...,66]
    # ori = ori / np.max(ori) * 255
    # seg_output_06 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_06[0])).transpose((2,1,0))[...,66]
    # seg_output_06 = seg_output_06 / np.max(seg_output_06) * 255
    # seg_output_12 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_12[0])).transpose((2,1,0))[...,66]
    # seg_output_12 = seg_output_12 / np.max(seg_output_12) * 255
    # seg_output_24 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_24[0])).transpose((2,1,0))[...,66]
    # seg_output_24 = seg_output_24 / np.max(seg_output_24) * 255
    # Image.fromarray(ori).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/06mo/06_intensity_66.png')
    # Image.fromarray(seg_output_06).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/06mo/06_tissue_66.png')
    # Image.fromarray(seg_output_12).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/12mo/warped_ori_to_06_66_syn.png')
    # Image.fromarray(seg_output_24).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/24mo/warped_ori_to_06_66_syn.png')
    #
    #
    # ori_path = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/06mo/intensity.nii.gz'))
    # seg_output_path_06 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/06mo/tissue.nii.gz'))
    # seg_output_path_12 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/12mo/warped_ori_to_06.nii.gz'))
    # seg_output_path_24 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/24mo/warped_ori_to_06.nii.gz'))
    # ori = sitk.GetArrayFromImage(sitk.ReadImage(ori_path[0])).transpose((2,1,0))[...,66]
    # ori = ori / np.max(ori) * 255
    # seg_output_06 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_06[0])).transpose((2,1,0))[...,66]
    # seg_output_06 = seg_output_06 / np.max(seg_output_06) * 255
    # seg_output_12 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_12[0])).transpose((2,1,0))[...,66]
    # seg_output_12 = seg_output_12 / np.max(seg_output_12) * 255
    # seg_output_24 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_24[0])).transpose((2,1,0))[...,66]
    # seg_output_24 = seg_output_24 / np.max(seg_output_24) * 255
    # Image.fromarray(ori).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/06mo/06_intensity_66.png')
    # Image.fromarray(seg_output_06).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/06mo/06_tissue_66.png')
    # Image.fromarray(seg_output_12).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/12mo/warped_ori_to_06_66.png')
    # Image.fromarray(seg_output_24).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/brain_reg_ants/IBIS1350/24mo/warped_ori_to_06_66.png')


def mask_out_brain2():
    """Mask out the brain."""
    intensity_path = sorted(glob.glob('/data/infant_brain_seg_reg/*/*/intensity.nii.gz'))
    segment_path = sorted(glob.glob('/data/infant_brain_seg_reg/*/*/segment.nii.gz'))
    strip_path = sorted(glob.glob('/data/infant_brain_seg_reg/*/*/strip.nii.gz'))
    for i in range(len(intensity_path)):
        intensity = sitk.GetArrayFromImage(sitk.ReadImage(intensity_path[i]))
        segment = sitk.GetArrayFromImage(sitk.ReadImage(segment_path[i]))
        strip = sitk.GetArrayFromImage(sitk.ReadImage(strip_path[i]))
        intensity_mask_out = intensity * strip
        segment_mask_out = segment * strip
        sitk.WriteImage(sitk.GetImageFromArray(intensity_mask_out),
                        os.path.join(os.path.dirname(intensity_path[i]), 'intensity_mask_out.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(segment_mask_out),
                        os.path.join(os.path.dirname(intensity_path[i]), 'segment_mask_out.nii.gz'))

def move_files():
    """Copy files"""
    target_path = '/data/ndar-original/copyDataOut/'
    source_path = '/data/V12_V24_tissue_segment/'
    target_dirs = sorted(os.listdir(target_path))
    source_dirs = sorted(os.listdir(source_path))
    print(len(target_dirs))
    print(len(source_dirs))
    count = 0
    for i in range(len(source_dirs)):
        for j in range(len(target_dirs)):
            if source_dirs[i] == target_dirs[j]:
                print(source_dirs[i])
                print(target_dirs[j])
                count = count + 1
                target_file = os.path.join(target_path, target_dirs[j], 'segment.nii.gz')
                source_file = os.path.join(source_path, source_dirs[i], 'segment.nii.gz')
                shutil.copyfile(source_file, target_file)


def generate_csv():
    """
    TODO: Write file paths into a csv file in a K-fold cross validation mode.
    """
    # General setting
    csv_path = ''
    img_path = sorted(glob.glob('/data/infant_brain_seg_reg/*'))
    seg_path = sorted(glob.glob('/data/infant_brain_seg_reg/*'))
    file_index = np.arange(len(img_path))
    kf = KFold(n_splits=5) # number of k-folds
    fold = 1
    # if os.path.exists(csv_path):
    #     os.remove(csv_path)
    # os.mknod(csv_path)

    # Split path list into train and test set in K-fold format and write them into csv file.
    # csv header: Fold, Flage, img_path, seg_path
    for train_index, test_index in kf.split(file_index):
        print(fold, train_index)
        print(fold, test_index)
        fold += 1
    #     img_path_fold_train = []
    #     seg_path_fold_train = []
    #     img_path_fold_test = []
    #     seg_path_fold_test = []
    #     train_flag = []
    #     test_flag = []
    #     for i in train_index:
    #         img_path_fold_train.append(img_path[i])
    #         seg_path_fold_train.append(seg_path[i])
    #         train_flag.append('train')
    #     for j in test_index:
    #         img_path_fold_test.append(img_path[j])
    #         seg_path_fold_test.append(seg_path[j])
    #         test_flag.append('test')
    #     fold_num = [fold for i in range(len(img_path_fold_train + img_path_fold_test))]
    #     dataframe = pd.DataFrame({
    #         'Fold': fold_num,
    #         'Flag': train_flag + test_flag,
    #         'img_path': img_path_fold_train + img_path_fold_test,
    #         'seg_path': seg_path_fold_train + seg_path_fold_test
    #     })
    #     with open('', "a") as csv:
    #         dataframe.to_csv(csv, index=False)
    #     fold += 1
    #
    # # Check if the csv file has the format and content we wanted
    # df = pd.read_csv(csv_path)
    # df_train = df[(df['Fold'] == '1') & (df['Flag'] == 'train')]

    """
    row_all = df['Fold'].tolist()
    ind = np.where(np.array([x == '1' for x in row_all]))
    row_fold = df.iloc[ind]
    print(row_fold.values)
    print(type(row_fold))
    for k in range(len(ind)):
        pass
    print('aaa')
    """

def settle_fold_ID():
    fold_ID = []
    df = pd.read_csv('../csvfile/jiameng.csv', index_col=False)
    for i in range(df.shape[0]):
        if (i + 1) % 5 == 0:
            fold_ID.append(5)
        else:
            fold_ID.append((i + 1) % 5)
    print(fold_ID)

    df.insert(df.shape[1], 'Fold', fold_ID)
    # df = pd.DataFrame(fold_ID, columns=['Fold'])
    # df.to_csv('../csvfile/filelist.csv')
    # df.reset_index(drop=True)
    # df.to_csv('../csvfile/filelist.csv')
    # del df['Unnamed: 0']
    # df.reset_index(drop=True)
    # df.reset_index(drop=True, inplace=True)

    df.to_csv('../csvfile/jiameng.csv', index=False)


def remove_files():
    data_root = '/data/infant_brain_seg_reg'
    for fold in range(1, 6):
        file_list = pd.read_csv('../csvfile/filelist_06_12_24.csv')
        file_list = file_list.loc[file_list['Fold'] == fold, 'ID']
        file_list = file_list.values.tolist()
        for i in range(len(file_list)):
            mo_list = os.listdir(os.path.join(data_root, file_list[i]))
            print(mo_list)
            for mo in mo_list[1:]:
                print(mo)
                remove_seg_path = os.path.join(data_root, file_list[i], mo, 'warped_seg_to_06.nii.gz')
                remove_ori_path = os.path.join(data_root, file_list[i], mo, 'warped_ori_to_06.nii.gz')
                os.remove(remove_ori_path)
                os.remove(remove_seg_path)


def remove_files_2():
    data_root = '/data/infant_brain_seg_reg'
    ID_list = os.listdir('/data/infant_brain_seg_reg')
    for i in range(len(ID_list)):
        mo_list = os.listdir(os.path.join(data_root, ID_list[i]))
        print(mo_list)
        for mo in mo_list:
            print(mo)
            remove_seg_path = os.path.join(data_root, ID_list[i], mo, 'warped_ori_img.nii.gz')
            remove_ori_path = os.path.join(data_root, ID_list[i], mo, 'warped_ori.nii.gz')
            if os.path.exists(remove_ori_path):
                print(remove_ori_path)
                os.remove(remove_ori_path)
                print('delete ori')
            if os.path.exists(remove_seg_path):
                print(remove_seg_path)
                os.remove(remove_seg_path)
                print('delete seg')


def plot_image():
    img_path = sorted(glob.glob('/home/user/program/brain_seg/runs/brain_seg/test/IBIS0683/seg/06_seg.nii.gz'))
    for path in img_path:
        intensity = sitk.GetArrayFromImage(sitk.ReadImage(path))[132, ...]
        intensity = intensity / np.max(intensity) * 255
        Image.fromarray(intensity).convert('L').save('/home/user/doc/weekly_report/07_02_2022/one_channel/0683/seg_output_%s.png'
                                                     % os.path.dirname(path).split('/')[-1])


def hyperparameters_loop():
    parameters = dict(
        fold_list = [1, 2, 3, 4, 5],
        crop_size_list = [[128, 128, 96], [96, 96, 96]],
        update_frequency_list = [1, 20, 50],
        trade_off_list = [1, 10, 20, 30],
        lr_policy_list = ['linear', 'step'])
    para_values = [v for v in parameters.values()]

    for cfg.fold, cfg.crop_size, cfg.trade_off_reg, cfg.lr_policy in product(*para_values):
        print(cfg.fold, cfg.crop_size, cfg.trade_off_reg, cfg.lr_policy)


def create_gif():
    """Generate GIF figure."""
    file_path = '/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/gif/'
    save_path_prefix = '/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/gif'
    image_list = [file_path + img for img in os.listdir(file_path) if '.png' in img]
    gif_name = os.path.join(save_path_prefix, 'affine_1092_97.gif')
    frames = []
    image_list.sort()
    for image_name in image_list:
        if image_name.endswith('png'):
            print(image_name)
            frames.append(imageio.imread(image_name))
    # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.8)


def compute_value_from_csv():
    """Compute mean or std from the csv file."""
    start_fold = 1
    end_fold = 1

    mo = '06' # 06 or 12 or 24

    for fold in range(start_fold, end_fold + 1):

        # file_path = '/home/user/doc/project_related/brain_seg_reg_file/2022-08-25/proposed/rec_seg_{}.csv'.format(mo)
        file_path = '/data/rec_seg_06_4.csv'
        df = pd.read_csv(file_path)

        # Dice (mean and std)
        dice_csf_mean = df['Dice_csf'].mean()
        dice_gm_mean = df['Dice_gm'].mean()
        dice_wm_mean = df['Dice_wm'].mean()

        dice_csf_std = df['Dice_csf'].std()
        dice_gm_std = df['Dice_gm'].std()
        dice_wm_std = df['Dice_wm'].std()

        # ASD (mean and std)
        asd_csf_mean = df['csf_ASD'].mean()
        asd_gm_mean = df['gm_ASD'].mean()
        asd_wm_mean = df['wm_ASD'].mean()

        asd_csf_std = df['csf_ASD'].std()
        asd_gm_std = df['gm_ASD'].std()
        asd_wm_std = df['wm_ASD'].std()

        # HD (mean and std)
        hd_csf_mean = df['csf_HD'].mean()
        hd_gm_mean = df['gm_HD'].mean()
        hd_wm_mean = df['wm_HD'].mean()

        hd_csf_std = df['csf_HD'].std()
        hd_gm_std = df['gm_HD'].std()
        hd_wm_std = df['wm_HD'].std()

        append_df = pd.DataFrame([['Mean', dice_csf_mean, dice_gm_mean, dice_wm_mean, asd_csf_mean, asd_gm_mean, asd_wm_mean, hd_csf_mean, hd_gm_mean, hd_wm_mean],
                                  ['Std', dice_csf_std, dice_gm_std, dice_wm_std, asd_csf_std, asd_gm_std, asd_wm_std, hd_csf_std, hd_gm_std, hd_wm_std]],
                                 columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm', 'csf_ASD', 'gm_ASD', 'wm_ASD', 'csf_HD', 'gm_HD', 'wm_HD'])

        df = df.append(append_df)

        df.to_csv(file_path, index=False)


def test():
    import seaborn as sns
    df = sns.load_dataset("penguins")
    sns.pairplot(df, hue="species")
    import matplotlib.pyplot as plt
    plt.show()

def _ants_img_info(img):
    origin = img.origin
    spacing = img.spacing
    direction = img.direction

    return img.numpy(), origin, spacing, direction


def _resample(img, new_spacing, type='image'):
    '''
    function to normalize image to standard spacing
    :param img: origin image
    :param new_spacing: standard image spacing
    :param type: image means linear interpolation, and seg denotes nearest interpolation
    :return: resampled image
    '''
    assert type == 'image' or type == 'seg', 'type error, type expire to image or seg your type is:'.format(type)
    origin, spacing, direction, data = _ants_img_info(img)
    zoom_factory = np.array(spacing) / np.array(new_spacing)
    if type == 'image':
        zoomed_data = zoom(data, zoom_factory, order=3)
    elif type == 'seg':
        zoomed_data = zoom(data, zoom_factory, order=0)

    return ants.from_numpy(zoomed_data, origin, new_spacing, direction)


def _resize(img, target_size, type='image'):
    img, origin, spacing, direction = _ants_img_info(img)
    origin_size = img.shape
    new_spacing = np.array(origin_size) * np.array(spacing) / np.array(target_size)
    zoom_rate = np.array(target_size) / np.array(origin_size)
    if type == 'image':
        n_data = zoom(img, zoom_rate, order=3)
    elif type == 'seg':
        n_data = zoom(img, zoom_rate, order=1)
    n_img = ants.from_numpy(n_data, origin=list(origin), spacing=list(new_spacing), direction=direction)
    return n_img


def count_tissue_volume():
    """TODO: Calculate volume number of different tissues."""
    img_path = ''
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    for i in range(1, img.max()):
        print(np.count_nonzero(img==i))

def plot_line_csf():
    csf = []
    gm = []
    wm = []
    img_paths = sorted(glob.glob('/data/brain_reg_seg/IBIS0036/*/tissue.nii.gz'))
    for path in img_paths:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        csf.append(np.count_nonzero(img==1))
        gm.append(np.count_nonzero(img==2))
        wm.append(np.count_nonzero(img==3))
    # fig, ax = plt.subplots()
    # for i in range(1,4):
    #     ax1 = plt.subplot(1,3,i, aspect='equal')
    #     ax1.plot([1,2,3])
    #     ax1.set_xticklabels(['06', '12', '24'])
    csf2 = [237006, 257062, 291965]
    csf = np.stack((csf, csf2), axis=1)
    print(csf)
    dataframe = pd.DataFrame(csf, columns=['volume1', 'volume2'])
    sns.lineplot(data=dataframe, legend=False)

    # ax.plot([1,2,3])
    # ax.set_ylabel('Volume ($\mathregular{cm^3}$)')
    # ax.set_xlabel('Timepoints')

    # plt.axis('off')

    ax = plt.gca()
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.title('CSF', color='white')
    plt.ylabel('Volume', color='white')
    plt.xticks([0,1,2], labels=['06-month', '12-month', '24-month'], rotation=45)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 5))
    plt.tight_layout()
    plt.savefig('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/volume_change_csf.png', transparent=True)
    # plt.show()

def plot_line_gm():
    csf = []
    gm = []
    wm = []
    img_paths = sorted(glob.glob('/data/brain_reg_seg/IBIS0036/*/tissue.nii.gz'))
    for path in img_paths:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        csf.append(np.count_nonzero(img==1))
        gm.append(np.count_nonzero(img==2))
        wm.append(np.count_nonzero(img==3))
    # fig, ax = plt.subplots()
    # for i in range(1,4):
    #     ax1 = plt.subplot(1,3,i, aspect='equal')
    #     ax1.plot([1,2,3])
    #     ax1.set_xticklabels(['06', '12', '24'])
    gm2 = [453447, 563520, 703522]
    gm = np.stack((gm, gm2), axis=1)
    print(gm)
    dataframe = pd.DataFrame(gm, columns=['volume1', 'volume2'])
    sns.lineplot(data=dataframe, legend=False)

    # ax.plot([1,2,3])
    # ax.set_ylabel('Volume ($\mathregular{cm^3}$)')
    # ax.set_xlabel('Timepoints')

    # plt.axis('off')

    ax = plt.gca()
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.title('GM', color='white')
    plt.ylabel('Volume', color='white')
    plt.xticks([0,1,2], labels=['06-month', '12-month', '24-month'], rotation=45)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 5))
    plt.tight_layout()
    plt.savefig('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/volume_change_gm.png', transparent=True)
    # plt.show()

def plot_line_wm():
    csf = []
    gm = []
    wm = []
    img_paths = sorted(glob.glob('/data/brain_reg_seg/IBIS0036/*/tissue.nii.gz'))
    for path in img_paths:
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        csf.append(np.count_nonzero(img==1))
        gm.append(np.count_nonzero(img==2))
        wm.append(np.count_nonzero(img==3))
    # fig, ax = plt.subplots()
    # for i in range(1,4):
    #     ax1 = plt.subplot(1,3,i, aspect='equal')
    #     ax1.plot([1,2,3])
    #     ax1.set_xticklabels(['06', '12', '24'])
    wm2 = [328756, 395859, 401303]
    wm = np.stack((wm, wm2), axis=1)
    print(wm)
    dataframe = pd.DataFrame(wm, columns=['volume1', 'volume2'])
    sns.lineplot(data=dataframe, legend=False)

    # ax.plot([1,2,3])
    # ax.set_ylabel('Volume ($\mathregular{cm^3}$)')
    # ax.set_xlabel('Timepoints')

    # plt.axis('off')
    # ax_xtick = ax.get_xticklabels()

    ax = plt.gca()
    ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    plt.title('WM', color='white')
    plt.ylabel('Volume', color='white')
    plt.xticks([0,1,2], labels=['06-month', '12-month', '24-month'], rotation=45)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 5))
    plt.tight_layout()
    plt.savefig('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/volume_change_wm.png', transparent=True)
    # plt.show()

def plot_histgram():
    img_path = '/data/brain_reg_seg/IBIS0036/24mo/intensity.nii.gz'
    seg_path = '/data/brain_reg_seg/IBIS0036/24mo/tissue.nii.gz'
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    img_1 = img[seg == 1]
    img_2 = img[seg == 2]
    img_3 = img[seg == 3]
    label_1 = np.ones_like(img_1)
    label_2 = 2 * np.ones_like(img_2)
    label_3 = 3 * np.ones_like(img_3)
    img_123 = np.concatenate((img_1, img_2, img_3))
    label_123 = np.concatenate((label_1, label_2, label_3))
    label_img = np.stack((label_123, img_123), axis=1)
    dataframe = pd.DataFrame(label_img, columns=['tissue', 'intensity'])
    sns.histplot(data=dataframe, x='intensity', hue='tissue', bins=50, element="step",
                 palette=sns.color_palette()[:3], kde=True, kde_kws={'bw_adjust':1.7}, legend=False)
    plt.ylim(0, 140000)
    plt.axis('off')
    plt.show()
    # plt.savefig('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/intensity_0036_24.png', transparent=True)

def crop_image():
    img_path_1 = '/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/dense_06_1242_66.png'
    img_path_2 = '/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/hyper_06_1242_66.png'
    img_path_3 = '/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/our_06_1242_66.png'
    img_path_4 = '/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/tissue_06_1242_66.png'
    img_path_5 = '/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/trans_06_1242_66.png'
    img_path_6 = '/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/unet_06_1242_66.png'
    img_1 = cv2.imread(img_path_1)
    print(img_1.shape)
    img_2 = cv2.imread(img_path_2)
    img_3 = cv2.imread(img_path_3)
    img_4 = cv2.imread(img_path_4)
    img_5 = cv2.imread(img_path_5)
    img_6 = cv2.imread(img_path_6)
    img_1_crop = img_1[92:123, 82:113,:]
    img_2_crop = img_2[92:123, 82:113,:]
    img_3_crop = img_3[92:123, 82:113,:]
    img_4_crop = img_4[92:123, 82:113,:]
    img_5_crop = img_5[92:123, 82:113,:]
    img_6_crop = img_6[92:123, 82:113,:]
    cv2.imwrite('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/dense_06_1242_66-crop.png', img_1_crop)
    cv2.imwrite('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/hyper_06_1242_66-crop.png', img_2_crop)
    cv2.imwrite('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/our_06_1242_66-crop.png', img_3_crop)
    cv2.imwrite('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/tissue_06_1242_66-crop.png', img_4_crop)
    cv2.imwrite('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/trans_06_1242_66-crop.png', img_5_crop)
    cv2.imwrite('/home/user/doc/project_related/brain_seg_reg_file/brain_reg_seg/2022-12-17/figure_4/unet_06_1242_66-crop.png', img_6_crop)

def grid_3D_plot():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x_q = np.linspace(-1, 2, 100)
    x, y, z = np.meshgrid(np.arange(-1, 2, 1),
                          np.arange(-1, 2, 1),
                          np.arange(-1, 2, 1))
    ax.scatter(x, y, z, 'o')
    ax.set_aspect('auto')
    plt.axis('off')

    plt.savefig('/home/user/doc/weekly_and_group/group_meeting/2023-06-24/figure_5_grid.tiff', bbox_inches='tight', dpi=300, transparent=True)
    plt.show()


def print_metadata():
    img_path = '/data/dataset/brain_reg_seg/IBIS0015/12mo/intensity.nii.gz'
    img = sitk.ReadImage(img_path)
    print(img.Get())
    print('a')

if __name__ == '__main__':
    download_csv()