import os
import cv2
import openpyxl
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from IPython import embed


def tissue_color(idx):
    if idx==3:
        return np.array([255,0,0])
    if idx==2:
        return np.array([0,255,0])
    if idx==1:
        return np.array([0,0,255])


def mri_normal(mri):
    mri = np.array(mri, dtype='float32')
    max_val = np.percentile(mri, 99)
    mri = np.clip(mri, mri.min(), max_val)
    mri /= max_val
    mri *= 255
    mri = np.array(mri, dtype='uint8')
    return mri


def get_axial_slice(T1, mask, idx):
    mri = T1[:, :, idx]
    mri = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
    mask = mask[:, :, idx]
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
    color_mask[mask == 1] = tissue_color(1)
    color_mask[mask == 2] = tissue_color(2)
    color_mask[mask == 3] = tissue_color(3)
    mri = cv2.addWeighted(mri, 0.5, color_mask, 0.5, 0)
    mri = cv2.flip(mri, 0)
    return mri


def get_coronal_slice(T1, mask, idx):
    mri = T1[:, idx, :]
    mri = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
    mask = mask[:, idx, :]
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
    color_mask[mask == 1] = tissue_color(1)
    color_mask[mask == 2] = tissue_color(2)
    color_mask[mask == 3] = tissue_color(3)
    mri = cv2.addWeighted(mri, 0.5, color_mask, 0.5, 0)
    mri = cv2.flip(mri, 0)
    return mri


def get_sagittal_slice(T1, mask, idx):
    mri = T1[idx, :, :]
    mri = cv2.cvtColor(mri, cv2.COLOR_GRAY2BGR)
    mask = mask[idx, :, :]
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
    color_mask[mask == 1] = tissue_color(1)
    color_mask[mask == 2] = tissue_color(2)
    color_mask[mask == 3] = tissue_color(3)
    mri = cv2.addWeighted(mri, 0.5, color_mask, 0.5, 0)
    mri = cv2.flip(mri, -1)
    return mri


def _sitk_img_info(img_path):
    img = sitk.ReadImage(img_path)
    origin = sitk.GetOrigin(img)
    spacing = sitk.GetSpacing(img)
    direction = sitk.GetDirection(img)
    img = sitk.GetArrayFromImage(img)

    return img, origin, spacing, direction


def _img_plot(img_path, seg_path, gap):
    img = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img)
    img = img.transpose(2, 1, 0)
    img = img[:, 180:360, :]
    img = mri_normal(img)

    seg = sitk.ReadImage(seg_path)
    seg = sitk.GetArrayFromImage(seg)
    seg = seg.transpose(2, 1, 0)
    seg = seg[:, 180:360, :]

    X, Y, Z = img.shape
    num_x, num_y, num_z = X//gap[0]-2, Y//gap[1]-2, Z//gap[2]-2

    demo_img_xy = np.zeros((Y, num_z*X, 3), dtype='uint8') # across Z direction
    demo_img_xz = np.zeros((Z, num_y*X, 3), dtype='uint8') # across Y direction
    demo_img_yz = np.zeros((Z, num_x*Y, 3), dtype='uint8') # across Z direction

    for idx in range(num_z):
        slice = get_axial_slice(img, seg, (idx+1)*gap[2])
        slice = np.rot90(slice, 1, (1, 0))
        demo_img_xy[:, idx*X:(idx+1)*X] = slice

    for idx in range(num_y):
        slice = get_coronal_slice(img, seg, (idx+1)*gap[1])
        slice = np.rot90(slice, 1, (0, 1))
        demo_img_xz[:, idx*X:(idx+1)*X, :] = slice

    for idx in range(num_x):
        slice = get_sagittal_slice(img, seg, (idx+1)*gap[0])
        slice = np.rot90(slice, 1, (1, 0))
        demo_img_yz[:, idx*Y:(idx+1)*Y, :] = slice

    return demo_img_xy, demo_img_xz, demo_img_yz








if __name__ == '__main__':
    source = '/media/liujm/IDEALab/femur/Normal'
    target = '/media/liujm/IDEALab/femur/qc'

    file_list = os.listdir(source)
    file_list.sort()
    gap = (20, 8, 10)
    for item in tqdm(file_list):
        img_path = os.path.join(source, item, 'img.nii.gz')
        seg_path = os.path.join(source, item, 'femur.nii.gz')
        xy, xz, yz, = _img_plot(img_path, seg_path, gap)
        cv2.imwrite(os.path.join(target, item+'_xy.jpg'), xy)
        cv2.imwrite(os.path.join(target, item+'_xz.jpg'), xz)
        cv2.imwrite(os.path.join(target, item+'_yz.jpg'), yz)


























