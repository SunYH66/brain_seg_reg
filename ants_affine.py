# --coding:utf-8--

import os
import time

import ants
import numpy as np
import pandas as pd


# basic configuration
data_root = '/data/infant_brain_seg_reg'

start_fold = 1
end_fold = 5

save_image = True

# start looping to calculate registration results (Dice, ASD and HD) with different folds
for fold in range(start_fold, end_fold + 1):

    rec_reg_12 = []
    rec_reg_24 = []

    # obtain file list for current fold
    file_list = pd.read_csv('./csvfile/filelist_06_12_24.csv')
    file_list = file_list.loc[file_list['Fold'] == fold, 'ID']
    file_list = file_list.values.tolist()

    for file_ID in file_list:

        # get the month list (06, 12, 24 month)
        mo_list = os.listdir(os.path.join(data_root, file_ID))

        # loop 12 and 24 month to register to the 06 month
        for mo in mo_list:

            # ========================================= Registration process =============================================
            print(f'Registering {file_ID} of {mo} month ......')

            mov_list = mo_list.copy()
            mov_list.remove(mo)
            print(mov_list)

            st = time.time()
            img_path_fix_seg = os.path.join(data_root, file_ID, mo, 'segment.nii.gz')
            img_path_any_seg_1 = os.path.join(data_root, file_ID, mov_list[0], 'segment.nii.gz')
            img_path_any_seg_2 = os.path.join(data_root, file_ID, mov_list[1], 'segment.nii.gz')

            img_path_fix_ori = os.path.join(data_root, file_ID, mo, 'intensity_mask_out.nii.gz')
            img_path_any_ori_1 = os.path.join(data_root, file_ID, mov_list[0], 'intensity_mask_out.nii.gz')
            img_path_any_ori_2 = os.path.join(data_root, file_ID, mov_list[1], 'intensity_mask_out.nii.gz')

            fix_seg = ants.image_read(img_path_fix_seg)
            mov_seg = ants.image_read(img_path_any_seg_1)
            fix_ori = ants.image_read(img_path_fix_ori)
            mov_ori = ants.image_read(img_path_any_ori_1)

            result = ants.registration(fixed=fix_seg, moving=mov_seg, type_of_transform='Affine')
            warped_seg = ants.apply_transforms(fixed=fix_seg, moving=mov_seg, transformlist=result['fwdtransforms'], interpolator='nearestNeighbor')
            warped_ori = ants.apply_transforms(fixed=fix_ori, moving=mov_ori, transformlist=result['fwdtransforms'])

            if save_image:
                print('saving the reg results to:', os.path.join(data_root, file_ID, mov_list[0]))
                ants.image_write(warped_seg, os.path.join(data_root, file_ID, mov_list[0], 'warped_seg_to_{}.nii.gz'.format(mo[0:2])))
                ants.image_write(warped_ori, os.path.join(data_root, file_ID, mov_list[0], 'warped_ori_to_{}.nii.gz'.format(mo[0:2])))


            fix_seg = ants.image_read(img_path_fix_seg)
            mov_seg = ants.image_read(img_path_any_seg_2)
            fix_ori = ants.image_read(img_path_fix_ori)
            mov_ori = ants.image_read(img_path_any_ori_2)

            result = ants.registration(fixed=fix_seg, moving=mov_seg, type_of_transform='Affine')
            warped_seg = ants.apply_transforms(fixed=fix_seg, moving=mov_seg, transformlist=result['fwdtransforms'], interpolator='nearestNeighbor')
            warped_ori = ants.apply_transforms(fixed=fix_ori, moving=mov_ori, transformlist=result['fwdtransforms'])

            if save_image:
                print('saving the reg results to:', os.path.join(data_root, file_ID, mov_list[1]))
                ants.image_write(warped_seg, os.path.join(data_root, file_ID, mov_list[1], 'warped_seg_to_{}.nii.gz'.format(mo[0:2])))
                ants.image_write(warped_ori, os.path.join(data_root, file_ID, mov_list[1], 'warped_ori_to_{}.nii.gz'.format(mo[0:2])))

            print((time.time() - st) / 60)



