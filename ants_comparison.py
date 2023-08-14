# --coding:utf-8--
import glob
import os
import time
import ants
import numpy as np
import SimpleITK as sitk
import pandas as pd
import scipy.io
from matplotlib import pyplot as plt

from utils.utils import multi_layer_dice_coefficient
from utils import metrics

def ibis_ants_reg():
    # basic configuration
    data_root = '/data/brain_reg_seg'

    start_fold = 1
    end_fold = 5

    mo_list = ['06', '12', '24']

    save_image = True

    verbose = False

    # start looping to calculate registration results (Dice, ASD and HD) with different folds
    for fold in range(start_fold, end_fold + 1):

        rec_reg_warped_1 = []
        rec_reg_warped_2 = []

        # obtain file list for current fold
        file_list = pd.read_csv('./csvfile/filelist_06_12_24_2.csv')
        file_list = file_list.loc[file_list['Fold'] == fold, 'ID']
        file_list = file_list.values.tolist()

        for file_ID in file_list:

            # loop 12 and 24 month to register to the 06 month
            for mo in mo_list[1:]:

                # ========================================= Registration process =============================================
                print(f'Registering {file_ID} of {mo} month ......')
                st = time.time()
                img_path_fix_seg = os.path.join(data_root, file_ID, mo_list[0]+'mo', 'tissue.nii.gz')
                img_path_any_seg = os.path.join(data_root, file_ID, mo+'mo', 'tissue.nii.gz')
                img_path_fix_ori = os.path.join(data_root, file_ID, mo_list[0]+'mo', 'intensity.nii.gz')
                img_path_any_ori = os.path.join(data_root, file_ID, mo+'mo', 'intensity.nii.gz')

                fix_seg = ants.image_read(img_path_fix_seg)
                mov_seg = ants.image_read(img_path_any_seg)
                fix_ori = ants.image_read(img_path_fix_ori)
                mov_ori = ants.image_read(img_path_any_ori)

                result = ants.registration(fixed=fix_ori, moving=mov_ori, type_of_transform='SyN')
                # warped_ori = result['fwdtransforms']

                warped_seg = ants.apply_transforms(fixed=fix_seg, moving=mov_seg, transformlist=result['fwdtransforms'], interpolator='nearestNeighbor')
                warped_ori = ants.apply_transforms(fixed=fix_ori, moving=mov_ori, transformlist=result['fwdtransforms'])

                if save_image:
                    print('saving the reg results to:', os.path.join(data_root, file_ID, mo+'mo'))
                    ants.image_write(warped_seg, os.path.join(data_root, file_ID, mo+'mo', 'warped_seg_to_{}.nii.gz'.format(mo_list[0])))
                    ants.image_write(warped_ori, os.path.join(data_root, file_ID, mo+'mo', 'warped_ori_to_{}.nii.gz'.format(mo_list[0])))
                print((time.time() - st) / 60)

                if verbose:

                    # ========================================= Save results process =============================================
                    # save the registration results (dice, asd and hd) to a csv file

                    # Dice
                    dice_csf, dice_gm, dice_wm = multi_layer_dice_coefficient(warped_seg.numpy(), fix_seg.numpy())

                    # ASD and HD
                    csf_surface_distance = metrics.compute_surface_distances(fix_seg.numpy() == 1,
                                                                             warped_seg.numpy() == 1,
                                                                             spacing_mm=[1, 1, 1])
                    gm_surface_distance = metrics.compute_surface_distances(fix_seg.numpy() == 2,
                                                                            warped_seg.numpy() == 2,
                                                                            spacing_mm=[1, 1, 1])
                    wm_surface_distance = metrics.compute_surface_distances(fix_seg.numpy() == 3,
                                                                            warped_seg.numpy() == 3,
                                                                            spacing_mm=[1, 1, 1])

                    csf_ASD = metrics.compute_average_surface_distance(csf_surface_distance)
                    gm_ASD = metrics.compute_average_surface_distance(gm_surface_distance)
                    wm_ASD = metrics.compute_average_surface_distance(wm_surface_distance)

                    csf_HD = metrics.compute_robust_hausdorff(csf_surface_distance, 99)
                    gm_HD = metrics.compute_robust_hausdorff(gm_surface_distance, 99)
                    wm_HD = metrics.compute_robust_hausdorff(wm_surface_distance, 99)

                    if mo == mo_list[1]:
                        print('{}_reg_results:'.format(mo), dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD)
                        rec_reg_warped_1.append([file_ID, dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD])
                        print('-------------------------------------------------------------------------------------------------------------------------')
                    if mo == mo_list[2]:
                        print('{}_reg_results:'.format(mo), dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD)
                        rec_reg_warped_2.append([file_ID, dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD])
                        print('-------------------------------------------------------------------------------------------------------------------------')


        if rec_reg_warped_1 is not None:
            df = pd.DataFrame(rec_reg_warped_1, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm',
                                                         'csf_ASD', 'gm_ASD', 'wm_ASD', 'csf_HD', 'gm_HD', 'wm_HD'])
            df.to_csv(os.path.join(data_root, 'rec_reg_{}_to_{}_fold{}_syn_ori.csv').format(mo_list[1], mo_list[0], fold), index=False)

        if rec_reg_warped_2 is not None:
            df = pd.DataFrame(rec_reg_warped_2, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm',
                                                         'csf_ASD', 'gm_ASD', 'wm_ASD', 'csf_HD', 'gm_HD', 'wm_HD'])
            df.to_csv(os.path.join(data_root, 'rec_reg_{}_to_{}_fold{}_syn_ori.csv').format(mo_list[2], mo_list[0], fold), index=False)


def dce_ants_reg():
    dir_name = os.listdir('/home/user/program/Projects/DCE_registration/results/ants_dce2')
    for i in range(len(dir_name)):
        os.makedirs(os.path.join('/home/user/program/Projects/DCE_registration/results/ants_dce', dir_name[i]), exist_ok=True)

    for i in range(11 * 22, 439, 22):
        for j in range(i, i + 22, 1):
            fix_img = ants.image_read(
                '/data/dataset/DCE_registration/DCE_MRI_train/original_Img_t%03d.nii.gz' % (i + 11))
            moving_img = ants.image_read(
                '/data/dataset/DCE_registration/DCE_MRI_train/original_Img_t%03d.nii.gz' % (j + 1))

            result = ants.registration(fix_img, moving_img, type_of_transform="SyN", syn_metric='CC')
            warped_moving = result['warpedmovout']
            dense_defor = sitk.ReadImage(result['fwdtransforms'][0])

            jac = ants.create_jacobian_determinant_image(fix_img, result['fwdtransforms'][0])


            sitk.WriteImage(dense_defor, os.path.join('/home/user/program/Projects/DCE_registration/results/ants_dce',
                                                         dir_name[np.floor(j / 22).astype(int)],
                                                         'defor_ants_t%02d_CC.nii.gz' % (np.mod(j, 22) + 1)))

            ants.image_write(warped_moving, os.path.join('/home/user/program/Projects/DCE_registration/results/ants_dce',
                                                         dir_name[np.floor(j / 22).astype(int)],
                                                         'ants_t%02d_CC.nii.gz' % (np.mod(j, 22) + 1)))

            ants.image_write(jac, os.path.join('/home/user/program/Projects/DCE_registration/results/ants_dce',
                                                         dir_name[np.floor(j / 22).astype(int)],
                                                         'jac_t%02d_CC.nii.gz' % (np.mod(j, 22) + 1)))

            print(os.path.join('/home/user/program/Projects/DCE_registration/results/ants_dce',
                                                         dir_name[np.floor(j / 22).astype(int)],
                                                         'ants_t%02d.nii.gz' % (np.mod(j, 22) + 1)))

"""
# ants.write_transform()

# img_path_fix_seg = '/public/bme/home/v-sunyh2/data/brain_reg_seg/IBIS0273/06mo/tissue.nii.gz'
# img_path_any_seg = '/public/bme/home/v-sunyh2/data/brain_reg_seg/IBIS0273/24mo/tissue.nii.gz'
# img_path_fix_ori = '/public/bme/home/v-sunyh2/data/brain_reg_seg/IBIS0273/06mo/intensity.nii.gz'
# img_path_any_ori = '/public/bme/home/v-sunyh2/data/brain_reg_seg/IBIS0273/24mo/intensity.nii.gz'
#
# fix_seg = ants.image_read(img_path_fix_seg)
# mov_seg = ants.image_read(img_path_any_seg)
# fix_ori = ants.image_read(img_path_fix_ori)
# mov_ori = ants.image_read(img_path_any_ori)
#
# result = ants.registration(fixed=fix_ori, moving=mov_ori, type_of_transform='SyN', syn_metric='demons')
# warped_ori = ants.apply_transforms(fixed=fix_ori, moving=mov_ori, transformlist=result['fwdtransforms'])
#
# print(result['fwdtransforms'])
# print(result['fwdtransforms'][1])
# print(result['fwdtransforms'][0])
#
# a = sitk.ReadImage(result['fwdtransforms'][0])
# sitk.WriteImage(a, '/public/bme/home/v-sunyh2/data/brain_reg_seg/IBIS0273/06mo/a.nii.gz')

# # tx_affine = ants.read_transform(result['fwdtransforms'][1])
#
#
# # ants.write_transform(tx_affine, '/data/a.mat')
# a = ants.image_read(result['fwdtransforms'][1])
# b = ants.transform_from_displacement_field(a)
#
# # compfield = ants.transform_from_displacement_field(fix_ori)
# # atx = ants.transform_from_displacement_field(compfield)
#
#
# # compfield = ants.compose_ants_transforms(result['fwdtransforms'])
# ants.write_transform(b, '/data/b.nii.gz')
#
#
# ants.write_transform(tx_dense, '/data/b.nii.gz')
#
# ants.image_write(warped_ori, os.path.join(os.path.dirname(img_path_any_ori), 'warped_ori_to_06_demons_seg.nii.gz'))

# import ants
# domain = ants.image_read( ants.get_ants_data('r16'))
# exp_field = ants.simulate_displacement_field(domain, field_type="exponential")
# bsp_field = ants.simulate_displacement_field(domain, field_type="bspline")
# bsp_xfrm = ants.transform_from_displacement_field(bsp_field * 3)
# domain_warped = ants.apply_ants_transform_to_image(bsp_xfrm, domain, domain)
"""

"""
def ants_test():
    img_path_fix_ori = '/data/dataset/brain_reg_seg/IBIS0015/06mo/intensity.nii.gz'
    img_path_mov2_ori = '/data/dataset/brain_reg_seg/IBIS0015/12mo/intensity2.nii.gz'
    img_path_mov3_ori = '/data/dataset/brain_reg_seg/IBIS0015/12mo/intensity3.nii.gz'
    img_path_mov4_ori = '/data/dataset/brain_reg_seg/IBIS0015/12mo/intensity4.nii.gz'

    fix_ori = ants.image_read(img_path_fix_ori)
    mov2_ori = ants.image_read(img_path_mov2_ori)
    mov3_ori = ants.image_read(img_path_mov3_ori)
    mov4_ori = ants.image_read(img_path_mov4_ori)

    print(fix_ori.shape)
    print(mov4_ori.shape)
    print(fix_ori[90, 133, 82])
    print(mov4_ori[90, 133, 82])
    print(fix_ori.shape)
    print(mov4_ori.shape)

    plt.imshow(fix_ori[:,:, 82], 'gray')
    plt.show()
    plt.imshow(mov2_ori[:,:, 82], 'gray')
    plt.show()
    plt.imshow(mov3_ori[:,:, 82], 'gray')
    plt.show()
    plt.imshow(mov4_ori[:,:, 82], 'gray')
    plt.show()

    result2 = ants.registration(fixed=fix_ori, moving=mov2_ori, type_of_transform='SyN')
    warped_ori2 = result2['warpedmovout']
    result3 = ants.registration(fixed=fix_ori, moving=mov3_ori, type_of_transform='SyN')
    warped_ori3 = result3['warpedmovout']
    result4 = ants.registration(fixed=fix_ori, moving=mov4_ori, type_of_transform='SyN')
    warped_ori4 = result4['warpedmovout']

    ants.image_write(warped_ori2, '/data/dataset/brain_reg_seg/IBIS0015/12mo/warped_ori2_to_06.nii.gz')
    ants.image_write(warped_ori3, '/data/dataset/brain_reg_seg/IBIS0015/12mo/warped_ori3_to_06.nii.gz')
    ants.image_write(warped_ori4, '/data/dataset/brain_reg_seg/IBIS0015/12mo/warped_ori4_to_06.nii.gz')
"""

def defor_cat():
    img_A_path = '/data/dataset/brain_reg_seg/IBIS0036/06mo/intensity.nii.gz'
    img_B_path = '/data/dataset/brain_reg_seg/IBIS0036/12mo/intensity.nii.gz'
    img_C_path = '/data/dataset/brain_reg_seg/IBIS0036/24mo/intensity.nii.gz'
    img_A = ants.image_read(img_A_path)
    img_B = ants.image_read(img_B_path)
    img_C = ants.image_read(img_C_path)

    reg_A_B = ants.registration(fixed=img_B, moving=img_A, type_of_transform='SyNOnly')
    defor_A_B = reg_A_B['fwdtransforms']
    reg_B_C = ants.registration(fixed=img_C, moving=img_B, type_of_transform='SyNOnly')
    defor_B_C = reg_B_C['fwdtransforms']
    reg_A_C = ants.registration(fixed=img_A, moving=img_C, type_of_transform='SyNOnly')
    defor_A_C = reg_A_C['fwdtransforms']
    print(defor_A_B, defor_B_C, defor_A_C)


if __name__ == '__main__':
    defor_cat()