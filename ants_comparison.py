# --coding:utf-8--
import os
import ants
import pandas as pd
from utils.utils import multi_layer_dice_coefficient
from utils import metrics

# basic configuration
data_root = '/data/infant_brain_seg_reg'

start_fold = 1
end_fold = 2

save_image = False

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
        for mo in mo_list[1:]:

            # ========================================= Registration process =============================================
            print(f'Registering {file_ID} of {mo} month ......')

            img_path_06_seg = os.path.join(data_root, file_ID, '06mo', 'segment.nii.gz')
            img_path_any_seg = os.path.join(data_root, file_ID, mo, 'segment.nii.gz')
            img_path_06_ori = os.path.join(data_root, file_ID, '06mo', 'intensity_mask_out.nii.gz')
            img_path_any_ori = os.path.join(data_root, file_ID, mo, 'intensity_mask_out.nii.gz')

            fix_seg = ants.image_read(img_path_06_seg)
            mov_seg = ants.image_read(img_path_any_seg)
            fix_ori = ants.image_read(img_path_06_ori)
            mov_ori = ants.image_read(img_path_any_ori)

            result = ants.registration(fixed=fix_ori, moving=mov_ori, type_of_transform='Affine')
            warped_seg = ants.apply_transforms(fixed=fix_seg, moving=mov_seg, transformlist=result['fwdtransforms'], interpolator='nearestNeighbor')
            warped_ori = ants.apply_transforms(fixed=fix_ori, moving=mov_ori, transformlist=result['fwdtransforms'])

            if save_image:
                print('saving the reg results to:', os.path.join(data_root, file_ID, mo))
                ants.image_write(warped_seg, os.path.join(data_root, file_ID, mo, 'warped_seg_to_06_syn_test.nii.gz'))
                ants.image_write(warped_ori, os.path.join(data_root, file_ID, mo, 'warped_ori_to_06_syn_test.nii.gz'))

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

            if mo == '12mo':
                print('12_reg_results:', dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD)
                rec_reg_12.append([file_ID, dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD])
                print('-------------------------------------------------------------------------------------------------------------------------')
            elif mo == '24mo':
                print('24_reg_results:', dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD)
                rec_reg_24.append([file_ID, dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD])
                print('-------------------------------------------------------------------------------------------------------------------------')

    if rec_reg_12 is not None:
        df = pd.DataFrame(rec_reg_12, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm',
                                               'csf_ASD', 'gm_ASD', 'wm_ASD', 'csf_HD', 'gm_HD', 'wm_HD'])
        df.to_csv(os.path.join(data_root, 'rec_reg_12_fold{}.csv').format(fold), index=False)

    if rec_reg_24 is not None:
        df = pd.DataFrame(rec_reg_24, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm',
                                               'csf_ASD', 'gm_ASD', 'wm_ASD', 'csf_HD', 'gm_HD', 'wm_HD'])
        df.to_csv(os.path.join(data_root, 'rec_reg_24_fold{}.csv').format(fold), index=False)

