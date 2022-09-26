# --coding:utf-8--
import os
import glob
import shutil
import imageio
from PIL import Image
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.ndimage import zoom
from config.config import cfg
from sklearn.model_selection import KFold
from itertools import product

def resample_image(new_sapcing=(2,2,2)):
    """TODO: Resample image"""
    img_path = '/data/infant_brain_seg_reg/copyDataOut/NDAR_V06-1_IBIS0015/intensity.nii.gz'
    img = sitk.ReadImage(img_path)
    spacing = img.GetSpacing()
    print('Old image shape:', img.GetSize())
    resample_ratio = np.array(spacing) / np.array(new_sapcing)
    resample_img = zoom(sitk.GetArrayFromImage(img), resample_ratio)
    print('New image shape:', resample_img.shape)

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
    # Build new folder
    file_list = sorted(glob.glob('/data/infant_brain_seg_reg/*'))
    valued_file_num = 0
    for path in file_list:
        print(path[-8:])
        if os.path.isdir(path):
            if 'V06' in path:
                month06_path = os.path.join('/data/infant_brain_seg_reg/', path[-8:], '06mo')
                shutil.copytree(path, month06_path)
                valued_file_num += 1
            elif 'V12' in path:
                month12_path = os.path.join('/data/infant_brain_seg_reg/', path[-8:], '12mo')
                shutil.copytree(path, month12_path)
                valued_file_num += 1
            elif 'V24' in path:
                month24_path = os.path.join('/data/infant_brain_seg_reg/', path[-8:], '24mo')
                shutil.copytree(path, month24_path)
                valued_file_num += 1
            else:
                raise ValueError('Unexpected file path...')
            print(valued_file_num)

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
    intensity_path = sorted(glob.glob('../dataset/infant_brain_seg_reg/*/*/intensity.nii.gz'))
    segment_path = sorted(glob.glob('../dataset/infant_brain_seg_reg/*/*/segment.nii.gz'))
    strip_path = sorted(glob.glob('../dataset/infant_brain_seg_reg/*/*/strip.nii.gz'))
    for i in range(len(intensity_path)):
        intensity = sitk.GetArrayFromImage(sitk.ReadImage(intensity_path[i]))
        segment = sitk.GetArrayFromImage(sitk.ReadImage(segment_path[i]))
        strip = sitk.GetArrayFromImage(sitk.ReadImage(strip_path[i]))
        intensity_mask_out = intensity * strip
        segment_mask_out = segment * strip
        sitk.WriteImage(sitk.GetImageFromArray(intensity_mask_out),
                        os.path.join(os.path.basename(intensity_path[i]), 'intensity_mask_out.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(segment_mask_out),
                        os.path.join(os.path.basename(intensity_path[i]), 'segment_mask_out.nii.gz'))


def generate_slice_image():
    seg_output_path_06 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/2022-08-25/proposed/IBIS0683/06mo/intensity_mask_out.nii.gz'))
    seg_output_path_12 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/2022-08-25/proposed/IBIS0683/12mo/warped_ori_to_06_affine.nii.gz'))
    seg_output_path_24 = sorted(glob.glob('/home/user/doc/project_related/brain_seg_reg_file/2022-08-25/proposed/IBIS0683/24mo/warped_ori_to_06_affine.nii.gz'))
    seg_output_06 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_06[0]))[151, ...]
    seg_output_06 = seg_output_06 / np.max(seg_output_06) * 255
    seg_output_12 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_12[0]))[151, ...]
    seg_output_12 = seg_output_12 / np.max(seg_output_12) * 255
    seg_output_24 = sitk.GetArrayFromImage(sitk.ReadImage(seg_output_path_24[0]))[151, ...]
    seg_output_24 = seg_output_24 / np.max(seg_output_24) * 255
    Image.fromarray(seg_output_06).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/2022-08-25/proposed/IBIS0683/06mo/06_intensity_151.png')
    Image.fromarray(seg_output_12).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/2022-08-25/proposed/IBIS0683/12mo/12_intensity_151.png')
    Image.fromarray(seg_output_24).convert('L').save('/home/user/doc/project_related/brain_seg_reg_file/2022-08-25/proposed/IBIS0683/24mo/24_intensity_151.png')


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
    df = pd.read_csv('../csvfile/filelist_06_12_24.csv', index_col=False)
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

    df.to_csv('../csvfile/filelist_06_12_24.csv', index=False)


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

    for cfg.fold, cfg.crop_size, cfg.update_frequency, cfg.trade_off, cfg.lr_policy in product(*para_values):
        print(cfg.fold, cfg.crop_size, cfg.update_frequency, cfg.trade_off, cfg.lr_policy)


def create_gif():
    """Generate GIF figure."""
    file_path = '/home/user/doc/project_related/brain_seg_reg_file/gif/'
    save_path_prefix = '/home/user/doc/project_related/brain_seg_reg_file/gif'
    image_list = [file_path + img for img in os.listdir(file_path) if '.png' in img]
    gif_name = os.path.join(save_path_prefix, 'affine_0015_129.gif')
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

        file_path = '/home/user/doc/project_related/brain_seg_reg_file/2022-08-25/proposed/rec_seg_{}.csv'.format(mo)

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


if __name__ == '__main__':
    create_gif()