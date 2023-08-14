# --coding:utf-8--
import os
import time
import torch
import shutil
import argparse
import pandas as pd

from config.config import cfg
from data import create_dataset
from model import create_model
from utils import metrics
from utils.utils import save_image, multi_layer_dice_coefficient
from monai.inferers import SlidingWindowInferer


if __name__ == '__main__':
    # add some specific options to run this project
    parser = argparse.ArgumentParser(description='simultaneous segmentation and registration for infant brain')
    parser.add_argument('--platform', choices=['local', 'server'], default='server', help='different platform, different dataroot')

    arg = parser.parse_args()

    if arg.platform == 'local':
        cfg.data_root = '/data/brain_reg_seg/'
        cfg.batch_size = 1
        cfg.num_workers = 2
    elif arg.platform == 'server':
        cfg.data_root = '/public/bme/home/v-sunyh2/data/brain_reg_seg'
        cfg.batch_size = 3
        cfg.num_workers = 3

    # edit configuration
    cfg.phase = 'test'
    cfg.batch_size = 1

    # create dataset and model
    dataset = create_dataset(cfg)
    model = create_model(cfg)
    model.setup_model(cfg)

    # remove existing fold for recording results and establist a new one
    csv_root = os.path.join(cfg.checkpoint_root, cfg.name, cfg.seg_folder, cfg.test_folder)
    if os.path.exists(csv_root):
        shutil.rmtree(csv_root)

    # inference loop
    rec_seg = list()

    for i, data in enumerate(dataset.test_loader):
        model.set_input(data)
        model.eval()
        start_time = time.time()

        # define sliding window inference object
        inferer = SlidingWindowInferer(cfg.crop_size, overlap=0.25)

        # start to inference
        with torch.no_grad(): # no gradients to accelerate

            # obtain seg net output
            # TODO: Change when test different months
            # model.seg_output = inferer(model.img_06, model.net_seg)
            model.seg_output = model.net_seg(model.img)
            model.seg_output = model.seg_output[0]
            # transfer results to one-channel results
            model.seg_output = model.seg_output.argmax(1).unsqueeze(1) # (B, 1, H, W, D)

            # save seg net output
            save_image(model.seg_output, cfg, model.get_image_path(), label='seg_output')

            # set data format to calculate dice
            # TODO: Change when test different months
            seg_output = model.seg_output.cpu().numpy() # (B, C, H, W, D) = (1, 1, 256, 256, 256)
            seg_GT = model.seg.cpu().numpy() # (B, C, H, W, D) = (1, 1, 256, 256, 256)


            # save the segmentation results (dice, hd) to a csv file
            # =======================================segmentation results===============================================
            for k in range(seg_output.shape[0]):
                # Dice
                dice_csf, dice_gm, dice_wm = multi_layer_dice_coefficient(seg_output[k, ...].squeeze(), seg_GT[k, ...].squeeze())

                # ASD and HD
                csf_surface_distance = metrics.compute_surface_distances(seg_GT[k, ...].squeeze() == 1,
                                                                         seg_output[k, ...].squeeze() == 1,
                                                                         spacing_mm=[1, 1, 1])
                gm_surface_distance = metrics.compute_surface_distances(seg_GT[k, ...].squeeze() == 2,
                                                                        seg_output[k, ...].squeeze() == 2,
                                                                        spacing_mm=[1, 1, 1])
                wm_surface_distance = metrics.compute_surface_distances(seg_GT[k, ...].squeeze() == 3,
                                                                        seg_output[k, ...].squeeze() == 3,
                                                                        spacing_mm=[1, 1, 1])

                csf_ASD = metrics.compute_average_surface_distance(csf_surface_distance)
                gm_ASD = metrics.compute_average_surface_distance(gm_surface_distance)
                wm_ASD = metrics.compute_average_surface_distance(wm_surface_distance)

                csf_HD = metrics.compute_robust_hausdorff(csf_surface_distance, 99)
                gm_HD = metrics.compute_robust_hausdorff(gm_surface_distance, 99)
                wm_HD = metrics.compute_robust_hausdorff(wm_surface_distance, 99)

                img_ID = model.get_image_path()
                img_ID = img_ID['img_path_main'][0].split('/')[-2]
                print(img_ID)

                if k == 0:
                    print('Seg_results:', dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD)
                    rec_seg.append([img_ID, dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD])


        print('Using time %.2f min for one batch.' % ((time.time() - start_time) / 60))
        print('-------------------------------------------------------------------------------------------------------------------------')


        if rec_seg is not None:
            df = pd.DataFrame(rec_seg, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm',
                                                   'csf_ASD', 'gm_ASD', 'wm_ASD', 'csf_HD', 'gm_HD', 'wm_HD'])
            # TODO: change csv file name for different months
            df.to_csv(os.path.join(csv_root, 'rec_seg_{:02d}.csv'.format(cfg.mo_list[0])), index=False)

