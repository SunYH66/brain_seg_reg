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
from model import network
from utils import metrics
from utils.utils import save_image, multi_layer_dice_coefficient


if __name__ == '__main__':
    # add some specific options to run this project
    parser = argparse.ArgumentParser(description='simultaneous segmentation and registration for infant brain')
    parser.add_argument('--platform', choices=['local', 'server'], default='server', help='different platform, different dataroot')

    arg = parser.parse_args()

    if arg.platform == 'local':
        cfg.data_root = '/data/infant_brain_seg_reg/'
        cfg.batch_size = 1
        cfg.num_workers = 2
    elif arg.platform == 'server':
        cfg.data_root = '/public_bme/home/sunyh/data/infant_brain_seg_reg'
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
    csv_root = os.path.join(cfg.checkpoint_root, cfg.name, 'test')
    if os.path.exists(csv_root):
        shutil.rmtree(csv_root)

    # inference loop
    rec_reg_12 = list()
    rec_reg_24 = list()

    for i, data in enumerate(dataset.test_loader):
        model.set_input(data)
        model.eval()
        start_time = time.time()

        # define Spatial Transformer (spa_tra)
        spa_tra = network.SpatialTransformer(cfg.ori_size, mode='bilinear')
        spa_tra.to('cuda')

        # start to inference
        with torch.no_grad(): # no gradients to accelerate
            # set reg net input
            model.fix_seg = torch.cat((model.seg_06, model.seg_06), dim=0)
            model.mov_seg = torch.cat((model.seg_12, model.seg_24), dim=0)
            model.mov_ori = torch.cat((model.img_12, model.img_24), dim=0)

            # obtain the deformation field from reg net TODO: change for different reg modes
            model.warped_seg, model.flow = model.net_reg(torch.cat((model.seg_12, model.seg_06), dim=1))

            # apply deformaton field to ori image TODO: change for different reg modes
            model.warped_ori = spa_tra(model.img_12, model.flow)

            # save reg net output
            save_image(model.warped_seg, cfg, model.get_image_path(), label='warped_seg')
            save_image(model.warped_ori, cfg, model.get_image_path(), label='warped_ori')

            # set data format to calculate dice
            warped_seg = model.warped_seg.cpu().numpy().squeeze(1)  # (B, 1, H, W, D) -> (B, H, W, D)
            fix_seg = model.fix_seg[0:cfg.batch_size, ...].cpu().numpy().squeeze()  # (B, 1, H, W, D) -> (H, W, D)

            # ==========================================registration results============================================
            for k in range(warped_seg.shape[0]):
                # Dice
                dice_csf, dice_gm, dice_wm = multi_layer_dice_coefficient(warped_seg[k, ...], fix_seg)

                # ASD and HD
                csf_surface_distance = metrics.compute_surface_distances(fix_seg == 1,
                                                                         warped_seg[k, ...] == 1,
                                                                         spacing_mm=[1, 1, 1])
                gm_surface_distance = metrics.compute_surface_distances(fix_seg == 2,
                                                                        warped_seg[k, ...] == 2,
                                                                        spacing_mm=[1, 1, 1])
                wm_surface_distance = metrics.compute_surface_distances(fix_seg == 3,
                                                                        warped_seg[k, ...] == 3,
                                                                        spacing_mm=[1, 1, 1])

                csf_ASD = metrics.compute_average_surface_distance(csf_surface_distance)
                gm_ASD = metrics.compute_average_surface_distance(gm_surface_distance)
                wm_ASD = metrics.compute_average_surface_distance(wm_surface_distance)

                csf_HD = metrics.compute_robust_hausdorff(csf_surface_distance, 99)
                gm_HD = metrics.compute_robust_hausdorff(gm_surface_distance, 99)
                wm_HD = metrics.compute_robust_hausdorff(wm_surface_distance, 99)

                # record image ID
                img_ID = model.get_image_path()
                img_ID = img_ID['img_06_path'][0].split('/')[-2]

                if k == 0:
                    print('12_reg', dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD)
                    rec_reg_12.append(
                        [img_ID, dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD])
                elif k == 1:
                    print('24_reg', dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD)
                    rec_reg_24.append(
                        [img_ID, dice_csf, dice_gm, dice_wm, csf_ASD, gm_ASD, wm_ASD, csf_HD, gm_HD, wm_HD])


        print('Using time %.2f min for one batch.' % ((time.time() - start_time) / 60))
        print('-------------------------------------------------------------------------------------------------------------------------')


        if rec_reg_12 is not None:
            df = pd.DataFrame(rec_reg_12, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm',
                                                   'csf_ASD', 'gm_ASD', 'wm_ASD', 'csf_HD', 'gm_HD', 'wm_HD'])
            df.to_csv(os.path.join(csv_root, 'rec_reg_12.csv'), index=False)
        if rec_reg_24 is not None:
            df = pd.DataFrame(rec_reg_24, columns=['ID', 'Dice_csf', 'Dice_gm', 'Dice_wm',
                                                   'csf_ASD', 'gm_ASD', 'wm_ASD', 'csf_HD', 'gm_HD', 'wm_HD'])
            df.to_csv(os.path.join(csv_root, 'rec_reg_24.csv'), index=False)