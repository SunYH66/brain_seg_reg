# --coding:utf-8--
import sys
import time
import argparse
from tqdm import tqdm
sys.path.append('..')

from brain_reg_seg.data import create_dataset
from brain_reg_seg.model import create_model
from brain_reg_seg.config.config import cfg
from brain_reg_seg.config import print_configs
from brain_reg_seg.utils.visualizer import Visualizer
from brain_reg_seg.utils.utils import save_image, save_resluts, save_best_val, check_val_folder


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
        cfg.data_root = '/public_bme/home/sunyh/data/brain_reg_seg/'
        cfg.batch_size = 3
        cfg.num_workers = 3

    # print and save configure
    print_configs(cfg)

    # setup dataset, model and visualizer
    current_best_dice = 0
    dataset = create_dataset(cfg)
    model = create_model(cfg)
    model.setup_model(cfg)
    visualizer = Visualizer(cfg)

    # start training
    for epoch in tqdm(range(cfg.resume_epoch + 1, cfg.num_epoch), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        data_progress = tqdm(dataset.train_loader, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        iter_num = len(dataset.train_dataset) // cfg.batch_size
        start_time = time.time()

        # get each batch_image and feed into the model
        for i, data in enumerate(data_progress):

            cfg.phase = 'train'
            model.set_input(data)
            model.train()
            model.optimize_train_parameters(epoch * iter_num + i, cfg)

            # display training results and lossses in the tensorboard
            visualizer.display_current_results(model.get_current_visuals(), epoch * iter_num + i)
            visualizer.display_current_loss(model.get_current_loss(), epoch * iter_num + i)

            # run the validation
            if (epoch * iter_num + i) % cfg.val_frequency == 0:

                model.eval()
                cfg.phase = 'val'
                check_val_folder(cfg)

                for j, val_data in enumerate(dataset.val_loader):
                    model.set_input(val_data)
                    model.optimize_val_parameters(cfg)
                    # save_resluts(model.get_current_visuals(), model, cfg)
                    save_image(model.get_current_visuals(), cfg, model.get_image_path())
                    visualizer.display_current_loss(model.get_current_loss(), epoch * iter_num + i, '_val')

                # current_best_dice = save_best_val(cfg, current_best_dice, model, epoch)

            # save the model
            if epoch % cfg.save_frequency == 0:
                model.save_model(epoch)

        # update the learning rate
        model.update_learning_rate(cfg)

        visualizer.print_current_loss(model.get_current_loss(), epoch)
        print('Training took time %.2f minutes for one epoch.' % ((time.time() - start_time) / 60))