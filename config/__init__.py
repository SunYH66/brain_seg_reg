# --coding:utf-8--
__all__ = ['print_configs']

import os
import shutil

def print_configs(cfg):
    """TODO:print and save configs to /checkpoint_root/experiment_name/opts.txt"""
    cfg_info = ''
    cfg_info += '--------------------------------------------Configure-----------------------------------------------\n'
    for key, value in cfg.items():
        cfg_info += '{:<50}:{:>50}\n'.format(str(key), str(value))
    cfg_info += '------------------------------------------------End---------------------------------------------------'

    print(cfg_info)

    # if os.path.exists(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'model')):
    #     print('Found non-empty checkpoint dir {}, \nenter {} to delete all files, {} to continue:'
    #           .format(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'model'), '\'Yes\'', '\'No\''))
    #
    #     choice = input().lower()
    #     if choice == 'yes':
    #         shutil.rmtree(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'model'))
    #         os.makedirs(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'model'))
    #     elif choice == 'no':
    #         pass
    #     else:
    #         raise ValueError('choice error')
    # else:
    #     os.makedirs(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'model'))

    if os.path.exists(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'model')):
        if cfg.resume_epoch == -1: #同一个project，重新跑
            shutil.rmtree(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder)))
            os.makedirs(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder)))
        else: #同一个project，load checkpoint继续跑
            pass
    else: #不同project，开始跑
        os.makedirs(os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'model'))

    cfg_dir = os.path.join(cfg.checkpoint_root, cfg.name, '{}'.format(cfg.reg_folder if cfg.model == 'reg' else cfg.seg_folder), 'log')
    if os.path.exists(cfg_dir):
        pass
    else:
        os.makedirs(cfg_dir)
    cfg_name = os.path.join(cfg_dir, 'cfg.txt')
    with open(cfg_name, 'w') as c_file:
        c_file.write(cfg_info)
