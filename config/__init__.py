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

    if os.path.exists(os.path.join(cfg.checkpoint_root, cfg.name, 'model')):
        print('Found non-empty checkpoint dir {}, \nenter {} to delete all files, {} to continue:'
              .format(os.path.join(cfg.checkpoint_root, cfg.name, 'model'), '\'Yes\'', '\'No\''))

        choice = input().lower()
        if choice == 'yes':
            shutil.rmtree(os.path.join(cfg.checkpoint_root, cfg.name, 'model'))
            os.makedirs(os.path.join(cfg.checkpoint_root, cfg.name, 'model'))
        elif choice == 'no':
            pass
        else:
            raise ValueError('choice error')
    else:
        os.makedirs(os.path.join(cfg.checkpoint_root, cfg.name, 'model'))

    cfg_dir = os.path.join(cfg.checkpoint_root, cfg.name, 'log')
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_name = os.path.join(cfg_dir, 'cfg.txt')
    with open(cfg_name, 'w') as c_file:
        c_file.write(cfg_info)
