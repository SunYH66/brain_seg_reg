# --coding:utf-8--
from easydict import EasyDict as edict

__C = edict()
cfg = __C

####### general parameters ######
__C.general = {}
__C.phase = 'train' # train or test
__C.fold = 1      # ============================ need to be checked every experiment ===============================


####### config file path #######
__C.name = 'brain_reg_12_24_to_06'    # experiment name
__C.data_root = './dataset/infant_brain_seg_reg/'
__C.csv_root = './csvfile/filelist_06_12_24.csv'
__C.checkpoint_root = './runs'


####### dataset config #######
# IBIS dataset of infant brain with 06, 12, 24 month
__C.datamode = 'IBIS'
__C.preprocess = 'crop'
__C.crop_size = [128, 128, 96] # crop_size_list for different crop size
__C.ori_size = [256, 256, 256]


###### training config ######
__C.batch_size = 4
__C.resume_epoch = -1
__C.num_workers = 4
__C.update_frequency = 1 # update_frequency_list for different frequencies===========================================
__C.continue_train = False
__C.num_epoch = 1000
__C.reg_net_type = 'voxelmorph'
__C.seg_net_type = 'unet'
__C.trade_off = 10 # different trade_off
__C.input_nc = 1
__C.output_nc = 4
__C.lr = 1e-4 # different learning rate
__C.save_frequency = 10
__C.val_frequency = 100
__C.model = 'reg'  # ===============================================================================================


###### test config ######
__C.load_epoch = 990
__C.test_folder = 'test'


####### learning rate config #######
__C.lr_policy = 'linear' # [linear | step | plateau | cosine] different policies
__C.decay_epoch = 300
