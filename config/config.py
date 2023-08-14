# --coding:utf-8--
from easydict import EasyDict as edict

__C = edict()
cfg = __C


####### general parameters ######
__C.general = {}
__C.phase = 'train' # train or test
__C.fold = 1      # ============================ need to be checked every experiment ===============================


####### config file path #######
__C.name = 'brain_reg_seg'    # experiment name
__C.reg_folder = 'reg_to_12_new'
__C.seg_folder = 'seg_12_consis'
__C.data_root = './dataset/brain_reg_seg/'
__C.csv_root = './csvfile/filelist_06_12_24_2.csv'
__C.checkpoint_root = './runs'


####### dataset config #######
# IBIS dataset of infant brain with 06, 12, 24 month
__C.datamode = 'IBIS'
__C.preprocess = 'crop'
__C.crop_size = [128, 128, 96] # crop_size_list for different crop size, 182, 218, 182
__C.ori_size = [192, 224, 192] # 256, 256, 256


###### training config ######
__C.model = 'proj'  # ==============================================================================================
__C.resume_epoch = -1 # ============================================================================================
__C.mo_list = [12, 6, 24] # month list
# __C.seg_train_mode = 'single' # ['fusion_only' | 'fusion_seg' | 'single' | 'trans' | 'attention' | 'DenseVoxelNet' | 'ResNetVAE' | 'HyperDenseNet' | 'CycleGAN']
__C.reg_trade_off = 10 # different trade_off_reg
__C.seg_trade_off = 1 # different trade_off_seg

__C.batch_size = 4
__C.num_workers = 4
__C.input_nc = 1
__C.output_nc = 1
__C.lr = 1e-3
__C.num_epoch = 1000
__C.save_frequency = 10
__C.val_frequency = 100
__C.pool_size = 50
__C.loss_verbose = True
__C.seg_fusion = True
__C.reg_net_type = 'voxelmorph'
__C.seg_net_type = 'unet'


###### test config ######
__C.load_epoch = 500
__C.test_folder = 'test' # [test | reg_results] ###===only model.test phase can use 'test'===###
__C.aff_tag = None # [None for train phase or specific epoch test | 'best_val' for test phase]


####### learning rate config #######
__C.lr_policy = 'linear' # [linear | step | plateau | cosine] different policies
__C.decay_epoch = 200

#TODO: change cfg.model == 'reg' to 'reg' in cfg.model'
#TODO: 灰度归一化的方式
#TODO: why zoom can interplate value that below zero
#TODO:__C.img_name = 'intensity.nii.gz' 'intensity_mask_out'
#TODO:__C.seg_name = 'tissue.nii.gz'  'segment'
#TODO: test batch_size >= 2
#TODO: test_reg_folder test_seg_folder
#TODO: sliding window inference
#TODO: simpleitk read image axis问题，有的时候transpose没有任何作用,direction就不同
#TODO: 石峰老师数据问题
#TODO: voxelmorph配准
#TODO：transformer 各种网络
#TODO: best validation


# warped_ori_to_{}               warped_seg_to_{}                      affine...ori_guided...linear
# warped_ori_to_{}_syn_ori       warped_seg_to_{}_syn_ori              syn...ori_guided...linear