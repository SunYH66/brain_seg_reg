# --coding:utf-8--
import numpy as np
import SimpleITK as sitk

a = np.load('/data/PredictV.npy')

sitk.WriteImage(sitk.GetImageFromArray(a), '/data/PredictV.nii.gz')