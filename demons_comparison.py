# --coding:utf-8--
import SimpleITK as sitk
import sys
import os

def demons():

    def command_iteration(filter):
        print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")

    img_path_fix_seg = '/data/brain_reg_seg/IBIS0273/06mo/tissue.nii.gz'
    img_path_any_seg = '/data/brain_reg_seg/IBIS0273/24mo/tissue.nii.gz'
    img_path_fix_ori = '/data/brain_reg_seg/IBIS0273/06mo/intensity.nii.gz'
    img_path_any_ori = '/data/brain_reg_seg/IBIS0273/24mo/intensity.nii.gz'

    fix_seg = sitk.ReadImage(img_path_fix_seg)
    mov_seg = sitk.ReadImage(img_path_any_seg)
    fix_ori = sitk.ReadImage(img_path_fix_ori)
    mov_ori = sitk.ReadImage(img_path_any_ori)

    matcher = sitk.HistogramMatchingImageFilter()
    if fix_ori.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(64)
    else:
        matcher.SetNumberOfHistogramLevels(64)
    matcher.SetNumberOfMatchPoints(2)
    matcher.ThresholdAtMeanIntensityOn()
    mov_ori = matcher.Execute(mov_ori, fix_ori)

    # The fast symmetric forces Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in
    # SimpleITK
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(60)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(1.0)

    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

    displacementField = demons.Execute(fix_ori, mov_ori)

    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f" RMS: {demons.GetRMSChange()}")

    outTx = sitk.DisplacementFieldTransform(displacementField)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fix_ori)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)

    out_ori = resampler.Execute(mov_ori)
    out_seg = resampler.Execute(mov_seg)
    sitk.WriteImage(out_ori, os.path.join(os.path.dirname(img_path_any_ori), 'warped_ori_to_06_demons.nii.gz'))
    sitk.WriteImage(out_seg, os.path.join(os.path.dirname(img_path_any_ori), 'warped_ori_to_06_demons_seg.nii.gz'))

if __name__ == '__main__':
    demons()