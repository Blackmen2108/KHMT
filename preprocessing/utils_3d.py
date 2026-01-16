import SimpleITK as sitk
import numpy as np

def resample_image(image, new_spacing=(1.0, 1.0, 1.0)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * osp / nsp))
        for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(image)

def clip_and_normalize(array, minHU=-1000, maxHU=400):
    array = np.clip(array, minHU, maxHU)
    array = (array - minHU) / (maxHU - minHU)
    return array.astype(np.float32)
