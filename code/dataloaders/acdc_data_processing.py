import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk

if __name__ == "__main__":
    slice_num = 0
    mask_path = sorted(glob.glob("D:/unet/ACDC/ACDC/database/training/*/*.nii.gz"))
    j = False
    for case in mask_path:
        print(case)
        case = case.replace("\\", "/")
        print(case)
        img_itk = sitk.ReadImage(case)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        image = sitk.GetArrayFromImage(img_itk)
        msk_path = case.replace("image", "label").replace(".nii.gz", "_gt.nii.gz")
        msk_path = msk_path.replace("\\", "/")
        if os.path.exists(msk_path):
            print(msk_path)
            msk_itk = sitk.ReadImage(msk_path)
            mask = sitk.GetArrayFromImage(msk_itk)
            print(mask)
            image = (image - image.min()) / (image.max() - image.min())
            print(image.shape)
            image = image.astype(np.float32)
            item = case.split("/")[-1].split(".")[0]
            if image.shape != mask.shape:
                print("Error")
            print("item")
            print(item)

            # if item[-2:] != "01":
            #     print("hello")
            #     item = item[:-2] + "02"
            #     print(item)
            if j:
                item = item[:-2] + "01"
            else:
                item = item[:-2] + "02"
            j = not j
            print(item)
            for slice_ind in range(image.shape[0]):
                f = h5py.File(
                    'D:/unet/SS-Net/data/ACDC/data/slices/{}_slice_{}.h5'.format(item, slice_ind+1), 'w')
                f.create_dataset(
                    'image', data=image[slice_ind], compression="gzip")
                f.create_dataset('label', data=mask[slice_ind], compression="gzip")
                f.close()
                slice_num += 1

            #validation list
            f_val = h5py.File(
                'D:/unet/SS-Net/data/ACDC/data/{}.h5'.format(item), 'w')
            f_val.create_dataset(
                'image', data=image, compression="gzip")
            f_val.create_dataset('label', data=mask, compression="gzip")
            f_val.close()
    print("Converted all ACDC volumes to 2D slices")
    print("Total {} slices".format(slice_num))
