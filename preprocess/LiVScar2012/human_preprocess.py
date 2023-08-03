import os
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import zoom

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Define the directory path to search for nrrd files
dir_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\human_dataset\human_dataset\training"

# Define the specific string format to search for in the file names
img_format = "de.nrrd"
infarct_format = "de_infarct.nrrd"
myo_format = "myo.nrrd"

src_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\meta_dataset\imagesTr"
infarct_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\meta_dataset\label_segTr"
myo_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\meta_dataset\label_myoTr"
sr_save_path = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\meta_dataset\label_srTr"

mkdir(src_save_path)
mkdir(infarct_save_path)
mkdir(myo_save_path)
mkdir(sr_save_path)

# Loop through all files in the directory
for file_name in os.listdir(dir_path):
    # Check if the file is an nrrd file and contains the specified string format
    if file_name.endswith(".nrrd") and img_format in file_name:
        # Load the nrrd file
        img_file_path = os.path.join(dir_path, file_name)
        img_data, header = nrrd.read(img_file_path)

        lable_file_path = os.path.join(dir_path, file_name.replace(img_format, infarct_format))
        label_data, header = nrrd.read(lable_file_path)

        myo_file_path = os.path.join(dir_path, file_name.replace(img_format, myo_format))
        myo_data, header = nrrd.read(myo_file_path)
        print(img_data.shape)
        h, w, n = img_data.shape
        for i in range(n):
            img_slice = img_data[:, :, i]
            label_slice = label_data[:, :, i]
            myo_slice = myo_data[:, :, i]

            if np.sum(label_slice != 0) != 0:
                # Find the indices of the non-zero elements in the array
                indices = np.nonzero(myo_slice)

                # Stack the indices together into an (n, 2) array
                points = np.column_stack(indices)

                # Calculate the centroid using the mean of the points
                centroid = points.mean(axis=0)
                centroid = np.floor(centroid)
                # Print the closest point as the approximate centroid
                print("Approximate centroid: ", centroid)

                row, col = int(centroid[0]), int(centroid[1])
                if w > 400:
                    img_slice = img_slice[row - 80:row + 80, col - 80:col + 80]
                    label_slice = label_slice[row - 80:row + 80, col - 80:col + 80]
                    myo_slice = myo_slice[row - 80:row + 80, col - 80:col + 80]
                    sr_slice = zoom(img_slice, 512 / 160, order=3)
                elif w > 300:
                    img_slice = img_slice[row-60:row+60, col-60:col+60]
                    label_slice = label_slice[row-60:row+60, col-60:col+60]
                    myo_slice = myo_slice[row-60:row+60, col-60:col+60]
                    sr_slice = zoom(img_slice, 512 / 120, order=3)
                else:
                    img_slice = img_slice[row - 48:row + 48, col - 48:col + 48]
                    label_slice = label_slice[row - 48:row + 48, col - 48:col + 48]
                    myo_slice = myo_slice[row - 48:row + 48, col - 48:col + 48]
                    sr_slice = zoom(img_slice, 512 / 96, order=3)

                plt.imshow(sr_slice, cmap='gray')
                plt.show()
                plt.imshow(label_slice, cmap='gray')
                plt.show()
                sitk.WriteImage(sitk.GetImageFromArray(img_slice), os.path.join(src_save_path, file_name.replace(img_format, f"de_{i}.nii.gz")))
                sitk.WriteImage(sitk.GetImageFromArray(label_slice), os.path.join(infarct_save_path, file_name.replace(img_format, f"de_infarct_{i}.nii.gz")))
                sitk.WriteImage(sitk.GetImageFromArray(myo_slice), os.path.join(myo_save_path, file_name.replace(img_format, f"myo_{i}.nii.gz")))
                sitk.WriteImage(sitk.GetImageFromArray(sr_slice), os.path.join(sr_save_path, file_name.replace(img_format, f"sr_{i}.nii.gz")))


