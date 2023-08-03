import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Define the input and output directory paths
image_dir = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\acdc\imagesTr"
myo_dir = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\acdc\labelsTr"

image_output_dir = r"C:\Users\wlc\PycharmProjects\MRes_Pipeline\data\meta_dataset_unsupervised\imagesTr"

# Define the target size for resizing
target_size = (128, 128, 64)

# Loop through all files in the input directory
for image_filename, myo_filename in zip(os.listdir(image_dir), os.listdir(myo_dir)):
    # Check if the file is a NIfTI file
    # Load the NIfTI file using nibabel
    image_path = os.path.join(image_dir, image_filename)
    myo_path = os.path.join(myo_dir, myo_filename)
    img = nib.load(image_path)
    myo = nib.load(myo_path)
    img = img.get_fdata()[:, :, 0]
    myo = myo.get_fdata()[:, :, 0]
    if np.sum(myo != 0) == 0:
        continue
    # Find the indices of the non-zero elements in the array
    indices = np.nonzero(myo)
    # Stack the indices together into an (n, 2) array
    points = np.column_stack(indices)
    # Calculate the centroid using the mean of the points
    centroid = points.mean(axis=0)
    print(centroid)
    x, y = int(np.floor(centroid[0])), int(np.floor(centroid[1]))
    img_cropped = img[x - 40:x + 40, y - 40:y + 40]
    plt.imshow(img_cropped)
    plt.show()
    # Save the resampled image to the output directory
    output_path = os.path.join(image_output_dir, image_filename)
    img_cropped = sitk.GetImageFromArray(img_cropped)
    sitk.WriteImage(img_cropped, output_path)