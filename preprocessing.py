import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
import monai
import torch
import PIL
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def nifti_to_2d_slices(input_folder: str, output_folder: str, axis: int, filtered, resize):
    complete_input_folder = sorted(os.listdir(input_folder))
    train, val = train_test_split(complete_input_folder, test_size=0.2, random_state=42)
    for fname in tqdm(train):

        if not fname.endswith("Label.nii.gz"):
            continue

        n_file = os.path.join(input_folder, fname)
        nifti = nib.load(n_file)

        np_data = nifti.get_fdata()
        np_data = np_data.astype(np.float16)
        # np_data.shape (512, 512, 120)

        f_basename = fname.split(".")[0]
        #np_data.shape[axis]
        for i in range(5,np_data.shape[axis] - 5):
            slc = [slice(None)] * len(np_data.shape)
            
            slc[axis] = i
            image = np_data[slc]
            # image.shape (512, 512)

            if resize:
                tr = monai.transforms.Resize((resize, resize))
                image = tr(image[None])[0]
                print("Here is original")
                print(image)

                image[image > 0] = 1
                print("Here is operated")
                print(image)

            if filtered:
                brain_mask = image > 0
                if brain_mask.sum() < 4000:
                    continue
            #image = PIL.Image.fromarray((np.array(image) * 255).astype(np.uint8))
            #print(image)
            fp = os.path.join(output_folder, f"{f_basename}_{i}")
            plt.imsave(f"{fp}.jpeg", np.array(image), cmap='gray')
            #image.save(f"{fp}.jpeg")
            #np.save(os.path.join(output_folder, f"{f_basename}_{i}.jpeg"), image)


input_dir = r"data\training_source"
output_dir = r"data\label_01"
axis = 2
do_filter = False
resize = 256
os.makedirs(output_dir, exist_ok=True)
if __name__ == "__main__":
    nifti_to_2d_slices(input_dir, output_dir, axis, do_filter, resize)

