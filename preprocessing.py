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
from torchvision.transforms.functional import center_crop

def nifti_to_2d_slices(input_folder: str, output_folder: str, axis: int, filtered, resize, folder,type="ceT1",start=0,end=0):
    all_data = [[],[],[]]
    file_count = 0
    for fname in tqdm(folder):
        
        if not fname.endswith(f"{type}.nii.gz"):
            continue

        n_file = os.path.join(input_folder, fname)
        nifti = nib.load(n_file)

        np_data = nifti.get_fdata()
        np_data = np_data.astype(np.float16)
        # np_data.shape (512, 512, 120)

        f_basename = fname.split(".")[0]
        #np_data.shape[axis]
        for i in range(start,np_data.shape[axis] - end):
            

            image = np_data[:,:,i]
            # image.shape (512, 512)

            if resize:
                tr = monai.transforms.Resize((resize, resize))
                image = tr(image[None])[0]
            
            if(type == 'Label'):         
                image[(image > 0) & (image < 1)] = 1
                image[image > 1] = 2
            
                image = center_crop(torch.tensor(image),174).numpy()

            if filtered:
                brain_mask = image > 0
                if brain_mask.sum() < 4000:
                    continue
            #image = PIL.Image.fromarray((np.array(image) * 255).astype(np.uint8))
            fp = os.path.join(output_folder, f"{f_basename}_{i}")
            # for image
            if(type == 'Label'):
                fp = f"{fp}.npy"
                np.save(fp, image)
            else:
                fp = f"{fp}.jpeg"
                plt.imsave(fp, np.array(image), cmap='gray')
            
        all_data[0].append(fname)
        all_data[1].append(np_data.shape[axis])
        all_data[2].append(nifti.affine)

    return all_data

            #image.save(f"{fp}.jpeg")
            # for label

input_dir = r"training_source"
output_dir = r"label_npy"
axis = 2
do_filter = False
resize = 256
os.makedirs(output_dir, exist_ok=True)
if __name__ == "__main__":
    complete_input_folder = sorted(os.listdir(input_dir))
    train, val = train_test_split(complete_input_folder, test_size=0.2, random_state=42)
    nifti_to_2d_slices(input_dir, output_dir, axis, do_filter, resize,folder=complete_input_folder,type="ceT1")

