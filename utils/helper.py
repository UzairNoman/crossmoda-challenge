from torchvision.datasets.folder import default_loader
import PIL
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import nibabel as nib
import torch 
from PIL import Image
import os
def sample_image(images, index, prg, randomize = False):
    if randomize:
        return prg.choice(images, size = 1)[0]

    if index >= len(images):
        return None

    return images[index]

def apply_if_not_none(fn, x):
    if x is None:
        return None

    return fn(x)

def load_images(paths, transform = None, label = False):
    result = apply_if_not_none(default_loader, paths[0])
    # if label:
    #     to_tensor = transforms.ToTensor()
    #     result = to_tensor(result)
    # else:
    #result = np.array(result)[:, :, 0]
    #     gs = transforms.Grayscale(num_output_channels=1)
    #     result = gs(result)
        # print(np.array(result))
        # print(f'=>>{np.array(result).shape}')

    if transform is not None:
        result = apply_if_not_none(transform, result) 
    return result

def draw_image(image):
    return PIL.Image.fromarray((np.array(image) * 255).astype(np.uint8))

def save_img(index, predict, fname):
    predict = predict.squeeze().cpu().numpy()
    predict = np.array((predict > 0.5) * 255).astype(np.uint8)
    mask = Image.fromarray(predict, mode='L')
    hw = (174,174)
    mask = mask.resize(hw)
    fg = Image.new('RGB', hw, (0, 0, 0))
    bg = Image.new('RGB', hw, (255, 255, 255))
    bg.paste(fg, mask=mask)
    bg.save(f'./predicts/{fname}')

def read_nii_file(path):
    nifti = nib.load(path)
    data_array = nifti.get_data()
    affine_matrix = nifti.affine
    return torch.from_numpy(data_array)

def dir_empty(dir_path):
    return not any((True for _ in os.scandir(dir_path)))