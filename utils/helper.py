from torchvision.datasets.folder import default_loader
import PIL
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import nibabel as nib
import torch 
from PIL import Image
import os
from uvcgan.torch.funcs import get_torch_device_smart, seed_everything
from uvcgan.cgan import construct_model
from uvcgan.config import Args
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
    # print(f'->>>{affine_matrix}')
    return torch.from_numpy(data_array)

def dir_empty(dir_path):
    return not any((True for _ in os.scandir(dir_path)))

def renaming_hrT2_to_lbls(directory):
    
    for file in os.listdir(directory):
        
        new_name = file[:file.find('.nii.gz')]+".nii.gz"
        new_name = new_name.replace("hrT2","Label")
        os.rename(os.path.join(directory,file),os.path.join(directory,new_name))
        print(f"Renamed to {new_name}")

def check_modality(filename):
    """
    check for the existence of modality
    return False if modality is not found else True
    """
    end = filename.find('.nii.gz')
    modality = filename[end-4:end]
    for mod in modality: 
        if not(ord(mod)>=48 and ord(mod)<=57): #if not in 0 to 9 digits
            return False
    return True

def rename_for_single_modality(directory):
    
    for file in os.listdir(directory):
        
        if check_modality(file)==False:
            new_name = file[:file.find('.nii.gz')]+"_0000.nii.gz"
            os.rename(os.path.join(directory,file),os.path.join(directory,new_name))
            print(f"Renamed to {new_name}")
        else:
            print(f"Modality present: {file}")

def calc_dataset_mean_std(dl):
    mean = 0.0
    meansq = 0.0
    count = 0
    for index, data in enumerate(dl):
        #mean= reduce((lambda x, y: x + y), data)
        mean = data.sum()
        #inter = reduce((lambda x, y: x**2 + y), data)
        meansq = meansq + (data**2).sum()
        count += np.prod(data.shape)
    total_mean = mean/count
    total_var = (meansq/count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    return (total_mean, total_var, total_std)

def i_t_i_translation():
    
        device = get_torch_device_smart()
        args   = Args.load('~/crossmoda-challenge/uvcgan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/')
        config = args.config
        model = construct_model(
        args.savedir, args.config, is_train = False, device = device
        )
 
        epoch = 200

        if epoch == -1:
            epoch = max(model.find_last_checkpoint_epoch(), 0)

        print("Load checkpoint at epoch %s" % epoch)

        seed_everything(args.config.seed)
        model.load(epoch)
        gen_ab = model.models.gen_ab
        gen_ab.eval()
        return gen_ab.cuda()
