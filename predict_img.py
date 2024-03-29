import os
import sys
import time
from unittest import result
import torch
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from models.unet import UNet
import torch.nn as nn
from datasets.cyclegan import CycleGANDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from uvcgan.torch.funcs import get_torch_device_smart, seed_everything
from uvcgan.cgan import construct_model
from uvcgan.config import Args
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import monai
import nibabel as nib
from utils.helper import i_t_i_translation
class BCELoss2d(nn.Module):
    
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        return self.bce_loss(predict, target)

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
    
    def predict_in_image_format(self, train_dataloader,gen_ab):
        train_loss, n_total, n_batch = 0, 0, len(train_dataloader)
        gen_ab.eval()
        for i_batch, sample_batched in enumerate(train_dataloader):
            inputs, fname = sample_batched['image'].to(self.opt.device), sample_batched['file_name']
            with torch.no_grad():
                features = gen_ab(inputs)
            
            file = features.detach().cpu().numpy()
            file_save = file.squeeze()
            print(np.array(file_save).shape)
            plt.imsave(f"~/data/crossmoda2022_training/nifti_syn/{fname[0]}", np.array(file_save), cmap='gray')  

        return 0
    
    def predict_syn_imgs(self):
        ds = CycleGANDataset('~/data/crossmoda2022_training/',is_train=True,transform = transforms.Compose([transforms.CenterCrop((224,224)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
        dl = DataLoader(ds, batch_size=opt.batch_size,shuffle=False)
        gen_ab = i_t_i_translation()
        train_loss = self.predict_in_image_format(dl,gen_ab)
        print("Finish..")

    def predict(self):
        type = "ceT1"
        base_path = '~/data/crossmoda2022_training'
        input_dir = f"{base_path}/training_source"
        output_dir = f"{base_path}/nifti_syn"
        complete_input_folder = sorted(os.listdir(input_dir))
        gen_ab = i_t_i_translation()
        gen_ab.eval()
        for fname in tqdm(complete_input_folder):
            if not fname.endswith(f"{type}.nii.gz"):
                continue


            n_file = os.path.join(input_dir, fname)
            # res = nib.load(f'{n_file}') # 512,512,120 # vertical MRI
            # np_arr = res.get_fdata() 
            inputImage = sitk.ReadImage(n_file) # 120,512,512 # horizontal MRI
            np_arr =sitk.GetArrayFromImage(inputImage).astype(np.float32)
            tensor = torch.tensor(np_arr).float()#.permute(2,0,1)
            resize = 256    
            # resize_fn = transforms.Resize((resize, resize))
            resize_fn = monai.transforms.Resize((resize, resize))
            image = resize_fn(tensor)
            cc = transforms.CenterCrop((224,224))
            image = cc(image)

            batch = image.unsqueeze(1)
            # 120,1,244,244
            with torch.no_grad():
                features = gen_ab(batch)
            feature_sq = features.detach().cpu().squeeze(1)
            result_image = sitk.GetImageFromArray(feature_sq)
            sitk.WriteImage(result_image, f'{output_dir}/{fname}.nii.gz')
        print("Finish..")

    def transform_labels(self):
        print("Transforming labels")
        type = "Label"
        base_path = '~/data/crossmoda2022_training'
        input_dir = f"{base_path}/training_source"
        output_dir = f"{base_path}/rs_cc_nifti_labels"
        complete_input_folder = sorted(os.listdir(input_dir))
        for fname in tqdm(complete_input_folder):
            if not fname.endswith(f"{type}.nii.gz"):
                continue
            n_file = os.path.join(input_dir, fname)
            # res = nib.load(f'{n_file}') # 512,512,120 # vertical MRI
            # np_arr = res.get_fdata() 
            inputImage = sitk.ReadImage(n_file) # 120,512,512 # horizontal MRI
            np_arr =sitk.GetArrayFromImage(inputImage).astype(np.float32)
            tensor = torch.tensor(np_arr).float()#.permute(2,0,1)
            resize = 256    
            # resize_fn = transforms.Resize((resize, resize))
            # print(tensor.shape)
            resize_fn = monai.transforms.Resize((resize, resize))
            image = resize_fn(tensor)
            cc = transforms.CenterCrop((224,224))
            cropped_img = cc(image)

            cropped_img[(cropped_img > 0) & (cropped_img < 1)] = 1
            cropped_img[cropped_img > 1] = 2

            result_image = sitk.GetImageFromArray(cropped_img)
            sitk.WriteImage(result_image, f'{output_dir}/{fname}.nii.gz')
        print("Finish..")
    
    
    

if __name__ == '__main__':
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,    # default lr=0.01
        'adam': torch.optim.Adam,          # default lr=0.001
        'adamax': torch.optim.Adamax,      # default lr=0.002
        'asgd': torch.optim.ASGD,          # default lr=0.01
        'rmsprop': torch.optim.RMSprop,    # default lr=0.01
        'sgd': torch.optim.SGD,            # default lr=0.1
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    ''' For dataset '''
    parser.add_argument('--impath', default='shoe_dataset', type=str)
    parser.add_argument('--imsize', default=256, type=int)
    parser.add_argument('--aug_prob', default=0.5, type=float)
    ''' For training '''
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--use_bilinear', default=False, type=float)
    ''' For environment '''
    parser.add_argument('--backend', default=False, type=bool)
    parser.add_argument('--prefetch', default=False, type=bool)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--multi_gpu', default=None, type=str, help='on, off')
    opt = parser.parse_args()
    
    opt.model_name = 'unet_bilinear' if opt.use_bilinear else 'unet'
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device(opt.device) if opt.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.multi_gpu = opt.multi_gpu if opt.multi_gpu else 'on' if torch.cuda.device_count() > 1 else 'off'
    
    opt.impaths = {
        'train': os.path.join('.', opt.impath, 'train'),
        'val': os.path.join('.', opt.impath, 'val'),
        'test': os.path.join('.', opt.impath, 'test'),
        'btrain': os.path.join('.', opt.impath, 'bg', 'train'),
        'bval': os.path.join('.', opt.impath, 'bg', 'val')
    }
    
    for folder in ['figs', 'logs', 'state_dict', 'predicts']:
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    if opt.backend: # Disable the matplotlib window
        mpl.use('Agg')
    
    ins = Instructor(opt)
    # get us sythetic T2 given ceT1
    # ins.predict()
    # get us synthetic T2 in graycale img form (not viable)
    # ins.predict_syn_imgs()
    # nifti to nifti labels (compatible to images e.g cropped and resized) transformation
    ins.transform_labels()