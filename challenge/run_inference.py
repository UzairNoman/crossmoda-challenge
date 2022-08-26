#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import SimpleITK as sitk
import os
from datasets.cyclegan import CycleGANDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from models.unet import UNet
import torch
import numpy as np
import segmentation_models_pytorch as smp
from preprocessing import nifti_to_2d_slices
import nibabel as nib
from tqdm import tqdm

# input_dir = '/input/'
# path_img = os.path.join(input_dir,'{}_hrT2.nii.gz')
# path_pred = '/output/{}_Label.nii.gz'

# list_case = [k.split('_hrT2')[0] for k in os.listdir(input_dir)]

data_root= ""

def create_or_delete(mydir):
    if os.path.isdir(mydir):
        for f in os.listdir(mydir):
            if not f.endswith(".jpeg"):
                continue
            os.remove(os.path.join(mydir, f))
    else:
        os.makedirs(mydir, exist_ok=True)


if __name__ == '__main__':
    type = "hrT2"
    input_dir = "input/"
    syn_result = "syn_result"
    nifti_out = "output"
    full_input_dir =  os.path.join(data_root, input_dir)
    full_syn_result = os.path.join(data_root, syn_result)
    nifti_output =  os.path.join(data_root, nifti_out)
    axis = 2
    do_filter = False
    os.makedirs(full_syn_result, exist_ok=True)
    create_or_delete(syn_result)

    if not os.path.exists(nifti_output):
        os.mkdir(nifti_output)
    complete_input_folder = sorted(os.listdir(full_input_dir))
    data_info = nifti_to_2d_slices(full_input_dir, full_syn_result, axis, do_filter,folder=complete_input_folder,type=type)

    opt = {}
    opt["checkpoint"] = 'resnet34_fcl_no_dice0.7776_best162757.pt'
    opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt["model_name"] = 'resnet34'
    print(f"Device: {opt['device']}")
    test_ds = CycleGANDataset(data_root,is_test=True,transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
    model = smp.Unet(encoder_name=opt['model_name'], encoder_depth= 5, encoder_weights= None, decoder_use_batchnorm= True,in_channels = 1, classes= 3, activation= 'softmax')
    model.load_state_dict(torch.load('weights/{:s}'.format(opt['checkpoint']), map_location=opt['device']))
    model = torch.nn.DataParallel(model) 
    model = model.to(opt['device'])
    model.eval()
    print('checkpoint {:s} has been loaded'.format(opt['checkpoint']))
    batch_mat = []
    counter = 0
    test_dataloader = DataLoader(dataset=test_ds, batch_size=2, shuffle=False,drop_last=False)
    for idx, data in enumerate(tqdm(test_dataloader)):
        inputs = data['image'].float()
        # inputs = inputs.unsqueeze(0)
        predict = model(inputs)
        # 90, 3, 174 ,174
        batch_mat.append(torch.argmax(predict, dim=1).squeeze()) 

    batch_result = torch.cat(batch_mat, dim=0)
    print(batch_result[batch_result > 0]) # this shouldnt be zero (checked: argmax is making it zero otherwise append simple predict has values)
    name_arr = data_info[0]
    x = 0
    for i in range(len(name_arr)):
        y = data_info[1][i] + x
        image = batch_result[x:y,:,:]
        x = y
        ni_img = nib.Nifti1Image(image, affine=data_info[2][i],dtype="int32")
        print(data_info[0][i])
        final_output = data_info[0][i].replace(type, "Label")
        nib.save(ni_img, f'{nifti_output}/{data_info[0][i]}')

    print("Finish..")