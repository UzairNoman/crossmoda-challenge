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
from preprocessing import nifti_to_2d_slices
import nibabel as nib
# input_dir = '/input/'
# path_img = os.path.join(input_dir,'{}_hrT2.nii.gz')
# path_pred = '/output/{}_Label.nii.gz'

# list_case = [k.split('_hrT2')[0] for k in os.listdir(input_dir)]

data_root= "/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/"



if __name__ == '__main__':

    input_dir = r"crossmoda2022_subset_validation"
    output_dir = r"val_subset"
    nifti_out = r"nifti_output"
    full_input_dir =  os.path.join(data_root, input_dir)
    full_output_dir = os.path.join(data_root, output_dir)
    nifti_output =  os.path.join(data_root, nifti_out)
    axis = 2
    do_filter = False
    resize = 256
    os.makedirs(full_output_dir, exist_ok=True)
    complete_input_folder = sorted(os.listdir(full_input_dir))
    data_info = nifti_to_2d_slices(full_input_dir, full_output_dir, axis, do_filter, resize,folder=complete_input_folder,type="hrT2")
    transform = transforms.Pad(41)

    opt = {}
    opt["checkpoint"] = 'unet_dice0.8417_save1660349507.pt'
    opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_ds = CycleGANDataset(data_root,is_test=True,transform = transforms.Compose([transforms.CenterCrop((174,174)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
    model = UNet(n_channels=1, n_classes=3, bilinear=False)
    model.load_state_dict(torch.load('./state_dict/{:s}'.format(opt['checkpoint']), map_location=opt['device']))
    model = torch.nn.DataParallel(model) 
    model = model.to(opt['device'])
    model.eval()
    print('checkpoint {:s} has been loaded'.format(opt['checkpoint']))
    batch_mat = []
    counter = 0
    test_dataloader = DataLoader(dataset=test_ds, batch_size=90, shuffle=False,drop_last=False)
    for idx, data in enumerate(test_dataloader):
        inputs = data['image'].float()
        # inputs = inputs.unsqueeze(0)
        predict = model(inputs)
        print(torch.argmax(torch.softmax(predict), dim=1))
        # 90, 3, 174 ,174
        batch_mat.append(torch.argmax(predict, dim=1).squeeze())
        # if(counter == dim):
        #     counter = 0
        #     result = torch.stack(single_img, dim=2)
        #     print(result.shape)
        #     batch_mat = []
    batch_result = torch.cat(batch_mat, dim=0)
    name_arr = data_info[0]
    x = 0
    for i in range(len(name_arr)):
        y = data_info[1][i] + x - 1
        image = batch_result[x:y,:,:]
        image = transform(image).cpu().numpy()
        x = y
        ni_img = nib.Nifti1Image(image, affine=data_info[2][i],dtype="int32")
        nib.save(ni_img, f'{nifti_output}/{data_info[0][i]}')




# for case in list_case:
#     t2_img = sitk.ReadImage(path_img.format(case))

#     ##
#     # your logic here. Below we do binary thresholding as a demo
#     ##

#     # using SimpleITK to do binary thresholding between 100 - 10000
#     vs_pred = sitk.BinaryThreshold(t2_img, lowerThreshold=400, upperThreshold=500)
#     cochlea_pred = sitk.BinaryThreshold(t2_img, lowerThreshold=900, upperThreshold=1100)

#     result = vs_pred + 2*cochlea_pred

#     # save the segmentation mask
#     sitk.WriteImage(result, path_pred.format(case))


