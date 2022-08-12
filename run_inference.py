#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK as sitk
import os
from datasets.cyclegan import CycleGANDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from models.unet import UNet
import torch
# input_dir = '/input/'
# path_img = os.path.join(input_dir,'{}_hrT2.nii.gz')
# path_pred = '/output/{}_Label.nii.gz'

# list_case = [k.split('_hrT2')[0] for k in os.listdir(input_dir)]

data_root= "/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/"



if __name__ == '__main__':



    opt = {}
    opt["checkpoint"] = 'unet_dice0.3378_save1660335644.pt'
    opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_ds = CycleGANDataset(data_root,is_test=True,transform = transforms.Compose([transforms.CenterCrop((174,174)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]),force_recreate= True) # transforms.Normalize(0.0085,0.2753)
    model = UNet(n_channels=1, n_classes=3, bilinear=False)
    model.load_state_dict(torch.load('./state_dict/{:s}'.format(opt['checkpoint']), map_location=opt['device']))
    model = torch.nn.DataParallel(model) 
    model = model.to(opt['device'])
    model.eval()
    print('checkpoint {:s} has been loaded'.format(opt['checkpoint']))
    single_img = []
    counter = 0
    for idx, data in enumerate(test_ds):
        inputs, index, fname, dim = data['image'].float(), data['label'], data['file_name'], data['dim']
        counter += 1
        inputs = inputs.unsqueeze(0)
        predict = model(inputs)
        single_img.append(predict)
        #print(idx % dim == dim - 1)
        if(counter == dim):
            counter = 0
            result = torch.stack(single_img, dim=2)
            print(result.shape)
            single_img = []
        # index, inputs = sample_batched['image'], sample_batched[label_str].to(self.opt.device)


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


