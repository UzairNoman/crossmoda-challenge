#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK as sitk
import os

input_dir = '/input/'
path_img = os.path.join(input_dir,'{}_hrT2.nii.gz')
path_pred = '/output/{}_Label.nii.gz'

list_case = [k.split('_hrT2')[0] for k in os.listdir(input_dir)]

for case in list_case:
    t2_img = sitk.ReadImage(path_img.format(case))

    ##
    # your logic here. Below we do binary thresholding as a demo
    ##

    # using SimpleITK to do binary thresholding between 100 - 10000
    vs_pred = sitk.BinaryThreshold(t2_img, lowerThreshold=400, upperThreshold=500)
    cochlea_pred = sitk.BinaryThreshold(t2_img, lowerThreshold=900, upperThreshold=1100)

    result = vs_pred + 2*cochlea_pred

    # save the segmentation mask
    sitk.WriteImage(result, path_pred.format(case))


