from uvcgan.uvcgan.config import Args
import torch
import fnmatch
import os
import random
import shutil
import string
import time
from abc import abstractmethod
from collections import defaultdict
from time import sleep

import numpy as np
import monai
from monai.transforms import AddChannel, Compose, Resize, ScaleIntensity, ToTensor
from torch.utils.data import DataLoader, Dataset
from datasets.cyclegan import CycleGANDataset
import torchvision.transforms as transforms
from uvcgan.uvcgan.models.generator import construct_generator
from uvcgan.uvcgan.torch.funcs       import get_torch_device_smart, seed_everything
import os
from datasets.segmentation import SegModel
import pytorch_lightning as pl
from uvcgan.uvcgan.cgan import construct_model

class ImageTranslation:
    def __init__(self):
        device = get_torch_device_smart()
        args   = Args.load('/dss/dsshome1/lxc09/ra49tad2/crossmoda-challenge/uvcgan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/')
        config = args.config
        model = construct_model(
        args.savedir, args.config, is_train = False, device = device
        )
        # for m in model.models:
        #   m = torch.nn.DataParallel(m)
        # ckpt = torch.load(os.path.join('/dss/dsshome1/lxc09/ra49tad2/crossmoda-challenge/uvcgan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/net_gen_ab.pth'))
        # state_dict = ckpt

        epoch = -1

        if epoch == -1:
            epoch = max(model.find_last_checkpoint_epoch(), 0)

        print("Load checkpoint at epoch %s" % epoch)

        seed_everything(args.config.seed)
        model.load(epoch)
        print("TES")
        # gen_ab = model.models.gen_ab
        # gen_ab.eval()


if __name__ == "__main__":
    #gen_ab = ImageTranslation()



    #os.environ['CUDA_VISIBLE_DEVICES'] = '8' # NO CUDA DEVICES FOUND
    ######
    
    ds = CycleGANDataset('/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/',is_train=True,transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.CenterCrop((224,224)),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
    val_ds = CycleGANDataset('/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/',is_train=False,transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.CenterCrop((224,224)),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)

    dl = DataLoader(ds, batch_size=110,shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=110,shuffle=False)

    
    model = SegModel("unet", "resnet34", in_channels=1, out_classes=1)
    trainer = pl.Trainer(
        gpus=2, 
        max_epochs=100,
        accelerator='cuda',
    )

    trainer.fit(
        model, 
        train_dataloaders=dl, 
        val_dataloaders=val_dl,
    )

    ## try ddp and ddp2 on plain seg and then add gen_ab = plain works but combined says cuda:1 ,6,3, LOCAL_RANK error on ddp2
    # without for loop it works still
    # specifying dp on plain seg = grad can be implicitly created only for scalar outputse
    # but after wraping it in DataParallel class in SegModel, it get back to cuda0,1 err
    # updated pl to 1.7 (myenv)

    # still same err with DataParallel nothing change on retrain, so training now with gpu=1 is still seems okay, loss is 0.9 as usuall
    # cat trainig note1.txt shows constant 0.4 each epoch but in local machine it is improving like hell 0.07 batch loss during the first epoch.
    #meaning: could be code problem as 0.416 is not good but on improving on local code.

    # next: convert binary_segment nb file to py and run that on cat data to check and cancel in half. Secondly, make change to it to take cet1 with labels and verify if improved



