from uvcgan.config import Args
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
from uvcgan.models.generator import construct_generator
from uvcgan.torch.funcs       import get_torch_device_smart, seed_everything
import os
from datasets.segmentation import SegModel
import pytorch_lightning as pl
from uvcgan.cgan import construct_model

class ImageTranslation:
    def __init__(self):
        device = get_torch_device_smart()
        args   = Args.load('/dss/dsshome1/lxc09/ra49tad2/crossmoda-challenge/uvcgan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/')
        config = args.config
        model = construct_model(
        args.savedir, args.config, is_train = False, device = device
        )
        for m in model.models:
          m = torch.nn.DataParallel(m)
        # ckpt = torch.load(os.path.join('/dss/dsshome1/lxc09/ra49tad2/crossmoda-challenge/uvcgan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/net_gen_ab.pth'))
        # state_dict = ckpt

        epoch = -1

        if epoch == -1:
            epoch = max(model.find_last_checkpoint_epoch(), 0)

        print("Load checkpoint at epoch %s" % epoch)

        seed_everything(args.config.seed)
        model.load(epoch)
        # gen_ab = model.models.gen_ab
        # gen_ab.eval()


if __name__ == "__main__":
    #gen_ab = ImageTranslation()




    ######
    
    ds = CycleGANDataset('/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/',is_train=True,transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.CenterCrop((224,224)),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
    val_ds = CycleGANDataset('/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/',is_train=False,transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.CenterCrop((224,224)),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)

    dl = DataLoader(ds, batch_size=110,shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=110,shuffle=False)

    
    model = SegModel("unet", "resnet34", in_channels=3, out_classes=1)
    trainer = pl.Trainer(
    gpus=8, 
    max_epochs=5,
    accelerator='dp',
    )

    trainer.fit(
        model, 
        train_dataloader=dl, 
        val_dataloaders=val_dl,
    )
