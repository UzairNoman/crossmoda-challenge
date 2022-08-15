import os
import sys
import time
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

def i_t_i_translation():
    
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
        gen_ab = model.models.gen_ab
        gen_ab.eval()
        return gen_ab.cuda()

class BCELoss2d(nn.Module):
    
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        return self.bce_loss(predict, target)

def dice_coeff(predict, target):
    smooth = 0.001
    batch_size = predict.size(0)
    predict = (predict > 0.5).float()
    m1 = predict.view(batch_size, -1)
    m2 = target.view(batch_size, -1)
    intersection = (m1 * m2).sum(-1)
    return ((2.0 * intersection + smooth) / (m1.sum(-1) + m2.sum(-1) + smooth)).mean()

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
    
    def _train(self, train_dataloader,gen_ab):
        train_loss, n_total, n_batch = 0, 0, len(train_dataloader)
        gen_ab.eval()
        for i_batch, sample_batched in enumerate(train_dataloader):
            inputs, target, fname = sample_batched['image'].to(self.opt.device), sample_batched['label'].to(self.opt.device), sample_batched['file_name']
            with torch.no_grad():
                features = gen_ab(inputs)
            
            file = features.detach().cpu().numpy()
            file_save = file.squeeze()
            plt.imsave(f"/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/training_source_syn/{fname[0]}", np.array(file_save), cmap='gray')  

        return 0
    
    
    def run(self):
        ds = CycleGANDataset('/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/',is_train=True,transform = transforms.Compose([transforms.CenterCrop((224,224)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
        #val_ds = CycleGANDataset('/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/',is_train=False,transform = transforms.Compose([transforms.CenterCrop((224,224)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
        dl = DataLoader(ds, batch_size=opt.batch_size,shuffle=False)
        gen_ab = i_t_i_translation()
        train_loss = self._train(dl,gen_ab)
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
    ''' For inference '''
    parser.add_argument('--inference', default=False, type=bool)
    parser.add_argument('--use_crf', default=False, type=bool)
    parser.add_argument('--checkpoint', default=None, type=str)
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
    # if opt.inference:
    #     ins.inference()
    # else:
    ins.run()