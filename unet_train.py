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
import segmentation_models_pytorch as smp
import numpy as np
import math
from utils.helper import save_img
from torchmetrics.functional import dice_score
# from torchgeometry.losses import DiceLoss
# from torchmetrics import Dice
from monai.losses.dice import DiceLoss, GeneralizedWassersteinDiceLoss, one_hot, DiceFocalLoss 
from monai.losses import *
import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
def set_loaders(opt):
    if opt.dataset == 'cat':
        # init train, val, test sets
        from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
        root = './data_files'
        train_dataset = SimpleOxfordPetDataset(root, "train")
        valid_dataset = SimpleOxfordPetDataset(root, "valid")
        test_dataset = SimpleOxfordPetDataset(root, "test")

        # It is a good practice to check datasets don`t intersects with each other
        assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
        assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
        assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

        print(f"Train size: {len(train_dataset)}")
        print(f"Valid size: {len(valid_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        dl = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dl = DataLoader(valid_dataset, batch_size=16, shuffle=False)
        test_dl = DataLoader(test_dataset, batch_size=16, shuffle=False)
    else:
        ds = CycleGANDataset(opt.data_root,is_train=True,transform = transforms.Compose([transforms.CenterCrop((174,174)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
        val_ds = CycleGANDataset(opt.data_root,is_train=False,transform = transforms.Compose([transforms.CenterCrop((174,174)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
        dl = DataLoader(ds, batch_size=opt.batch_size,shuffle=False)
        val_dl = DataLoader(val_ds, batch_size=opt.batch_size,shuffle=False)

    return (dl,val_dl)



def adjust_learning_rate(args,optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        # steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        # if steps > 0:
        #     lr = lr * (args.lr_decay_rate ** steps)
        """initial LR decayed by 10 every 30 epochs"""
        lr = lr * (0.1 ** (epoch // 30))
        

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def i_t_i_translation():
    
        device = get_torch_device_smart()
        args   = Args.load('/dss/dsshome1/lxc09/ra49tad2/uvcgan/outdir/selfie2anime/model_d(cyclegan)_m(cyclegan)_d(basic)_g(vit-unet)_cyclegan_vit-unet-12-none-lsgan-paper-cycle_high-256/')
        config = args.config
        model = construct_model(
        args.savedir, args.config, is_train = False, device = device
        )
        # for m in model.models:
        #   m = torch.nn.DataParallel(m)

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
# def dice_coeff(y_pred, y_true, epsilon=1e-6):
#     y_true_flatten = np.asarray(y_true.detach().cpu()).astype(np.bool)
#     y_pred_flatten = np.asarray(y_pred.detach().cpu()).astype(np.bool)

#     if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
#         return 1.0

#     return (2. * np.sum(y_true_flatten * y_pred_flatten)) /\
#            (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon)

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt    
        self.model = UNet(n_channels=1, n_classes=3, bilinear=self.opt.use_bilinear)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Model param: {pytorch_total_params}')
        if opt.checkpoint:
            self.model.load_state_dict(torch.load('./state_dict/{:s}'.format(opt.checkpoint), map_location=self.opt.device))
            print('checkpoint {:s} has been loaded'.format(opt.checkpoint))
        if opt.multi_gpu == 'on':
            self.model = torch.nn.DataParallel(self.model)   # 1,174,174 | 3,174,174
        self.model = self.model.to(opt.device)
        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        self.info = 'n_trainable_params: {0}, n_nontrainable_params: {1}\n'.format(n_trainable_params, n_nontrainable_params)
        self.info += 'training arguments:\n' + '\n'.join(['>>> {0}: {1}'.format(arg, getattr(self.opt, arg)) for arg in vars(self.opt)])
        if self.opt.device.type == 'cuda':
            print('cuda memory allocated:', torch.cuda.memory_allocated(opt.device.index))
        print(self.info)
    
    def _reset_records(self):
        self.records = {
            'best_epoch': 0,
            'best_dice': 0,
            'train_loss': list(),
            'val_loss': list(),
            'val_dice': list(),
            'checkpoints': list()
        }
    
    def _update_records(self, epoch, train_loss, val_loss, val_dice):
        if val_dice > self.records['best_dice']:
            path = './state_dict/{:s}_dice{:.4f}_temp{:s}.pt'.format(self.opt.model_name, val_dice, str(time.time())[-6:])
            if self.opt.multi_gpu == 'on':
                torch.save(self.model.module.state_dict(), path)
            else:
                torch.save(self.model.state_dict(), path)
            self.records['best_epoch'] = epoch
            self.records['best_dice'] = val_dice
            self.records['checkpoints'].append(path)
        self.records['train_loss'].append(train_loss)
        self.records['val_loss'].append(val_loss)
        self.records['val_dice'].append(val_dice)
    
    def _draw_records(self):
        timestamp = str(int(time.time()))
        print('best epoch: {:d}'.format(self.records['best_epoch']))
        print('best train loss: {:.4f}, best val loss: {:.4f}'.format(min(self.records['train_loss']), min(self.records['val_loss'])))
        print('best val dice {:.4f}'.format(self.records['best_dice']))
        os.rename(self.records['checkpoints'][-1], './state_dict/{:s}_dice{:.4f}_save{:s}.pt'.format(self.opt.model_name, self.records['best_dice'], timestamp))
        for path in self.records['checkpoints'][0:-1]:
            os.remove(path)
        # Draw figures
        # plt.figure()
        # trainloss, = plt.plot(self.records['train_loss'])
        # valloss, = plt.plot(self.records['val_loss'])
        # plt.legend([trainloss, valloss], ['train', 'val'], loc='upper right')
        # plt.title('{:s} loss curve'.format(timestamp))
        # plt.savefig('./figs/{:s}_loss.png'.format(timestamp), format='png', transparent=True, dpi=300)
        # plt.figure()
        # valdice, = plt.plot(self.records['val_dice'])
        # plt.title('{:s} dice curve'.format(timestamp))
        # plt.savefig('./figs/{:s}_dice.png'.format(timestamp), format='png', transparent=True, dpi=300)
        # Save report
        report = '\t'.join(['val_dice', 'train_loss', 'val_loss', 'best_epoch', 'timestamp'])
        report += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:d}\t{:s}\n{:s}".format(self.records['best_dice'], min(self.records['train_loss']), min(self.records['val_loss']), self.records['best_epoch'], timestamp, self.info)
        with open('./logs/{:s}_log.txt'.format(timestamp), 'w') as f:
            f.write(report)
        print('report saved:', './logs/{:s}_log.txt'.format(timestamp))
    
    def _train(self, train_dataloader, criterion, optimizer,opt):
        self.model.train()
        train_loss, n_total, n_batch = 0, 0, len(train_dataloader)
        for i_batch, sample_batched in enumerate(train_dataloader):
            inputs, target = sample_batched['image'].float().to(self.opt.device), sample_batched[label_str].long().to(self.opt.device) # .long()
            predict = self.model(inputs)
            #target = target.squeeze()

            # target_idx = target
            # target = one_hot(target_idx[:, None, ...], num_classes=3)
            # print(target.shape)
            # target = target.requires_grad_().to(self.opt.device)
                

            optimizer.zero_grad()
            
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(sample_batched)
            n_total += len(sample_batched)
   
            ratio = int((i_batch+1)*50/n_batch)
            sys.stdout.write("\r["+">"*ratio+" "*(50-ratio)+"] {}/{} {:.2f}%".format(i_batch+1, n_batch, (i_batch+1)*100/n_batch))
            sys.stdout.flush()
        print()
        return train_loss / n_total
    
    def _evaluation(self, val_dataloader, criterion):
        self.model.eval()
        val_loss, val_dice, n_total = 0, 0, 0
        with torch.no_grad():
            for sample_batched in val_dataloader:
                inputs, target = sample_batched['image'].float().to(self.opt.device), sample_batched[label_str].long().to(self.opt.device)
                predict = self.model(inputs)
                #target = target.squeeze()
                loss = criterion(predict, target)
                dice = 1 - loss
                #dice = dice_coeff(predict, target)
                #dice = dice_score(predict, target)
                val_loss += loss.item() * len(sample_batched)
                val_dice += dice.item() * len(sample_batched)
                n_total += len(sample_batched)

        return val_loss / n_total, val_dice / n_total
    
    def run(self):
        folder_counter = sum([len(folder) for r, d, folder in os.walk(opt.tb_path)])
        print(f'Version: {folder_counter}')
        writer = SummaryWriter(f'{opt.tb_path}/{opt.dataset}-{folder_counter}_{opt.epochs}')
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        #optimizer = torch.optim.SGD(_params, lr=self.opt.lr,weight_decay=1e-4,momentum=0.9)
        #criterion = BCELoss2d()#MULTICLASS_MODE
        criterion = DiceLoss(to_onehot_y=True)
        #criterion = TverskyLoss(include_background=False, to_onehot_y=True)
        # dist_mat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]], dtype=np.float32)
        # criterion = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)

        #criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        #criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
        dl, val_dl = set_loaders(opt)
        #gen_ab = i_t_i_translation()
        self._reset_records()
        for epoch in range(self.opt.epochs):
            adjust_learning_rate(opt, optimizer, epoch)
            train_loss = self._train(dl, criterion, optimizer,opt)
            val_loss, val_dice = self._evaluation(val_dl, criterion)
            self._update_records(epoch, train_loss, val_loss, val_dice)
            print('{:d}/{:d} > train loss: {:.4f}, val loss: {:.4f}, val dice: {:.4f}'.format(epoch+1, self.opt.epochs, train_loss, val_loss, val_dice))

            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar("Val Loss", val_loss,epoch)
            writer.add_scalar("Val Dice", val_dice,epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        self._draw_records()
    
    def inference(self):
        test_ds = CycleGANDataset(opt.data_root,is_test=True,transform = transforms.Compose([transforms.CenterCrop((174,174)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
        #dl = DataLoader(test_ds, batch_size=opt.batch_size,shuffle=False)
        test_dataloader = DataLoader(dataset=test_ds, batch_size=1, shuffle=False)
        n_batch = len(test_dataloader)
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(test_dataloader):
                inputs, index, fname = sample_batched['image'].float().to(self.opt.device), sample_batched[label_str].long().to(self.opt.device), sample_batched['file_name']
                # index, inputs = sample_batched['image'], sample_batched[label_str].to(self.opt.device)
                predict = self.model(inputs)
                # [1, 3, 174, 174]
                #save_img(index.item(), predict, fname)
                ratio = int((i_batch+1)*50/n_batch)
                sys.stdout.write("\r["+">"*ratio+" "*(50-ratio)+"] {}/{} {:.2f}%".format(i_batch+1, n_batch, (i_batch+1)*100/n_batch))
                sys.stdout.flush()
        print()
    

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
    parser.add_argument('--dataset', type=str, default='crossmoda',
                        choices=['crossmoda','cat'], help='dataset')
    parser.add_argument('--imsize', default=256, type=int)
    parser.add_argument('--aug_prob', default=0.5, type=float)
    ''' For training '''
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,100,150',
                        help='where to decay lr, can be a list')
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--use_bilinear', default=False, type=float)
    ''' For inference '''
    parser.add_argument('--inference', default=False, type=bool)
    parser.add_argument('--use_crf', default=False, type=bool)
    parser.add_argument('--checkpoint', default=None, type=str)
    ''' For environment '''
    parser.add_argument('--backend', default=False, type=bool)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--prefetch', default=False, type=bool)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--multi_gpu', default=None, type=str, help='on, off')
    opt = parser.parse_args()
    
    opt.model_name = 'unet_bilinear' if opt.use_bilinear else 'unet'
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device(opt.device) if opt.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.multi_gpu = opt.multi_gpu if opt.multi_gpu else 'on' if torch.cuda.device_count() > 1 else 'off'
    if opt.dataset == 'cat':
        label_str = "mask"
        opt.lr = 0.0001
        
    else:
        label_str = "label"
        opt.batch_size = 32
    
    opt.impaths = {
        'train': os.path.join('.', opt.impath, 'train'),
        'val': os.path.join('.', opt.impath, 'val'),
        'test': os.path.join('.', opt.impath, 'test'),
        'btrain': os.path.join('.', opt.impath, 'bg', 'train'),
        'bval': os.path.join('.', opt.impath, 'bg', 'val')
    }
    opt.tb_path = 'logs/{}_models_seg'.format(opt.dataset)

    repo_root = os.path.abspath(os.getcwd())
    opt.data_root = os.path.join(repo_root, "../data/crossmoda2022_training/")


    for folder in ['figs', 'logs', 'state_dict', 'predicts']:
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    if opt.backend: # Disable the matplotlib window
        mpl.use('Agg')
    
    ins = Instructor(opt)
    if opt.inference:
        ins.inference()
    else:
        ins.run()