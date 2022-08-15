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
from monai.losses.dice import DiceLoss, one_hot, DiceFocalLoss 
from monai.losses import TverskyLoss
from monai.metrics import DiceMetric
from monai.losses import *
import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter
from losses.focal_tversky import FocalTversky
from torch.nn import functional as F 
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import *
import copy
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
        ds = CycleGANDataset(opt.data_root,is_train=True,transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
        train, val = train_test_split(ds, test_size=0.1, random_state=42)
        print(f"Train:{len(train)}")
        print(f"Val:{len(val)}")
        # val_ds = CycleGANDataset(opt.data_root,is_train=False,transform = transforms.Compose([transforms.CenterCrop((174,174)),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
        dl = DataLoader(train, batch_size=opt.batch_size,shuffle=True)
        val_dl = DataLoader(val, batch_size=opt.batch_size,shuffle=True)
        print(f"Train dl:{len(dl)}")
        print(f"Val dl:{len(val_dl)}")

    return (dl,val_dl)



def adjust_learning_rate(args,optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        args.lr_decay_epochs = "5,10,20"
        steps = np.sum(epoch if epoch in np.asarray(args.lr_decay_epochs) else 0)
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
        

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

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        self.model = smp.Unet(encoder_name=opt.model_name, encoder_depth= 5, encoder_weights= None, decoder_use_batchnorm= True,in_channels = 1, classes= 3, activation= 'softmax')
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print(f'Model param: {pytorch_total_params}')
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
            path = './challenge/weights/{:s}_dice{:.4f}_temp{:s}.pt'.format(self.opt.model_name, val_dice, str(time.time())[-6:])
            if self.opt.multi_gpu == 'on':
                torch.save(self.model.module.state_dict(), path)
            else:
                torch.save(self.model.state_dict(), path)
            print(f'Saved model: {path}')
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
        os.rename(self.records['checkpoints'][-1], './challenge/weights/{:s}_dice{:.4f}_save{:s}.pt'.format(self.opt.model_name, self.records['best_dice'], timestamp))
        for path in self.records['checkpoints'][0:-1]:
            os.remove(path)

        report = '\t'.join(['val_dice', 'train_loss', 'val_loss', 'best_epoch', 'timestamp'])
        report += "\n{:.4f}\t{:.4f}\t{:.4f}\t{:d}\t{:s}\n{:s}".format(self.records['best_dice'], min(self.records['train_loss']), min(self.records['val_loss']), self.records['best_epoch'], timestamp, self.info)
        with open('./logs/{:s}_log.txt'.format(timestamp), 'w') as f:
            f.write(report)
        print('report saved:', './logs/{:s}_log.txt'.format(timestamp))
    
    def _train(self, train_dataloader, criterion, optimizer,opt,arr):
        self.model.train()
        arr = []
        train_loss, n_total, n_batch = 0, 0, len(train_dataloader)
        for i_batch, sample_batched in enumerate(train_dataloader):
            inputs, target = sample_batched['image'].float().to(self.opt.device), sample_batched[label_str].long().to(self.opt.device) # .long()
            predict = self.model(inputs)
            #target = target.squeeze()
            # torch.set_printoptions(profile="full")
            # print(f"_> {predict}")
            # print(f"_> {torch.sum(predict, dim=(1,2,3))}")
            arr.append(torch.sum(predict, dim=(1,2)))
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
   
            sys.stdout.flush()
        print()
        batch_result = torch.cat(arr)
        return train_loss / n_total, batch_result
    
    def _evaluation(self, val_dataloader, criterion):
        self.model.eval()
        val_loss, val_dice, n_total, dice_metric,vs_dice,c_dice = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for sample_batched in val_dataloader:
                inputs, target = sample_batched['image'].float().to(self.opt.device), sample_batched[label_str].long().to(self.opt.device)
                predict = self.model(inputs)
                #target = target.squeeze()
                loss = criterion(predict, target)       
                loss_for_metric = DiceLoss(include_background=False,to_onehot_y=True)
                dice_metric = 1 - loss_for_metric(predict, target)
                # etc = torchmetrics.functional.dice(torch.argmax(predict, dim=1).unsqueeze(dim=1), target,average = 'none',ignore_index=0,num_classes=2, zero_division=0 )
                # print(etc)

                
                target = target.squeeze()
                target = F.one_hot(target, num_classes=3)
                target = target.permute(0, 3, 1,2)
                predict = torch.argmax(predict, dim=1)
                predict = F.one_hot(predict, num_classes=3)
                predict = predict.permute(0, 3, 1,2)


                metric = DiceMetric(include_background=False,reduction='none')
                dice_with_nan = metric(y_pred = predict,y = target).cpu().numpy()
                dice = np.nan_to_num(dice_with_nan).mean(axis=0)
                vs_dice +=  dice[0]
                c_dice +=  dice[1]


                val_loss += loss.item() * len(sample_batched)
                val_dice += dice_metric.item() * len(sample_batched)
                n_total += len(sample_batched)
            

        return val_loss / n_total, val_dice / n_total, vs_dice / n_total, c_dice / n_total
    
    def run(self):
        folder_counter = sum([len(folder) for r, d, folder in os.walk(opt.tb_path)])
        print(f'Version: {folder_counter}')
        writer = SummaryWriter(f'{opt.tb_path}/{opt.dataset}-{folder_counter}_{opt.epochs}')
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)
        #optimizer = torch.optim.SGD(_params, lr=self.opt.lr,weight_decay=1e-4,momentum=0.9)
        #criterion = BCELoss2d()#MULTICLASS_MODE
        criterion = DiceLoss(include_background=False,to_onehot_y=True)
        #criterion = FocalTversky()

        #criterion = TverskyLoss(include_background=False, to_onehot_y=True)
        # criterion = DiceFocalLoss(include_background=False, to_onehot_y=True)
        # dist_mat = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.5], [1.0, 0.5, 0.0]], dtype=np.float32)
        # criterion = GeneralizedWassersteinDiceLoss(dist_matrix=dist_mat)

        #criterion = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        #criterion = torch.nn.CrossEntropyLoss(ignore_index=1)
        dl, val_dl = set_loaders(opt)
        #gen_ab = i_t_i_translation()
        self._reset_records()
        # scheduler1 = ExponentialLR(optimizer, gamma=0.9)
        # scheduler1 = CosineAnnealingLR(optimizer,T_max=10, eta_min=0)
        patience = 30
        vs_dice, c_dice = 0,0
        best_model, best_epoch, best_dev_acc = None, 0, -np.inf
        batch_res = []
        for epoch in range(self.opt.epochs):
            #adjust_learning_rate(opt, optimizer, epoch)
            train_loss, batch_res = self._train(dl, criterion, optimizer,opt,batch_res)
            # torch.set_printoptions(profile="full")
            # print(f'=> Sample wis {batch_res}')
            # print(f'=> Sample wise sum {torch.sum(batch_res, dim=(1,2,3))}')
            # scheduler1.step()
            val_loss, val_dice, new_vs_dice, new_c_dice = self._evaluation(val_dl, criterion)
            if val_dice > best_dev_acc:
                best_epoch = epoch
                best_dev_acc = val_dice
                best_model = copy.deepcopy(self.model) 
                # We want to return the model from the best epoch, not from the last epoch

            if epoch - best_epoch > patience:
                # print(f"==> last pred {print(torch.argmax(predict, dim=1).sum(dim=0))}")
                break


            vs_dice = np.max([vs_dice, new_vs_dice])
            c_dice = np.max([c_dice, new_c_dice])
            # self._update_records(epoch, train_loss, val_loss, val_dice)
            print('{:d}/{:d} > train loss: {:.4f}, val loss: {:.4f}, dice score: {:.4f}, vs dice: {:.4f}, cochlea dice: {:.4f}'.format(epoch+1, self.opt.epochs, train_loss, val_loss, val_dice, vs_dice, c_dice))


            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar("Val Loss", val_loss,epoch)
            writer.add_scalar("Dice metric", val_dice,epoch)
            writer.add_scalar("VS Dice", vs_dice,epoch)
            writer.add_scalar("Cochlea Dice", c_dice,epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        path = './challenge/weights/{:s}_dice{:.4f}_best{:s}.pt'.format(self.opt.model_name, val_dice, str(time.time())[-6:])
        if self.opt.multi_gpu == 'on':
            torch.save(best_model.module.state_dict(), path)
        else:
            torch.save(best_model.state_dict(), path)
        print(f'Saved model with Early Stopping: {path}')

        # self._draw_records()

    def dev_eval(self,opt):
        opt.checkpoint = 'resnet34_no_0008_dice0.7728_best039548.pt'
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dl, val_dl = set_loaders(opt)
        model = smp.Unet(encoder_name=opt.model_name, encoder_depth= 5, encoder_weights= None, decoder_use_batchnorm= True,in_channels = 1, classes= 3, activation= 'softmax')
        model.load_state_dict(torch.load('challenge/weights/{:s}'.format(opt.checkpoint), map_location=opt.device))
        model = model.to(opt.device)
        model.eval()
        print('checkpoint {:s} has been loaded'.format(opt.checkpoint))
        batch_mat = []
        for idx, data in enumerate(dl):
            inputs = data['image'].float().cuda()
            # inputs = inputs.unsqueeze(0)
            predict = model(inputs)
            print(predict.shape)
            # 90, 3, 174 ,174
            batch_mat.append(torch.argmax(predict, dim=1).squeeze())
        batch_result = torch.cat(batch_mat, dim=0)
        print(batch_result.shape)

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
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,100,150',
                        help='where to decay lr, can be a list')
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--use_bilinear', default=False, type=float)
    ''' For environment '''
    parser.add_argument('--backend', default=False, type=bool)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--prefetch', default=False, type=bool)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--multi_gpu', default=None, type=str, help='on, off')
    opt = parser.parse_args()
    
    opt.model_name = 'resnet34'
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device(opt.device) if opt.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.multi_gpu = opt.multi_gpu if opt.multi_gpu else 'on' if torch.cuda.device_count() > 1 else 'off'
    if opt.dataset == 'cat':
        label_str = "mask"
        opt.lr = 0.0001
        
    else:
        label_str = "label"
        # opt.batch_size = 32
    
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
    #ins.dev_eval(opt)
    ins.run()