from pickletools import optimize
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
import segmentation_models_pytorch as smp
import argparse
import math
import tensorboard_logger as tb_logger

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='50,100,150',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='crossmoda',
                        choices=['cifar10', 'cifar100', 'svhn', 'isic'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # temperature
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../../DATA/'
    opt.model_path = './saved_models/{}_models_seg'.format(opt.dataset)
    opt.tb_path = 'logs/{}_models_seg'.format(opt.dataset)
    opt.save_folder = opt.model_path
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    return opt






def i_t_i_translation():
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
        gen_ab = model.models.gen_ab
        gen_ab.eval()
        return gen_ab.cuda()

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



        

def set_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.0001)


def train(train_loader, model, segmentation, criterion, optimizer, epoch,opt):
    """one epoch training"""
    model.eval()
    segmentation.train()

 
    for idx, batch in enumerate(train_loader):

        images = batch['image'].float().cuda(non_blocking=True)
        labels = batch['label'].cuda(non_blocking=True)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model(images)
        output = segmentation(features.detach())
        loss = criterion(output, labels)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = output.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), labels.long(), mode="binary")

    return {
        "loss": loss,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

# def validate(val_loader, model, classifier, criterion, opt):
#     """validation"""
#     model.eval()
#     classifier.eval()

#     with torch.no_grad():
#         end = time.time()
#         for idx, (images, labels) in enumerate(val_loader):
#             images = images.float().cuda()
#             labels = labels.cuda()
#             bsz = labels.shape[0]

#             # forward
#             output = classifier(model.encoder(images))
#             loss = criterion(output, labels)

#             # update metric
#             losses.update(loss.item(), bsz)
#             acc1, acc5 = accuracy(output, labels, topk=(1, 5))
#             top1.update(acc1[0], bsz)

#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if idx % opt.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                     idx, len(val_loader), batch_time=batch_time,
#                     loss=losses, top1=top1))

#     print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
#     return losses.avg, top1.avg

def main():
    opt = parse_option()
    ds = CycleGANDataset('/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/',is_train=True,transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.CenterCrop((224,224)),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)
    val_ds = CycleGANDataset('/dss/dsshome1/lxc09/ra49tad2/data/crossmoda2022_training/',is_train=False,transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.CenterCrop((224,224)),transforms.ToTensor()])) # transforms.Normalize(0.0085,0.2753)

    dl = DataLoader(ds, batch_size=110,shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=110,shuffle=False)

    gen_ab = i_t_i_translation()
    segmentation = SegModel("unet", "resnet34", in_channels=3, out_classes=1)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = set_optimizer(segmentation.cuda())

    

    for epoch in range(opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)
        trained_model = train(dl,gen_ab,segmentation,criterion,optimizer,epoch,opt)

        logger = tb_logger.Logger(logdir=opt.tb_path, flush_secs=2)
        logger.log_value('loss', trained_model['loss'], epoch)
        logger.log_value('tp', trained_model['tp'], epoch)
        logger.log_value('fp', trained_model['fp'], epoch)
        logger.log_value('fn', trained_model['fn'], epoch)
        logger.log_value('tn', trained_model['tn'], epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        print(trained_model)

        # loss, val_acc = validate(val_dl, model, classifier, criterion, opt)
        # if val_acc > best_acc:
        #     best_acc = val_acc
























if __name__ == "__main__":
    main()