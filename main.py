from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import sys
sys.path.append('./vision')
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data','-d', metavar='DATA', default='cub',
                    help='dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--save_path', '-s', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--is_fix', dest='is_fix', action='store_true',
                    help='is_fix.')
                    
''' opt '''
parser.add_argument('--opt', metavar='OPT', default='adam', help='opt type')
                                   
''' loss params '''
parser.add_argument('--w_rec', dest='w_rec', default=5, type=float,
                    help='loss weight for L_rec.')
parser.add_argument('--alpha', dest='alpha', default=0.01, type=float,
                    help='loss weight for L_rec.')
                    
best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ''' random seed '''
    if args.seed is not None:
        random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)
        
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    print('==> random seed:',args.seed)
    

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
                                
    ''' data load info '''
    data_info = h5py.File(os.path.join('./data',args.data,'data_info.h5'), 'r')
    img_path = str(data_info['img_path'][...])
    nc = data_info['all_att'][...].shape[0]
    sf_size = data_info['all_att'][...].shape[1]
    semantic_data = {'seen_class':data_info['seen_class'][...],
                     'unseen_class': data_info['unseen_class'][...],
                     'all_class':np.arange(nc),
                     'all_att': data_info['all_att'][...]}
    
    ''' create model '''
    lws = [args.w_rec,args.alpha]
    params = {'num_classes':nc,'is_fix':args.is_fix, 'sf_size':sf_size}
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model,criterion = models.__dict__[args.arch](lws=lws,pretrained=True,params=params)
    else:
        print("=> creating model '{}'".format(args.arch))
        model,criterion = models.__dict__[args.arch](lws=lws,pretrained=False,params=params)
    print("=> is the backbone fixed: '{}'".format(args.is_fix))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = criterion.cuda(args.gpu)

    ''' optimizer '''
    if args.opt=='adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                betas=(0.5,0.999),
                                weight_decay=args.weight_decay)
    elif args.opt=='sgd':
      optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                     args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ''' optionally resume from a checkpoint '''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            if(best_prec1==0):
                best_prec1 = checkpoint['best_prec1']
            print('=> pretrained acc {:.4F}'.format(best_prec1))
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ''' Data loading code '''
    traindir = os.path.join('./data',args.data,'train.list')
    valdir1 = os.path.join('./data',args.data,'test_seen.list')
    valdir2 = os.path.join('./data',args.data,'test_unseen.list')

    train_transforms, val_transforms = preprocess_strategy(args.data)

    train_dataset = datasets.ImageFolder(img_path,traindir,train_transforms)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader1 = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir1, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    val_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir2, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
	
    ''' model training'''
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args.lr)
	
        # train for one epoch
        train(args, train_loader, semantic_data, model, criterion, optimizer, epoch,is_fix=args.is_fix)
        
        # evaluate on validation set
        prec1 = validate(args,val_loader1,val_loader2, semantic_data, model)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save model
        if args.is_fix:
            save_path = os.path.join(args.save_path,'fix.model')
        else:
            save_path = os.path.join(args.save_path,args.arch+('_{:.4f}.model').format(best_prec1))
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                #'optimizer' : optimizer.state_dict(),
            },filename=save_path)
            print('saving!!!!')

def train(args,train_loader, semantic_data, model, criterion, optimizer, epoch, is_fix):
    ''' load semantic data'''
    seen_c = semantic_data['seen_class']
    unseen_c = semantic_data['unseen_class']
    all_sf = semantic_data['all_att']
    sf =  torch.from_numpy(all_sf).cuda(args.gpu,non_blocking=True)

    # switch to train mode
    model.train()
    if(is_fix):
        freeze_bn(model) 

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        ''' load data & processing'''
        # data proc
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        data_time = time.time() - end
        
        ''' compute output '''
        end = time.time()
        emb,logits = model(input,sf)
        model_time = time.time() - end
            
        ''' loss '''
        loss_set = criterion(target,sf,emb,logits,seen_c,unseen_c)
        
        ''' update '''
        optimizer.zero_grad()
        loss_set[0].backward()
        optimizer.step()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Data Time {data_time:.3f} \t'
                  'Model Time {model_time:.3f}\t'.format(
                   epoch, i, len(train_loader), data_time=data_time,
                   model_time=model_time))

            for l in loss_set:
                print('{:.4f} '.format(l.item()), end='')
            print('')

def validate(args,val_loader1, val_loader2, semantic_data, model):
    ''' switch to evaluate mode '''
    model.eval()

    ''' load semantic data'''
    seen_c = semantic_data['seen_class']
    unseen_c = semantic_data['unseen_class']
    all_c = semantic_data['all_class']
    all_sf = torch.from_numpy(semantic_data['all_att']).cuda(args.gpu,non_blocking=True)
    
    ''' validation 1 & 2'''
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader1):
            # data proc
            if input.dim()==5:
                [_,_,_,height,width] = input.size()
                input = torch.reshape(input,[-1,3,height,width])
                target = torch.reshape(target,[-1])
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
			
            # compute embedding dict
            if(i==0):
                emb,logits = model(input,all_sf)
                seen_emb=emb[1].cpu().numpy();seen_emb=seen_emb[seen_c,...]
                unseen_emb=emb[2].cpu().numpy();unseen_emb=unseen_emb[unseen_c,...]
            
            # inference
            emb,logits = model(input,all_sf)
            emb_v2s = emb[0].cpu().numpy()
            cls = logits[0].cpu().numpy()
			
            # evaluation
            if(i==0):
                gt_s = target.cpu().numpy()
                rank_s_S= get_RANK(emb_v2s, seen_emb , seen_c)
                score_s = softmax(cls)
            else:
                gt_s = np.hstack([gt_s,target.cpu().numpy()])
                pre = get_RANK(emb_v2s, seen_emb, seen_c)
                rank_s_S = np.hstack([rank_s_S,pre])
                score_s = np.vstack([score_s,softmax(cls)])

        for i, (input, target) in enumerate(val_loader2):
            # data proc
            if input.dim()==5:
                [_,_,_,height,width] = input.size()
                input = torch.reshape(input,[-1,3,height,width])
                target = torch.reshape(target,[-1])
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # inference
            emb,logits = model(input,all_sf)
            emb_v2s = emb[0].cpu().numpy()
            cls = logits[0].cpu().numpy()
			
            if(i==0):
                gt_t = target.cpu().numpy()
                rank_t_T= get_RANK(emb_v2s, unseen_emb , unseen_c)
                score_t = softmax(cls)
            else:
                gt_t = np.hstack([gt_t,target.cpu().numpy()])
                pre= get_RANK(emb_v2s, unseen_emb , unseen_c)
                rank_t_T = np.hstack([rank_t_T,pre])
                score_t = np.vstack([score_t,softmax(cls)])
        
        
        SS = compute_class_accuracy_total(gt_s, rank_s_S,seen_c)
        UU = compute_class_accuracy_total(gt_t, rank_t_T,unseen_c)
        
        H_opt,S_opt,U_opt,tau_opt = opt_pre(rank_s_S,rank_t_T,score_s,score_t,gt_s, gt_t, seen_c,unseen_c)
	
        print(' SS: {:.4f} UU: {:.4f} ST: {:.4f} UT: {:.4f} H: {:.4f} tau: {:.4f}'
              .format(SS,UU,S_opt,U_opt,H_opt,tau_opt))
              

    return H_opt

if __name__ == '__main__':
    main()

