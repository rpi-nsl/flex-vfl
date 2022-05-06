#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020 Tongzhou Wang
import argparse
import builtins
import os
import random
import shutil
import time
import socket
import warnings
import pickle
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils
import moco.loader
from server_model import serverModel
from custom_dataset import MultiViewDataSet

class SplitImageTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        out = []
        for transform in self.transforms:
            out.append(transform(x))
        return out

model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_un', '--unsupervised-learning-rate', default=10.0, type=float,
                    metavar='LR', help='initial learning rate for final linear layer', dest='lr_un')
#parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
#                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--schedule', default=[], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpus', default=None, nargs='+', type=int,
                    help='GPU id(s) to use. Default is all visible GPUs.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--num_clients', default=12, type=int,
                    help='number of clients')
parser.add_argument('--mode', default="flex", type=str,
        help='VFL algorithm to use: flex, sync, vafl, or pbcd')
parser.add_argument('--labeled_frac', default=1.0, type=float,
                    help='fraction of training data that is labeled')
#parser.add_argument('--local_epochs', default=1, type=float,
#                    help='Number of local iterations')
parser.add_argument('--server_time', default=1.0, type=float,
                    help='How long roundtrip server communication takes')

args = parser.parse_args()

def main():

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    save_folder_terms = [
        f'MVCNN',
        f'b{args.batch_size}',
        f'lr{args.lr:g}',
        f'mode{args.mode}',
        f'st{args.server_time}',
        f'seed{args.seed}',
        f'e{",".join(map(str, args.schedule))},200',
    ]

    args.save_folder = os.path.join('results', '_'.join(save_folder_terms))
    os.makedirs(args.save_folder, exist_ok=True)
    print(f"save_folder: '{args.save_folder}'")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.gpus is None:
        args.gpus = list(range(torch.cuda.device_count()))

    if args.multiprocessing_distributed and len(args.gpus) == 1:
        warnings.warn('You have chosen to use multiprocessing distributed '
                      'training. But only one GPU is available on this node. '
                      'The training will start within the launching process '
                      'instead to minimize process start overhead.')
        args.multiprocessing_distributed = False

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        # Assuming we have len(args.gpus) processes per node, we need to adjust
        # the total world_size accordingly
        args.world_size = len(args.gpus) * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=len(args.gpus), args=(args,))
    else:
        # Simply call main_worker function
        main_worker(0, args)


def main_worker(index, args):
    # We will do a bunch of `setattr`s such that
    #
    # args.rank               the global rank of this process in distributed training
    # args.index              the process index to this node
    # args.gpus               the GPU ids for this node
    # args.gpu                the default GPU id for this node
    # args.batch_size         the batch size for this process
    # args.workers            the data loader workers for this process
    # args.seed               if not None, the seed for this specific process, computed as `args.seed + args.rank`

    args.index = index
    args.gpu = args.gpus[index]
    assert args.gpu is not None
    torch.cuda.set_device(args.gpu)

    # suppress printing for all but one device per node
    if args.multiprocessing_distributed and args.index != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    print(f"Use GPU(s): {args.gpus} for training on '{socket.gethostname()}'")

    # init distributed training if needed
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            ngpus_per_node = len(args.gpus)
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + index
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size and data
            # loader workers based on the total number of GPUs we have.
            assert args.batch_size % ngpus_per_node == 0
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0

    if args.seed is not None:
        args.seed = args.seed + args.rank
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = True

    # build data loaders before initializing model, since we need num_classes for the latter
    train_loader, val_loader, classes = create_data_loaders(args)

    # Create models
    models = []
    optimizers = []
    for m in range(args.num_clients+1):
        # create model
        if m != args.num_clients:
            print(f"=> creating model '{args.arch}' with {len(classes)} classes")
            model = torchvision.models.__dict__[args.arch](num_classes=128)

        else:
            print(f"=> creating server model")
            model = serverModel(num_clients=args.num_clients, num_classes=len(classes))
            print("Number of classes:",len(classes))
            # init the fc layer
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()

        model.cuda(args.gpu)
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.multiprocessing_distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpus)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=args.gpus)
            else:
                model = torch.nn.DataParallel(model, device_ids=args.gpus)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)

        models.append(model)
        optimizers.append(optimizer)

    best_acc1 = 0
    args.start_epoch = 0
    train_loss = []
    train_acc1 = []
    train_acc5 = []
    test_loss = []
    test_acc1 = []
    test_acc5 = []

    # optionally resume from a checkpoint
    for client in range(args.num_clients+1):
        save_filename = os.path.join(args.save_folder, f"client{client}.pth.tar")
        if os.path.isfile(save_filename):
            print("=> loading checkpoint '{}'".format(save_filename))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(save_filename, map_location=torch.device('cuda', args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if isinstance(best_acc1, torch.Tensor):
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            models[client].load_state_dict(checkpoint['state_dict'])
            optimizers[client].load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            train_loss = pickle.load(open(os.path.join(args.save_folder,'train_loss.pkl'), 'rb'))
            train_acc1 = pickle.load(open(os.path.join(args.save_folder,'train_acc1.pkl'), 'rb'))
            train_acc5 = pickle.load(open(os.path.join(args.save_folder,'train_acc5.pkl'), 'rb'))
            test_loss = pickle.load(open(os.path.join(args.save_folder,'test_loss.pkl'), 'rb'))
            test_acc1 = pickle.load(open(os.path.join(args.save_folder,'test_acc1.pkl'), 'rb'))
            test_acc5 = pickle.load(open(os.path.join(args.save_folder,'test_acc5.pkl'), 'rb'))

    if args.start_epoch == 0:
        loss, acc1, acc5 = validate(val_loader, models, criterion, args)
        test_loss.append(loss)
        test_acc1.append(acc1)
        test_acc5.append(acc5)


    # Main training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        for m in range(args.num_clients+1):
            lr = args.lr
            adjust_learning_rate(optimizers[m], epoch, lr)

        # train for one epoch
        if args.mode != "vafl" and args.mode != "vafl2" and args.mode != "vafl3":
            loss, acc1, acc5 = train(train_loader, models, criterion, optimizers, epoch, args)
        else:
            # Load embeddings
            embeddings = []
            for i, (images, _) in enumerate(train_loader):
                embeddings.append([])
                for client in range(args.num_clients):
                    images[client] = images[client].cuda(args.gpu, non_blocking=True)

                # compute inital embeddings 
                for client in range(args.num_clients):
                    image_local = images[client]
                    with torch.no_grad():
                        embeddings[i].append(models[client](image_local))
            loss, acc1, acc5 = train_vafl(train_loader, models, criterion, optimizers, epoch, args, embeddings)
        train_loss.append(loss)
        train_acc1.append(acc1)
        train_acc5.append(acc5)

        # evaluate on validation set
        loss, acc1, acc5 = validate(val_loader, models, criterion, args)
        test_loss.append(loss)
        test_acc1.append(acc1)
        test_acc5.append(acc5)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            print(f"New best Acc1 {best_acc1:.4f}")

        if (args.distributed and args.rank == 0) or (args.index == 0):
            pickle.dump(train_loss, open(os.path.join(args.save_folder,'train_loss.pkl'), 'wb'))
            pickle.dump(train_acc1, open(os.path.join(args.save_folder,'train_acc1.pkl'), 'wb'))
            pickle.dump(train_acc5, open(os.path.join(args.save_folder,'train_acc5.pkl'), 'wb'))
            pickle.dump(test_loss, open(os.path.join(args.save_folder,'test_loss.pkl'), 'wb'))
            pickle.dump(test_acc1, open(os.path.join(args.save_folder,'test_acc1.pkl'), 'wb'))
            pickle.dump(test_acc5, open(os.path.join(args.save_folder,'test_acc5.pkl'), 'wb'))

            for client in range(args.num_clients+1):
                # Reset optimizers
                optimizers[client] = torch.optim.SGD(models[client].parameters(), args.lr, momentum=args.momentum)

                # Save client models
                save_filename = os.path.join(args.save_folder, f"client{client}.pth.tar")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': models[client].state_dict(),
                    'best_acc1': best_acc1,
                    'acc1': acc1,
                    'acc5': acc5,
                    'optimizer' : optimizers[client].state_dict(),
                }, is_best, save_filename)
                print(f"saved to '{save_filename}'")
                #if epoch == args.start_epoch:
                #    sanity_check(model.state_dict(), args.pretrained)


def create_data_loaders(args):
    # Data loading code
    traindir = args.data
    valdir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = MultiViewDataSet(traindir, 'train',
            transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        MultiViewDataSet(traindir, 'test',
                transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
            ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes 

def train_vafl(train_loader, models, criterion, optimizers, epoch, args, embeddings):
    # Train asynchronous VFL

    batch_time = utils.AverageMeter('Time', '6.3f')
    data_time = utils.AverageMeter('Data', '6.3f')
    losses = utils.AverageMeter('Loss', '.4e')
    top1 = utils.AverageMeter('Acc1', '6.2f')
    top5 = utils.AverageMeter('Acc5', '6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, utils.ProgressMeter.BR, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    wait = []
    for client in range(args.num_clients+1):
        models[client].train()
        wait.append(0.0)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        for client in range(args.num_clients):
            images[client] = images[client].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # Train clients and server for Q rounds
        for client in range(args.num_clients):
            image_local = images[client]
            if wait[client] < 1.0:
                optimizers[client] = torch.optim.SGD(models[client].parameters(), args.lr, momentum=args.momentum)
                optimizers[client].zero_grad()
                optimizers[-1].zero_grad()

                embedding_view = [client_view.detach().clone() for client_view in embeddings[i]]
                embedding_view[client] = models[client](image_local)
                output = models[-1](torch.cat(embedding_view,axis=1))

                # compute gradient and do SGD step
                loss = criterion(output, target)
                loss.backward()

                with torch.no_grad():
                    embeddings[i][client] = models[client](image_local)

                optimizers[client].step()
                optimizers[-1].step()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss, images[0].size(0))
                top1.update(acc1, images[0].size(0))
                top5.update(acc5, images[0].size(0))

                if args.mode == "vafl":
                    if client < 10:
                        wait[client] += (20/(client*2+1))+args.server_time 
                    else:
                        wait[client] += 1.0+args.server_time
                elif args.mode == "vafl2":
                    cpus = [0.3638092, 0.17014983, 0.14333789, 0.33265191, 0.17415424, 0.06804619, 0.14446286, 0.29251583, 0.2207424, 0.12800219, 0.27062089, 0.13972783, 0.30291598]
                    epochs = (20*(1-np.array(cpus))).astype(int)
                    wait[client] += (20/epochs[client]) + args.server_time
                elif args.mode == "vafl3":
                    if client < 3:
                        wait[client] += (20/5)+args.server_time 
                    elif client < 6:
                        wait[client] += (20/10)+args.server_time 
                    elif client < 9:
                        wait[client] += (20/15)+args.server_time 
                    else:
                        wait[client] += 1.0+args.server_time
            else:
                wait[client] -= 1.0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    loss, acc1, acc5 = losses.avg, top1.avg, top5.avg
    print(f'Training * Loss {loss:.5f} Acc1 {acc1:.3f} Acc5 {acc5:.3f}')
    return loss, acc1, acc5



def train(train_loader, models, criterion, optimizers, epoch, args):
    batch_time = utils.AverageMeter('Time', '6.3f')
    data_time = utils.AverageMeter('Data', '6.3f')
    losses = utils.AverageMeter('Loss', '.4e')
    top1 = utils.AverageMeter('Acc1', '6.2f')
    top5 = utils.AverageMeter('Acc5', '6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, utils.ProgressMeter.BR, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    for client in range(args.num_clients+1):
        models[client].train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        for client in range(args.num_clients):
            images[client] = images[client].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute inital embeddings 
        embeddings = []
        for client in range(args.num_clients):
            image_local = images[client]
            with torch.no_grad():
                embeddings.append(models[client](image_local))

        # Number of local iterations chosen for each party
        # based on the algorithm of choice
        local_epochs = []
        if args.mode == 'sync': 
            for _ in range(args.num_clients+1):
                local_epochs.append(20)
        elif args.mode == 'sync2': 
            #for _ in range(args.num_clients+1):
            #    local_epochs.append(10)
            cpus = [0.3638092, 0.17014983, 0.14333789, 0.33265191, 0.17415424, 0.06804619, 0.14446286, 0.29251583, 0.2207424, 0.12800219, 0.27062089, 0.13972783, 0.30291598]
            epochs = (20*(1-torch.tensor(cpus))).type(torch.int)
            for _ in range(args.num_clients+1):
                local_epochs.append(torch.min(epochs))
        elif args.mode == 'sync3': 
            for _ in range(args.num_clients+1):
                local_epochs.append(5)
        elif args.mode == 'flex': 
            for i in range(10):
                local_epochs.append(i*2+1)
            local_epochs.append(20)
            local_epochs.append(20)
            local_epochs.append(20)
        elif args.mode == 'flex2': 
            #for i in range(6):
            #    local_epochs.append(10)
            #for i in range(6):
            #    local_epochs.append(20)
            #local_epochs.append(20)
            cpus = [0.3638092, 0.17014983, 0.14333789, 0.33265191, 0.17415424, 0.06804619, 0.14446286, 0.29251583, 0.2207424, 0.12800219, 0.27062089, 0.13972783, 0.30291598]
            local_epochs = (20*(1-torch.tensor(cpus))).type(torch.int)
        elif args.mode == 'flex3': 
            for i in range(3):
                local_epochs.append(5)
            for i in range(3):
                local_epochs.append(10)
            for i in range(3):
                local_epochs.append(15)
            for i in range(3):
                local_epochs.append(20)
            local_epochs.append(20)
        elif args.mode == 'pbcd': 
            for _ in range(args.num_clients+1):
                local_epochs.append(1)
        else:
            print("Invalid algorithm chosen:", args.mode)
            return None
        # Train clients and server for Q rounds
        for client in range(args.num_clients+1):
            optimizers[client] = torch.optim.SGD(models[client].parameters(), args.lr, momentum=args.momentum)
            adjust_learning_rate(optimizers[client], epoch, args.lr)
            for q in range(local_epochs[client]):
                if client != args.num_clients:
                    image_local = images[client]
                    embedding_view = embeddings.copy()
                    embedding_view[client] = models[client](image_local)
                else:
                    embedding_view = embeddings
                output = models[-1](torch.cat(embedding_view,axis=1))

                # compute gradient and do SGD step
                loss = criterion(output, target)

                optimizers[client].zero_grad()
                loss.backward()
                optimizers[client].step()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss, images[0].size(0))
                top1.update(acc1, images[0].size(0))
                top5.update(acc5, images[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    loss, acc1, acc5 = losses.avg, top1.avg, top5.avg
    print(f'Training * Loss {loss:.5f} Acc1 {acc1:.3f} Acc5 {acc5:.3f}')
    return loss, acc1, acc5


def validate(val_loader, models, criterion, args):
    # Get accuracy of models on data in val_loader

    batch_time = utils.AverageMeter('Time', '6.3f')
    losses = utils.AverageMeter('Loss', '.4e')
    top1 = utils.AverageMeter('Acc1', '6.2f')
    top5 = utils.AverageMeter('Acc5', '6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [batch_time, losses, utils.ProgressMeter.BR, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    for i in range(args.num_clients+1):
        models[i].eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            for client in range(args.num_clients):
                images[client] = images[client].cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            embeddings = []
            for i in range(args.num_clients):
                image_local = images[i]
                embeddings.append(models[i](image_local))
            output = models[-1](torch.cat(embeddings,axis=1))
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss, images[0].size(0))
            top1.update(acc1, images[0].size(0))
            top5.update(acc5, images[0].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    loss, acc1, acc5 = losses.avg, top1.avg, top5.avg
    print(f'Test * Loss {loss:.5f} Acc1 {acc1:.3f} Acc5 {acc5:.3f}')

    return loss, acc1, acc5


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.split(filename)[0], 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr):
    """Decay the learning rate based on schedule"""
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
