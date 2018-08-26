# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:

Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

__author__ = "Pau Rodríguez López, ISELAB, CVC-UAB"
__email__ = "pau.rodri1@gmail.com"

import argparse
import os
import json
import torch
from torch.nn import init
import torch.nn.functional as F
import torchvision.datasets as dset
from ai_challenger import Ai_Challenger
import torchvision.transforms as transforms
from models.model import CifarResNeXt
from models.resnext_101_32x4d import resnext_101 as resnext101
from models.resnext_50_32x4d import resnext_50 as resnext50
from models.densenet import densenet161
import numpy as np     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ResNeXt on ai_challenger', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('data_path', type=str, help='Root for the ai_challenger dataset.')
    parser.add_argument('dataset', type=str, choices=['ai_challenger'], help='Choose ai_challenger.')
    parser.add_argument('net', type=str, choices=['resnext50', 'resnext101', 'densenet161'], help='Choose training model.')
    # Optimization options
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001, help='The Learning Rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=60)
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 60, 100, 150],
                        help='Decrease learning rate at these epochs.') # 28 use the same lr(1e-4) for part1 and part2 
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./snapshots', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--device', type=int, default=-1, help='Set a device.')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # Display
    parser.add_argument('--display', type=int, default=50, help='Display the loss')
    # i/o
    parser.add_argument('--log', type=str, default='./logs', help='Log folder.')
    args = parser.parse_args()

    if args.device >= 0:
        torch.cuda.set_device(args.device)

    # Init logger
    if not os.path.isdir(args.log):
        os.makedirs(args.log)
    if not os.path.isdir(os.path.join(args.log, args.net+'_'+str(args.resize))):
        os.makedirs(os.path.join(args.log, args.net+'_'+str(args.resize)))

    log = open(os.path.join(args.log, args.net+'_'+str(args.resize), 'log.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    #log.write(json.dumps(state) + '\n')

    # Calculate number of epochs wrt batch size
    #args.epochs = args.epochs * 128 // args.batch_size
    #args.schedule = [x * 128 // args.batch_size for x in args.schedule]

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)


    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    crop_size = 224 if args.resize == 256 else 336
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(crop_size), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.CenterCrop(crop_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    transform1 = transforms.Compose([transforms.CenterCrop(crop_size), transforms.Normalize(mean, std),transforms.ToTensor()])
    print 'Loading data...'
    train_data = Ai_Challenger(args.data_path, train='train', resize=args.resize, transform=train_transform, download=False)
    val_data = Ai_Challenger(args.data_path, train='val', resize=args.resize, transform=test_transform, download=False)
    nlabels = 80
    #if args.dataset == 'cifar10':
    #    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    #    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    #    nlabels = 10
    #else:
    #    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    #    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    #    nlabels = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.prefetch, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)
    print 'Loading data done.'
    # Init checkpoints
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(os.path.join(args.save, args.net+'_'+str(args.resize))):
        os.makedirs(os.path.join(args.save, args.net+'_'+str(args.resize)))

    print 'Loading model...'
    # Init model, criterion, and optimizer
    state_dict = torch.load(args.load) if args.load != '' else None
    if args.net == 'resnext50':
        net = resnext50(state_dict)
    elif args.net == 'resnext101':
        net = resnext101(state_dict)
    elif args.net == 'densenet161':
        net = densenet161(state_dict)
    print 'Loading model done.'

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()

    optimizer = torch.optim.SGD([{'params': net.features.parameters(), 'lr':0.},
        {'params': net.classifier.parameters(), 'lr': state['learning_rate']}], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    # train function (forward, backward, update)
    def train():
        net.train()
        loss_avg = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.2 + loss.data[0] * 0.8
            if batch_idx % args.display == 0:
                print 'batch id: {}, loss: {}, avg_loss: {}'.format(batch_idx, loss.data[0], loss_avg)
        state['train_loss'] = loss_avg


    # test function (forward only)
    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        correct_t3 = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = torch.autograd.Variable(data.cuda()), torch.autograd.Variable(target.cuda())

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            topk = output.data.topk(3, dim=1)[1]
            N = target.data.shape[0]
            target_reshape = target.data.view(N, 1)
            target_enlarge = torch.cat((target_reshape, target_reshape, target_reshape), 1)
            
            #target_tmp = torch.from_numpy(np.tile(
            #    np.array(target.data.numpy()).reshape(N, 1), (1, 3)))
            correct += pred.eq(target.data).sum()
            correct_t3 += topk.eq(target_enlarge).sum()

            # test loss average
            loss_avg += loss.data[0]

        state['test_loss'] = loss_avg / len(val_loader)
        state['test_accuracy'] = correct / len(val_loader.dataset)
        state['test_accuracy_t3'] = correct_t3 / len(val_loader.dataset)
        #print state['test_accuracy_t3']

    # Main loop
    shift = 38
    best_accuracy = 0.0
    for epoch in range(args.epochs-shift):
        print 'Epoch {}'.format(epoch + shift)
        if (epoch + shift) in args.schedule:
            state['learning_rate'] *= args.gamma
            #optimizer.param_groups[1]['lr'] = state['learning_rate']
            #if (epoch + shift) >= args.epochs * 0.5:
                #optimizer.param_groups[0]['lr'] = state['learning_rate']
            #for param_group in optimizer.param_groups:
            #   param_group['lr'] = state['learning_rate']
        optimizer.param_groups[1]['lr'] = state['learning_rate']
        optimizer.param_groups[0]['lr'] = state['learning_rate']
        state['epoch'] = epoch + shift
        train()
        test()
        if state['test_accuracy_t3'] > best_accuracy:
            best_accuracy = state['test_accuracy_t3']
            torch.save(net.state_dict(), os.path.join(args.save, args.net+'_'+str(args.resize), 'ckpt_epoch_{}.pytorch'.format(epoch+shift)))
        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        print("Best top_3 accuracy: %f" % best_accuracy)

    log.close()
