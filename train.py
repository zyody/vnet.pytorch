#!/usr/bin/env python3
import os
import sys
import math
import pdb

from local import *
import time
import torch

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import torchbiomed.transforms as biotransforms
import torchbiomed.loss as bioloss
import torchbiomed.utils as utils

import SimpleITK as sitk

import shutil

import setproctitle

import vnet
import DataManager as DM
import promise12
import make_graph
from functools import reduce
import operator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)


def inference(params, args, loader, model):
    src = params['ModelParams']['dirInfer']
    dst = params['ModelParams']['dirResult']

    model.eval()
    # assume single GPU / batch size 1
    for batch_idx, data in enumerate(loader):
        data, id = data
        id = id[0]
        itk_img = sitk.ReadImage(os.path.join(src, id))
        origin = np.array(list(reversed(itk_img.GetOrigin())))
        spacing = np.array(list(reversed(itk_img.GetSpacing())))

        # pdb.set_trace()
        _, _, z, y, x = data.shape # need to subset shape of 3-d. by Chao.
        # convert names to batch tensor
        if args.cuda:
            data.pin_memory()
            data = data.cuda()
        with torch.no_grad():
            data = Variable(data)
        output = model(data)
        _, output = output.max(1)
        output = output.view((x, y, z))
        # pdb.set_trace()
        output = output.cpu()

        print("save {}".format(id))
        utils.save_updated_image(output, os.path.join(dst, id + "_predicted.mhd"), origin, spacing)

# performing post-train test:
# train.py --resume <model checkpoint> --i <input directory (*.mhd)> --save <output directory>

def noop(x):
    return x

def main(params, args):
    best_prec1 = 100. # accuracy? by Chao
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    resultDir = 'results/vnet.base.{}'.format(datestr())
    nll = True
    if args.dice:
        nll = False
    weight_decay = args.weight_decay
    setproctitle.setproctitle(resultDir)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    model = vnet.VNet(elu=False, nll=nll)
    batch_size = args.batchSz
    torch.cuda.set_device(0) # why do I have to add this line? It seems the below line is useless to apply GPU devices. By Chao.
    model = nn.parallel.DataParallel(model, device_ids=[0])

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    if nll:
        train = train_nll
        test = test_nll
    else:
        train = train_dice
        test = test_dice

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if args.cuda:
        model = model.cuda()

    if os.path.exists(resultDir):
        shutil.rmtree(resultDir)
    os.makedirs(resultDir, exist_ok=True)

    # transform
    trainTransform = transforms.Compose([
        transforms.ToTensor()
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    print("\nloading training set")
    dataManagerTrain = DM.DataManager(params['ModelParams']['dirTrain'],
                                          params['ModelParams']['dirResult'],
                                          params['DataManagerParams'])
    dataManagerTrain.loadTrainingData() # required
    numpyImages = dataManagerTrain.getNumpyImages()
    numpyGT = dataManagerTrain.getNumpyGT()

    trainSet = promise12.PROMISE12(mode='train', images=numpyImages, GT=numpyGT, transform=trainTransform)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)

    print("\nloading test set")
    dataManagerTest = DM.DataManager(params['ModelParams']['dirTest'],
                                      params['ModelParams']['dirResult'],
                                      params['DataManagerParams'])
    dataManagerTest.loadTestingData()  # required
    numpyImages = dataManagerTest.getNumpyImages()
    numpyGT = dataManagerTest.getNumpyGT()

    testSet = promise12.PROMISE12(mode='test', images=numpyImages, GT=numpyGT, transform=testTransform)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, **kwargs)

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    trainF = open(os.path.join(resultDir, 'train.csv'), 'w')
    testF = open(os.path.join(resultDir, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, model, trainLoader, optimizer, trainF)
        testDice = test(args, epoch, model, testLoader, optimizer, testF) # err is accuracy??? by Chao.
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1}, path=resultDir, prefix="vnet")
    os.system('./plot.py {} {} &'.format(len(trainLoader), resultDir))

    trainF.close()
    testF.close()

    # inference, i.e. output predicted mask for test data in .mhd
    if params['ModelParams']['dirInfer'] != '':
        print("loading inference data")
        dataManagerInfer = DM.DataManager(params['ModelParams']['dirInfer'],
                                          params['ModelParams']['dirResult'],
                                          params['DataManagerParams'])
        dataManagerInfer.loadInferData()  # required.  Create .loadInferData??? by Chao.
        numpyImages = dataManagerInfer.getNumpyImages()

        inferSet = promise12.PROMISE12(mode='infer', images=numpyImages, GT=None, transform=testTransform)
        inferLoader = DataLoader(inferSet, batch_size=batch_size, shuffle=True, **kwargs)
        inference(params, args, inferLoader, model)

# def train_nll(args, epoch, model, trainLoader, optimizer, trainF):
#     model.train()
#     nProcessed = 0
#     nTrain = len(trainLoader.dataset)
#     for batch_idx, output in enumerate(trainLoader):
#         data, target, id = output
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         target = target.view(target.numel())
#         loss = F.nll_loss(output, target)
#         dice_loss = bioloss.dice_error(output, target)
#         # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
#         loss.backward()
#         optimizer.step()
#         nProcessed += len(data)
#         pred = output.data.max(1)[1]  # get the index of the max log-probability
#         incorrect = pred.ne(target.data).cpu().sum()
#         err = 100.*incorrect/target.numel()
#         partialEpoch = epoch + batch_idx / len(trainLoader) - 1
#         print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tError: {:.3f}\t Dice: {:.6f}'.format(
#             partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
#             loss.data[0], err, dice_loss))
#
#         trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
#         trainF.flush()
#
# def test_nll(args, epoch, model, testLoader, optimizer, testF):
#     model.eval()
#     test_loss = 0
#     dice_loss = 0
#     incorrect = 0
#     numel = 0
#     for data, target in testLoader:
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target)
#         target = target.view(target.numel())
#         numel += target.numel()
#         output = model(data)
#         test_loss += F.nll_loss(output, target, weight=weights).data[0]
#         dice_loss += bioloss.dice_error(output, target)
#         pred = output.data.max(1)[1]  # get the index of the max log-probability
#         incorrect += pred.ne(target.data).cpu().sum()
#
#     test_loss /= len(testLoader)  # loss function already averages over batch size
#     dice_loss /= len(testLoader)
#     err = 100.*incorrect/numel
#     print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%) Dice: {:.6f}\n'.format(
#         test_loss, incorrect, numel, err, dice_loss))
#
#     testF.write('{},{},{}\n'.format(epoch, test_loss, err))
#     testF.flush()
#     return err

def train_dice(args, epoch, model, trainLoader, optimizer, trainF):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, output in enumerate(trainLoader):
        data, target, id = output
        # print("training with {}".format(id[0]))
        target = target[0,:,:,:].view(-1) # right? added by Chao. 
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data = Variable(data)
        target = Variable(target)
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # pdb.set_trace()
        loss = bioloss.dice_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        err = 100.*(1. - loss.data[0]) # loss.data[0] is dice coefficient? By Chao.
        # partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        # print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tError: {:.8f}'.format(
        #     partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
        #     loss.data[0], err))

    print('\nFor trainning: Epoch: {} \tdice_coefficient: {:.4f}\tError: {:.4f}\n'.format(
    epoch, loss.data[0], err))

        # trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
    trainF.write('{},{},{}\n'.format(epoch, loss.data[0], err))
    trainF.flush()

def test_dice(args, epoch, model, testLoader, optimizer, testF):
    model.eval()
    test_dice = 0
    incorrect = 0
    for batch_idx, output in enumerate(testLoader):
        data, target, id = output
        # print("testing with {}".format(id[0]))
        target = target[0,:,:,:].view(-1) # right? added by Chao. 
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = Variable(data)
        target = Variable(target)
        output = model(data)
        dice = bioloss.dice_loss(output, target).data[0]
        test_dice += dice
        incorrect += (1. - dice)

    nTotal = len(testLoader)
    test_dice /= nTotal  # loss function already averages over batch size
    err = 100.*incorrect/nTotal
    # print('\nTest set: Average Dice Coeff: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, incorrect, nTotal, err))
    #
    # testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    print('\nFor testing: Epoch:{}\tAverage Dice Coeff: {:.4f}\tError:{:.4f}\n'.format(epoch, test_dice, err))

    testF.write('{},{},{}\n'.format(epoch, test_dice, err))
    testF.flush()
    return test_dice

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
