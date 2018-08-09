#!/anaconda/envs/torvnet python3

import sys
import os
import argparse

import numpy as np

import train

basePath = os.getcwd()

params = dict()
params['DataManagerParams'] = dict()
params['ModelParams'] = dict()

#  params of the algorithm
params['ModelParams']['numcontrolpoints'] = 2  #？？？
params['ModelParams']['sigma'] = 15
params['ModelParams']['device'] = 0
params['ModelParams']['snapshot'] = 0
params['ModelParams']['dirTrain'] = os.path.join(basePath,'dataset/Train')  # dataset from PROMISE12: prostate MRI scans, training case00-44(https://promise12.grand-challenge.org/download/#)
params['ModelParams']['dirTest'] = os.path.join(basePath,'dataset/Test') # dataset from PROMISE12: prostate MRI scans, training case45-49
params['ModelParams']['dirInfer'] = os.path.join(basePath,'dataset/Infer') # dataset from PROMISE12: prostate MRI scans, 30 testing
params['ModelParams']['dirResult'] = os.path.join(basePath,'results')  # where we need to save the results (relative to the base path)
params['ModelParams']['dirSnapshots'] = os.path.join(basePath,'Models/MRI_cinque_snapshots/')  # where to save the models while training
# params['ModelParams']['batchsize'] = 1  # the batchsize
params['ModelParams']['numIterations'] = 100000  # the number of iterations
params['ModelParams']['baseLR'] = 0.0001  # the learning rate, initial one
params['ModelParams']['nProc'] = 1  # the number of threads to do data augmentation


#params of the DataManager
params['DataManagerParams']['dstRes'] = np.asarray([1,1,1.5],dtype=float)
params['DataManagerParams']['VolSize'] = np.asarray([64,64,32],dtype=int)
params['DataManagerParams']['normDir'] = False  # if rotates the volume according to its transformation in the mhd file. Not reccommended.

print('\n+preset parameters:\n' + str(params))


#  parse sys.argv
parser = argparse.ArgumentParser()
parser.add_argument('--batchSz', type=int, default=1)
parser.add_argument('--dice', action='store_true', default=True)
parser.add_argument('--gpu_ids', type=list, default=[1])
parser.add_argument('--nEpochs', type=int, default=10)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-8)')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--opt', type=str, default='adam',
                    choices=('sgd', 'adam', 'rmsprop'))
args = parser.parse_args()

print('\n+sys arguments:\n' + str(args))

# install torchbiomed: https://github.com/mattmacy/torchbiomed

#  load dataset, train, test(i.e. output predicted mask for test data in .mhd)
train.main(params, args)

