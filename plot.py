#!/usr/bin/env python3

import argparse
import os
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
from matplotlib import rcParams
rcParams.update({'figure.autolayout':True})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nBatches', type=int)
    parser.add_argument('expDir', type=str)
    args = parser.parse_args()

    trainP = os.path.join(args.expDir, 'train.csv')
    trainData = np.loadtxt(trainP, delimiter=',').reshape(-1, 3)
    testP = os.path.join(args.expDir, 'test.csv')
    testData = np.loadtxt(testP, delimiter=',').reshape(-1, 3)

    # N = args.nBatches
    trainI, trainDice, trainErr = np.split(trainData, [1,2], axis=1)
    trainI, trainDice, trainErr = [x.ravel() for x in
                                   (trainI, trainDice, trainErr)]
    # trainI_, trainDice_, trainErr_ = rolling(N, trainI, trainDice, trainErr)

    testI, testDice, testErr = np.split(testData, [1,2], axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(trainI, trainDice, label='Train')
    # plt.plot(trainI_, trainLoss_, label='Train')
    plt.plot(testI, testDice, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Dice coefficient')
    plt.legend()
    ax.set_yscale('linear')
    dice_fname = os.path.join(args.expDir, 'dice.png')
    plt.savefig(dice_fname)
    print('Created {}'.format(dice_fname))

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plt.plot(trainI, trainErr, label='Train')
    # plt.plot(trainI_, trainErr_, label='Train')
    plt.plot(testI, testErr, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    ax.set_yscale('linear')
    plt.legend()
    err_fname = os.path.join(args.expDir, 'error.png')
    plt.savefig(err_fname)
    print('Created {}'.format(err_fname))

    dice_err_fname = os.path.join(args.expDir, 'dice-error.png')
    os.system('convert +append {} {} {}'.format(dice_fname, err_fname, dice_err_fname))
    print('Created {}'.format(dice_err_fname))

def rolling(N, i, dice, err):
    i_ = i[N-1:]
    K = np.full(N, 1./N)
    dice_ = np.convolve(dice, K, 'valid')
    err_ = np.convolve(err, K, 'valid')
    return i_, dice_, err_

if __name__ == '__main__':
    main()
