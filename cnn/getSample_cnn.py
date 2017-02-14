# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Get samples to train CNN

Totally 22 samples, in which 17 are for training and 5 for test.
"""

import os
import argparse
import numpy as np
import scipy.io as sio
from astropy.io import fits
from scipy.misc import imread

import cnn_utils as utils

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Get samples to train CNN")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    parser.add_argument("numtrain", help="number of observations for training.")
    parser.add_argument("numtest", help="number of observations for test.")
    args = parser.parse_args()

    inpath = args.inpath
    numtrain = int(args.numtrain)
    numtest = int(args.numtest)

    # Init
    train_mat = None
    train_label = None
    test_mat = None
    test_label = None
    samples_train = []
    samples_test = []

    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    samples = os.listdir(inpath)
    num_sample = len(samples)
    # randomlly get samples
    if numtrain + numtest > num_sample:
        print("Warning: Number of observations do not match the input numtrain and numtest.")
        idx_train = np.random.permutation(num_sample - 5)
        idx_test = np.random.permutation(5) + (num_sample - 5)
    else:
        idx_train = np.random.permutation(numtrain)
        idx_test = np.random.permutation(numtest) + numtrain

    print("Generating trainging samples...")
    # get training sample
    for i in range(len(idx_train)):
        obspath = os.path.join(inpath, samples[idx_train[i]])
        samples_train.append(samples[idx_train[i]])
        # Judge existance of the path
        if not os.path.exists(obspath):
            print("The observation %s does not exist." % obspath)
        else:
            print("Processing on sample %s ..." % samples[idx_train[i]])
            markpath = os.path.join(obspath, 'img_mark.png')
            imgpath = os.path.join(obspath, 'img_cut.fits')
            # load images
            h = fits.open(imgpath)
            img = h[0].data
            # normalize
            imgnorm = (img - np.min(img)) / (np.max(img) - np.min(img))
            # load mask
            mask = imread(markpath) / np.int32(255)
            # get cav_mat
            data,label = utils.gen_sample(imgnorm, mask, rate=0.4,
                                          boxsize=10, px_over=5)
            if train_mat is None:
                train_mat = np.array(data)
                train_label = np.array(label)
            else:
                train_mat = np.row_stack((train_mat,data))
                train_label = np.row_stack((train_label, label))

    # get test sample
    print("Generating test samples...")
    for i in range(len(idx_test)):
        obspath = os.path.join(inpath, samples[idx_test[i]])
        samples_test.append(samples[idx_test[i]])
        # Judge existance of the path
        if not os.path.exists(obspath):
            print("The observation %s does not exist." % obspath)
        else:
            print("Processing on sample %s ..." % samples[idx_test[i]])
            markpath = os.path.join(obspath, 'img_mark.png')
            imgpath = os.path.join(obspath, 'img_cut.fits')
            # load images
            h = fits.open(imgpath)
            img = h[0].data
            # normalize
            imgnorm = (img - np.min(img)) / (np.max(img) - np.min(img))
            # load mask
            mask = imread(markpath) / np.int32(255)
            # get cav_mat
            data,label = utils.gen_sample(imgnorm, mask, rate=0.4,
                                          boxsize=10, px_over=5)
            if test_mat is None:
                test_mat = np.array(data)
                test_label = np.array(label)
            else:
                test_mat = np.row_stack((test_mat,data))
                test_label = np.row_stack((test_label, label))

    # save
    train_path = os.path.join(inpath, 'sample_train')
    test_path = os.path.join(inpath, 'sample_test')
    print("Saving samples...")
    sio.savemat(train_path,{"data":train_mat,"label":train_label,
                            "name":samples_train,"index":idx_train})
    sio.savemat(test_path,{"data":test_mat,"label":test_label,
                           "name":samples_test,"index":idx_test})

if __name__ == "__main__":
    main()


