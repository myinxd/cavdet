# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Load data, and train the CNN network with downsampled major sample.

Note
====
Model and parameters are saved as 'model*.pkl' and 'params*.mat'
"""

import os
import argparse

import numpy as np

import utils_cnn as utils


def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(
        description="Load data, and train the network.")
    # Parameters
    parser.add_argument("inpath", help="path of the samples")
    parser.add_argument("outpath", help="path to save result.")
    parser.add_argument("numepoch", help="Number of epochs")
    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath
    numepoch = int(args.numepoch)

    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test, boxsize = utils.load_data(
        inpath=inpath, ratio_train=0.9, ratio_val=0.2)
    # build the network
    # train
    # numgra = x_train['numgra']
    cav_data = x_train['cav']
    ext_data = x_train['ext']
    bkg_data = x_train['bkg']
    cav_label = y_train['cav']
    ext_label = y_train['ext']
    bkg_label = y_train['bkg']

    idx = np.random.permutation(bkg_label.shape[0])
    idx = idx[0:ext_label.shape[0]+cav_label.shape[0]]
    print("Training on the upsampled dataset")
    data_train = np.row_stack((bkg_data[idx,:,:,:],
                               cav_data,
                               ext_data))
    label_train = np.row_stack((bkg_label[idx,:],
                                cav_label,
                                ext_label))
    label_train = label_train[:, 0]
    print(data_train.shape)
    print(label_train.shape)

    network, test_fn, est_fn = utils.cnn_train(inputs_train=data_train,
                                              targets_train=label_train,
                                              inputs_val=x_val, targets_val=y_val,
                                              network=None,
                                              batchsize=100, num_epochs=numepoch)
    # test
    utils.cnn_test(inputs=x_test, targets=y_test,
                    network=network, test_fn=test_fn, batchsize=100)
    # save model
    modelname = ("model_down.pkl")
    savepath = os.path.join(outpath, modelname)
    savedict = {'network': network, 'est_fn': est_fn}
    utils.cnn_save(savepath=savepath, savedict=savedict)
    # save params
    paramname = ("params_down.mat")
    parampath = os.path.join(outpath, paramname)
    utils.get_map(network, parampath)


if __name__ == "__main__":
    main()
