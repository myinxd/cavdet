# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
This script generates a convolutional neural network based classifier to detect X-ray astronomical cavities.
The codes are written under the structure designed by Theano and Lasagne.

Note
====
[2017-02-23] Add script to handle the imbalanced situation

References
==========
[1] Lasagne tutorial
    http://lasagne.readthedocs.io/en/latest/user/tutorial.html
[2] Theano tutorial
    http://www.deeplearning.net/software/theano/

Methods
=======
load_data: load the prepared dataset
cnn_build: build the cnn network
cnn_train: train the cnn network
cnn_test: test and estimate by the trained network
iterate_minibathces: a batch helper method
"""

import os
import time
import pickle
import numpy as np
import scipy.io as sio

import theano
import theano.tensor as T
import lasagne


def load_data(inpath, ratio_train=0.8, ratio_val=0.2):
    """
    Load the prepared dataset, and reshape for granular cnn training

    Inputs
    ======
    inpath: str
        Path of the mat dataset
    ratio_train: float
        Ratio of training samples in the sample set, default as 0.8
    ratio_val: float
        Ratio of validation samples in the training set, default as 0.2

    Outputs
    =======
    x_train: np.ndarray
        The training data
    y_train: np.ndarray
        Labels for the training data
    x_val: np.ndarray
        The validation data
    y_val: np.ndarray
        Labels for the validation data
    x_test: np.ndarray
        The test data
    y_test: np.ndarray
        Labels for the test data
    boxsize: integer
        boxsize of the subimage
    """
    # load the dataset
    try:
        data = sio.loadmat(inpath)
    except IOError:
        print("Path does not exist.")
        return

    data_bkg = data['data_bkg']
    data_ext = data['data_ext']
    data_cav = data['data_cav']
    label_bkg = data['label_bkg']
    label_ext = data['label_ext']
    label_cav = data['label_cav']
    # boxsize
    box = data_bkg.shape[1]
    boxsize = int(np.sqrt(box))

    # separate the major set into subsets
    numgra = int(np.round(len(label_bkg) / len(label_cav)))

    # calc train, val, test amounts, and shuffle
    idx_bkg = np.random.permutation(len(label_bkg))
    idx_ext = np.random.permutation(len(label_ext))
    idx_cav = np.random.permutation(len(label_cav))

    numtrain_bkg = int(np.floor(len(label_bkg) * ratio_train))
    numtrain_ext = int(np.floor(len(label_ext) * ratio_train))
    numtrain_cav = int(np.floor(len(label_cav) * ratio_train))

    numval_bkg = int(np.floor(numtrain_bkg * ratio_val))
    numval_ext = int(np.floor(numtrain_ext * ratio_val))
    numval_cav = int(np.floor(numtrain_cav * ratio_val))

    # form dataset
    x_train_bkg = data_bkg[idx_bkg[0:numtrain_bkg], :]
    y_train_bkg = label_bkg[idx_bkg[0:numtrain_bkg], :]
    x_test_bkg = data_bkg[idx_bkg[numtrain_bkg:], :]
    y_test_bkg = label_bkg[idx_bkg[numtrain_bkg:], :]

    x_train_ext = data_ext[idx_ext[0:numtrain_ext], :]
    y_train_ext = label_ext[idx_ext[0:numtrain_ext], :]
    x_test_ext = data_ext[idx_ext[numtrain_ext:], :]
    y_test_ext = label_ext[idx_ext[numtrain_ext:], :]

    x_train_cav = data_cav[idx_cav[0:numtrain_cav], :]
    y_train_cav = label_cav[idx_cav[0:numtrain_cav], :]
    x_test_cav = data_cav[idx_cav[numtrain_cav:], :]
    y_test_cav = label_cav[idx_cav[numtrain_cav:], :]

    # val
    x_val = np.row_stack((x_train_bkg[0:numval_bkg, :],
                          x_train_ext[0:numval_ext, :],
                          x_train_cav[0:numval_cav, :]))
    y_val = np.row_stack((y_train_bkg[0:numval_bkg],
                          y_train_ext[0:numval_ext],
                          y_train_cav[0:numval_cav]))
    x_val_temp = x_val.reshape(-1, 1, boxsize, boxsize)
    x_val = x_val_temp.astype('float32')
    y_val = y_val[:, 0].astype('int32')

    # test
    x_test = np.row_stack((x_test_bkg, x_test_ext, x_test_cav))
    y_test = np.row_stack((y_test_bkg, y_test_ext, y_test_cav))
    x_test_temp = x_test.reshape(-1, 1, boxsize, boxsize)
    x_test = x_test_temp.astype('float32')
    y_test = y_test[:, 0].astype('int32')

    # train
    # To save the memory, only output the idx
    cav_temp = x_train_cav[numval_cav:, :]
    cav_re = cav_temp.reshape(-1, 1, boxsize, boxsize)
    x_cav = cav_re.astype('float32')
    y_cav = y_train_cav[numval_cav:, ].astype('int32')

    ext_temp = x_train_ext[numval_ext:, :]
    ext_re = ext_temp.reshape(-1, 1, boxsize, boxsize)
    x_ext = ext_re.astype('float32')
    y_ext = y_train_ext[numval_ext:, ].astype('int32')

    bkg_temp = x_train_bkg[numval_bkg:, :]
    bkg_re = bkg_temp.reshape(-1, 1, boxsize, boxsize)
    x_bkg = bkg_re.astype('float32')
    y_bkg = y_train_bkg[numval_bkg:, ].astype('int32')

    x_train = {'bkg': x_bkg, 'cav': x_cav, 'ext': x_ext, 'numgra': numgra}
    y_train = {'bkg': y_bkg, 'cav': y_cav, 'ext': y_ext, 'numgra': numgra}

    return x_train, y_train, x_val, y_val, x_test, y_test, boxsize


def cnn_build(boxsize=10, num_class=3, kernel_size=[2, 3, 4], kernel_num=[12, 12, 12],
              pool_flag=[False, False, False], input_var=None):
    """
    Build the cnn network

    Inputs
    ======
    boxsize: integer
        The size of the image boxsize
    num_class: integer
        Number of classes
    kernel_size: list
        Kernel sizes in the convolutional layers
    kernel_num: list
        Number of kernels (feature maps) in the ConvLayers
    pool_flag: list
        Flags of whether max pooling after the ConvLayers
    input_var: np.ndarray
        The input dataset or batched data, default as None

    Output
    ======
    network: Lasagne.layers
        The pre-constructed network
    """
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, boxsize, boxsize),
                                        input_var=input_var)

    # ConvLayers
    s = boxsize  # size of the feature map at the last Conv layer
    for k in range(len(kernel_size)):
        # ConvLayer
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=kernel_num[k],
            filter_size=(kernel_size[k], kernel_size[k]),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
        # Max pooling
        if pool_flag[k]:
            network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
            s = (s - kernel_size[k] + 1) // 2
        else:
            s = s = kernel_size[k] + 1

    # Full connected layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.2),
        num_units=s**2 * kernel_num[-1],
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # Output Layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.2),
        num_units=num_class,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize=100, shuffle=True):
    """
    Design a iterator to generate batches for cnn training.

    Inputs
    ======
    inputs: np.ndarray
        The dataset
    targets: np.ndarray
        Labels of the samples with respect to inputs
    batchsize: integer
        Size of the batch
    shuffle: bool
        Whether shuffle the indices.

    output
    ======
    yield an iterator
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def cnn_train(inputs_train, targets_train, inputs_val, targets_val,
              network=None, batchsize=100, num_epochs=100):
    """
    Train the cnn network

    Inputs
    ======
    inputs: np.ndarray
        The training data
    targets: np.ndarray
        Labels of the samples
    network: lasagne.layers
        The CNN network
    batchsize: integer
        Size of the single batch
    num_epochs: integer
        Number of epochs for training

    Output
    ======
    network: lasagne.layers
        The trained network
    """
    # Init
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = cnn_build(boxsize=10, num_class=3,
                        kernel_size=[2, 3, 4], kernel_num=[15, 15, 15],
                        pool_flag=[False, False, False], input_var=input_var)

    # Create the loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create the update expressions for training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    # Create the loss expression for validation
    val_prediction = lasagne.layers.get_output(network, deterministic=True)
    val_loss = lasagne.objectives.categorical_crossentropy(val_prediction,
                                                           target_var)
    val_loss = val_loss.mean()
    val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), target_var),
                     dtype=theano.config.floatX)

    # Create the loss expression for estimation
    est_prediction = lasagne.layers.get_output(network, deterministic=True)
    label_est = T.argmax(est_prediction, axis=1)

    # Compile the train function
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    # Compile the validation function
    val_fn = theano.function([input_var, target_var], [val_loss, val_acc])
    # Compile the estimation function
    est_fn = theano.function([input_var], [label_est])

    # Training
    print("Starting training...")

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(inputs_train, targets_train, batchsize, shuffle=True):
            input_batch, target_batch = batch
            train_err += train_fn(input_batch, target_batch)
            train_batches += 1

        # validation
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(inputs_val, targets_val, batchsize, shuffle=True):
            input_batch, target_batch = batch
            err, acc = val_fn(input_batch, target_batch)
            val_err += err
            val_acc += acc
            val_batches += 1

        # print result
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    return network, val_fn, est_fn


def cnn_test(inputs, targets, network, test_fn, batchsize=100):
    """
    Test the trained cnn network

    Inputs
    ======
    inputs: np.ndarray
        The test dataset
    targets: np.ndarray
        Labels of the samples
    network: lasagne.layers
        The trained cnn network

    """
    # test
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(inputs, targets, batchsize, shuffle=True):
        input_batch, target_batch = batch
        err, acc = test_fn(input_batch, target_batch)
        test_err += err
        test_acc += acc
        test_batches += 1

    # print result
    print("Testing the network...")
    print("test loss:\t\t{:.6f}".format(test_err / test_batches))
    print("test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))


def cnn_estimate(inputs, network, est_fn):
    """
    Estimate label by the trained cnn network

    Inputs
    ======
    inputs: np.ndarray
        The test dataset
    network: lasagne.layers
        The trained cnn network

    Output
    ======
    label_est: np.ndarray
        The estimated labels
    """
    # estimate
    # print("Estimating ...")
    label_est = est_fn(inputs)

    return label_est[0]


def get_vote(inpath, numgra, data):
    """
    Estimate label using the granularied networks

    Inputs
    ======
    inpath: str
        Path to save the trained models
    numgra: integer
        Number of granule
    data: np.ndarray
        The data to test

    Ouput
    =====
    label_est: np.ndarray
        The estimated voted label
    """
    # Init
    numsample = data.shape[0]
    label = np.zeros((numsample, numgra))

    for i in range(numgra):
        print("Estimating by granula %d" % (i + 1))
        modelname = ("model%d.pkl" % (i + 1))
        modelpath = os.path.join(inpath, modelname)
        # load net
        model = cnn_load(modelpath)
        # est
        label[:, i] = cnn_estimate(data,
                                   model["network"],
                                   model["est_fn"])

    # vote
    idx_ext = np.where(label == 2)
    label[idx_ext] = 0

    label_sum = np.sum(label, axis=1)
    label_est = np.zeros((numsample,))
    thrs = numgra / 2
    label_est[np.where(label_sum >= thrs)] = 1

    return label_est


def cnn_save(savepath, savedict):
    """
    Save the trained network, and the corresponding theano functions

    Reference
    =========
    [1] Save python data by pickle
        http://www.cnblogs.com/pzxbc/archive/2012/03/18/2404715.html

    Inputs
    ======
    savepath: str
        Path to save the result
    savedict: dict
        The data to be saved.
        For instance: {'network':newtork,'est_fn':est_fun}

    """
    # Init
    fp = open(savepath, 'wb')

    # save
    pickle.dump(savedict, fp)

    # close
    fp.close()


def cnn_load(modelpath):
    """
    Load the saved model

    Input
    =====
    modelpath: str
        Path to load the saved model

    Output
    ======
    model: dict
        The dictonary that saved the network and functions.
    """
    # Init
    fp = open(modelpath, 'rb')

    # load
    model = pickle.load(fp)

    # close
    fp.close()

    return model


def get_map(network, savepath=None):
    """
    Get the feature maps from the trained network

    Inputs
    ======
    network: lasagne.layers
        The trained network
    savepath: str
        Path to save the maps, default as None

    Output
    ======
    maps: dict
        The dict that saves the maps
    """
    # Init
    params = lasagne.layers.get_all_params(network,
                                           regularizable=True,
                                           unwrap_shared=False)
    # delete useless params
    # del(params[-4:])
    # get weights and biases
    maps = {}
    # numlayer = len(params)//2
    numlayer = len(params)
    for i in range(numlayer - 2):
        w = params[i]
        weight = w.get_value()
        # weight = weight.sum(axis=1)
        # b = params[i+1]
        key_weight = ('w%d' % (i + 1))
        # key_bias = ('b%d' % (i+1))
        maps[key_weight] = weight
        # maps[key_bias] = b.get_value()

    if savepath is not None:
        print('Saving the parameters...')
        sio.savemat(savepath, maps)

    return maps


def get_assess(img_mask, img_re):
    """
    Calculate performance of the approach

    Inputs
    ======
    img_mask: np.ndarray
        The mask image
    img_re: np.ndarray
        The recovered image

    Outputs
    =======
    r_sen: float
        Sensitivity, TP / (TP + FN)
    r_spe: float
        Specificity, TN / (TN + FP)
    r_acc: float
        Accuracy, (TP+TN)/All
    """
    Num_pos = len(np.where(img_re == 1)[0])
    Num_neg = len(np.where(img_re == 0)[0])

    # TP
    img_mul = img_mask * img_re
    TP = len(np.where(img_mul == 1)[0])
    # TN
    img_mul_rev = (img_mask - 1) * (img_re - 1)
    TN = len(np.where(img_mul_rev == 1)[0])

    # Acc
    r_acc = (TP + TN) / (Num_pos + Num_neg)
    # Sen
    r_sen = TP / Num_pos
    # Spe
    r_spe = TN / Num_neg

    return r_acc, r_sen, r_spe
