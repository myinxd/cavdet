# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Detection cavities in X-ray astronomical images using concolutional neural networks.

The script aims to segment cavity regions in the X-ray astronomical images with the help of CNN, and it is designed under the example of Lasagne.

Changes
=======
Imblanced classification.

References
==========
[1] Lasagne tutorial
    http://lasagne.readthedocs.org/en/latest/user/tutorial.html
[2]

Methods
=======
load_sample: Load the samples

"""

from __future__ import print_function

# import sys
import os
import time
import argparse

import numpy as np
import scipy.io as sio
import theano
import theano.tensor as T

import lasagne

def load_sample(inpath, trainpath='sample_train.mat',
                testpath='sample_test.mat',rate_val = 0.2):
    """Load samples

    Input
    =====
    inpath: str
        path saving the sample data
    trainpath: str
        path of the training data, default as 'sample_train.mat'
    testpath: str
        path of the test data, default as 'sample_test.mat'
    rate_val: float
        the ratio of validation samples over the training samples

    Output
    ======
    X_train: np.ndarray
        the training set
    y_train: np.ndarray
        the label with respect to the training samples
    X_val: np.ndarray
        the validation set
    y_val: np.ndarray
        the label with respect to the validation samples
    X_test: np.ndarray
        the test set
    y_test: np.ndarray
        the label with respect to the test samples
    """
    # Init
    trainpath = os.path.join(inpath,trainpath)
    testpath = os.path.join(inpath, testpath)

    # Load training set
    sample_train = sio.loadmat(trainpath)
    train_data = sample_train['data'].astype('float32')
    # Get number of samples
    numtrain,boxsize = train_data.shape
    boxsize = int(np.sqrt(boxsize))
    # reshape
    X_data = train_data.reshape(-1,1,boxsize,boxsize)
    y_data = sample_train['label'].astype('int32')
    # y_data = y_data.reshape(numtrain,)
    # imbalanced situation
    idx_cav = np.where(y_data == 1)[0]
    idx_bkg = np.where(y_data == 0)[0]
    bkg_rand = np.random.permutation(len(idx_bkg))
    bkg_im = idx_bkg[bkg_rand[0:len(idx_cav)]]
    # get balanced data
    cav_data = X_data[idx_cav]
    cav_label = y_data[idx_cav]
    bkg_data = X_data[bkg_im]
    bkg_label = y_data[bkg_im]
    X_train = np.row_stack((cav_data,bkg_data))
    y_train = np.row_stack((cav_label,bkg_label))
    y_train = y_train.reshape(len(y_train),)
    # Load test set
    sample_test = sio.loadmat(testpath)
    test_data = sample_test['data'].astype('float32')
    X_test = test_data.reshape(-1,1,boxsize,boxsize)
    y_test = sample_test['label'].astype('int32')
    y_test = y_test.reshape(y_test.shape[0],)

    # Get validation set
    numval = int(np.floor(len(idx_cav) * 2 * rate_val))
    idx = np.random.permutation(len(idx_cav)*2)
    X_val = X_train[idx[0:numval]]
    y_val = y_train[idx[0:numval]]

    # Get train
    X_train = X_train[idx[numval:]]
    y_train = y_train[idx[numval:]]

    return X_train, y_train, X_val, y_val, X_test, y_test, boxsize


def build_cnn(input_var=None, boxsize=20):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, boxsize, boxsize),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # The third convolution layer
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify)
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))


    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
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


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main():
    """The main function"""
    # Init
    parser = argparse.ArgumentParser(description="Detect cavities using CNN.")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    parser.add_argument("epochtime",help="number of epoches")
    parser.add_argument("rate_val",help="ratio of valitation samples over training samples")
    args = parser.parse_args()

    inpath = args.inpath
    num_epochs = int(args.epochtime)
    rate_val = float(args.rate_val)

    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test,boxsize = load_sample(
        inpath, rate_val = rate_val)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var,boxsize=boxsize)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    main()
