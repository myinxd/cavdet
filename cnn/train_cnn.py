# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Load data, and train the network
"""

import os
import argparse

import cnn_build as utils

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Load data, and train the network.")
    # Parameters
    parser.add_argument("inpath", help="path of the samples")
    parser.add_argument("numepoch", help="Number of epochs")
    args = parser.parse_args()

    inpath = args.inpath
    numepoch = int(args.numepoch)

    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    # Load data
    x_train,y_train,x_val,y_val,x_test,y_test,boxsize = utils.load_data(inpath=inpath)
    # build the network
    input_var = utils.T.tensor4('inputs')
    numclass=3
    kernel_size=[2,3,4]
    kernel_num=[12,12,12]
    pool_flag=[False,False,False]
    '''
    network = utils.cnn_build(boxsize=boxsize, num_class=numclass,
                              kernel_size=kernel_size, kernel_num=kernel_num,
                              pool_flag=pool_flag, input_var=input_var)
    '''
    # train
    network = utils.cnn_train(inputs_train=x_train, targets_train=y_train,
                              inputs_val=x_val, targets_val=y_val,
                              network = None,
                              batchsize=100, num_epochs=numepoch)
    # test
    # utils.cnn_test(inputs=x_test, targets=y_test, network=network, batchsize=100)

if __name__ == "__main__":
    main()
