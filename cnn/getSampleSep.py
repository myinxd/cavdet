# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Get samples to train CNN

Totally 21 samples, in which 17 are for training and 5 for test.
"""

import os
import argparse
import numpy as np
import scipy.io as sio


def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Get samples to train CNN")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    parser.add_argument("ratio", help="ratio of training samples.")
    args = parser.parse_args()

    inpath = args.inpath
    # ratio = float(args.ratio)

    # Init
    data_bkg = None
    data_ext = None
    data_cav = None
    label_bkg = None
    label_ext = None
    label_cav = None

    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    samples = os.listdir(inpath)

    for s in samples:
        obspath = os.path.join(inpath, s)
        # Judge existance of the path
        if not os.path.exists(obspath):
            print("The observation %s does not exist." % obspath)
        else:
            print("Processing on sample %s ..." % s)
            matpath = os.path.join(obspath, 'sample_sm.mat')
            # load samples
            dataset = sio.loadmat(matpath)
            # separate samples
            data = dataset['data']
            label = dataset['label']
            idx_bkg = np.where(label==0)[0]
            idx_cav = np.where(label==1)[0]
            idx_ext = np.where(label==2)[0]
            # combine
            if data_bkg is None:
                data_bkg = data[idx_bkg,:]
                label_bkg = label[idx_bkg,:]
                data_ext = data[idx_ext,:]
                label_ext = label[idx_ext,:]
                data_cav = data[idx_cav,:]
                label_cav = label[idx_cav,:]
            else:
                data_bkg = np.row_stack((data_bkg,data[idx_bkg,:]))
                label_bkg = np.row_stack((label_bkg,label[idx_bkg,:]))
                data_ext = np.row_stack((data_ext,data[idx_ext,:]))
                label_ext = np.row_stack((label_ext,label[idx_ext,:]))
                data_cav = np.row_stack((data_cav,data[idx_cav,:]))
                label_cav = np.row_stack((label_cav,label[idx_cav,:]))

    # get train and test sample

    # save
    sample_path = os.path.join(inpath, 'sample_all')
    print("Saving samples ...")
    sio.savemat(sample_path,{"data_bkg": data_bkg,
                             "data_ext": data_ext,
                             "data_cav": data_cav,
                             "label_bkg":label_bkg,
                             "label_ext":label_ext,
                             "label_cav":label_cav})

if __name__ == "__main__":
    main()


