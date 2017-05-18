# Copyright (C) 2017 zxma_sjtu@qq.com

"""
Test the net and generate result staffs

Steps
=====
[0] Load trained subclassifiers
[1] Load sample set
[2] Get estimated labels
[3] Generate reunited mask image
[4] Locate cavity by the connected regions
[5] Assess performance of the net on this observation
[6] Save results

log
===
Remove the step of edge detection
"""

import os
import argparse
import scipy.io as sio
import scipy.misc as misc
# import numpy as np

import utils_pro as utils
import utils_cnn as builder


def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Test the trained net.")
    # Parameters
    parser.add_argument("obspath", help="path of the observation")
    parser.add_argument("netpath", help="path of the trained net.")
    parser.add_argument("numgra", help="number of granule")
    parser.add_argument("matname", help="name of the sample data mat.")
    args = parser.parse_args()

    obspath = args.obspath
    netpath = args.netpath
    numgra = int(args.numgra)
    matname = args.matname

    # get boxsize and px_over
    matinfo = matname.split('/')[-1]
    matinfo = matinfo[0:-4].split('_')
    boxsize = int(matinfo[-2])
    px_diff = int(matinfo[-1])
    px_over = boxsize - px_diff

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return


    samplepath = os.path.join(obspath, matname)
    imgrecover = os.path.join(obspath,'img_re.png')
    estpath = os.path.join(obspath, 'sample_est.mat')

    # load sample
    print('[1] Loading sample set...')
    sample = sio.loadmat(samplepath)
    # reshape
    temp = sample['data'].reshape(-1,1,10,10)
    data_re = temp.astype('float32')
    data = sample['data']
    # get test label
    print('[2] Getting estimated labels...')
    # label = builder.cnn_estimate(inputs=data_re, network=network, est_fn=est_fn)
    label = builder.get_vote(inpath=netpath, numgra=numgra, data=data_re)
    # get recovered image
    print('[3] Getting recovered images...')
    img_re = utils.img_recover(data, label, imgsize=(200,200), px_over=px_over, thrs=2)
    # locate cavities
    print('[4] Locating cavities...')
    utils.cav_locate(img_re, obspath=obspath, rate=0.7)

    # save result
    print('[5] Saving results...')
    sio.savemat(estpath, {'data': data, 'label': label})
    misc.imsave(imgrecover, img_re)

'''


def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Test the trained network.")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    parser.add_argument("netpath", help="path of the trained network.")
    parser.add_argument("numgra", help="number of granule.")
    parser.add_argument("matname", help="name of the sample data mat.")
    args = parser.parse_args()

    inpath = args.inpath
    netpath = args.netpath
    numgra = int(args.numgra)
    matname = args.matname
    # get boxsize and px_over
    matinfo = matname[0:-5].split('_')
    boxsize = int(matinfo[-2])
    px_diff = int(matinfo[-1])
    px_over = boxsize - px_diff

    # load the network
    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    samples = os.listdir(inpath)
    for obs in samples:
        obspath = os.path.join(inpath, obs)
        # Judge existance of the path
        if not os.path.exists(obspath):
            print("The observation %s does not exist." % obspath)
        else:
            print("Processing on sample %s ..." % obs)
            samplepath = os.path.join(obspath, matname)
            imgrecover = os.path.join(obspath, 'img_re.png')
            estpath = os.path.join(obspath, 'sample_est.mat')
            # load sample
            print('[1] Loading sample set...')
            sample = sio.loadmat(samplepath)
            # reshape
            temp = sample['data'].reshape(-1, 1, 10, 10)
            data_re = temp.astype('float32')
            data = sample['data']
            # get test label
            print('[2] Getting estimated labels...')
            label = builder.get_vote(
                inpath=netpath, numgra=numgra, data=data_re)
            # get recovered image
            print('[3] Getting recovered images...')
            img_re = utils.img_recover(
                data, label, imgsize=(200, 200), px_over=px_over)
            # locate cavities
            print('[4] Locating cavities...')
            utils.cav_locate(img_edge, obspath=obspath, rate=0.6)

            # save result
            print('[5] Saving results...')
            sio.savemat(estpath, {'data': data, 'label': label})
            misc.imsave(imgrecover, img_re)

'''

if __name__ == "__main__":
    main()
