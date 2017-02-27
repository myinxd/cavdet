# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Get recovered cavity regions from the true region file, a trick.
"""

import os
import argparse
from astropy.io import fits
from scipy.io import savemat
import scipy.misc as misc

import cnn_utils as utils

'''
def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Get recovered cavity regions.")
    # Parameters
    parser.add_argument("obspath", help="path of the observation")
    args = parser.parse_args()

    obspath = args.obspath

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    # cntpath = os.path.join(obspath, 'cnt.reg')
    # regpath = os.path.join(obspath, 'cav_det.reg')
    markpath = os.path.join(obspath, 'img_mark.png')
    imgpath = os.path.join(obspath, 'img_cut.fits')
    imgrecover = os.path.join(obspath,'img_re.png')
    imgedge = os.path.join(obspath, 'img_edge.png')
    samplepath = os.path.join(obspath, 'sample.mat')
    # load images
    h = fits.open(imgpath)
    img = h[0].data
    img_mark = misc.imread(markpath) / 255

    # get split samples
    print('[1] Generating split samples...')
    data,label = utils.gen_sample(img, img_mark, rate=0.5, boxsize=10, px_over=5)
    # get recovered image
    print('[2] Getting recovered images...')
    img_re = utils.img_recover(data,label,imgsize=img.shape, px_over=5)
    # get edge detection
    print('[3] Doing edge detection...')
    img_edge = utils.cav_edge(img_re, sigma=1)
    # locate cavities
    print('[4] Locating cavities...')
    utils.cav_locate(img_edge, obspath=obspath, rate=0.6)

    # save result
    print('[5] Saving results...')
    savemat(samplepath, {'data': data, 'label': label})
    misc.imsave(imgrecover, img_re)
    misc.imsave(imgedge, img_edge * 255)

'''


def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Generate marked images.")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    args = parser.parse_args()

    inpath = args.inpath

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
            markpath = os.path.join(obspath, 'img_mark.png')
            imgpath = os.path.join(obspath, 'img_cut.fits')
            imgrecover = os.path.join(obspath, 'img_re.png')
            imgedge = os.path.join(obspath, 'img_edge.png')
            samplepath = os.path.join(obspath, 'sample.mat')
            # load images
            h = fits.open(imgpath)
            img = h[0].data
            img_mark = misc.imread(markpath)

            # get split samples
            print('[1] Generating split samples...')
            data, label = utils.gen_sample_multi(
                img, img_mark, boxsize=10, px_over=5)
            # get recovered image
            print('[2] Getting recovered images...')
            img_re = utils.img_recover_multi(
                data, label, imgsize=img.shape, px_over=5)
            # get edge detection
            print('[3] Doing edge detection...')
            img_edge = utils.cav_edge(img_re, sigma=2)
            # locate cavities
            print('[4] Locating cavities...')
            utils.cav_locate(img_edge, obspath=obspath, rate=0.6)

            # save result
            print('[5] Saving results...')
            savemat(samplepath, {'data': data, 'label': label})
            misc.imsave(imgrecover, img_re)
            misc.imsave(imgedge, img_edge * 255)


if __name__ == "__main__":
    main()
