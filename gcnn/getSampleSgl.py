# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
A script that process on the single observations.

Steps
=====
[1] Load the true cavity regions
[2] Generate the mask image
[3] Generate samples of this observation
[4] Generate recovered image by the true mask image
[5] Do edge detection
[6] Loacate cavities by the connected regions 
[7] Save results

"""

import os
import argparse
from astropy.io import fits
from scipy.io import savemat
import scipy.misc as misc

import cnn_utils as utils


def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(
        description="Get recovered cavity regions.")
    # Parameters
    parser.add_argument("obspath", help="path of the observation")
    parser.add_argument(
        "logflag", help="whether to logarithmize the beta function")
    parser.add_argument(
        "thrshold", help="threshold between ext and faint background.")
    args = parser.parse_args()

    obspath = args.obspath
    ext_thrs = float(args.thrshold)
    logflag = bool(int(args.logflag))

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    cntpath = os.path.join(obspath, 'cnt.reg')
    cavpath = os.path.join(obspath, 'cav_man.reg')
    betapath = os.path.join(obspath, 'beta.fits')
    markpath = os.path.join(obspath, 'img_mark.png')
    imgpath = os.path.join(obspath, 'img_smooth.fits')
    imgrecover = os.path.join(obspath, 'img_re.png')
    imgedge = os.path.join(obspath, 'img_edge.png')
    samplepath = os.path.join(obspath, 'sample_20.mat')

    h = fits.open(imgpath)
    img = h[0].data
    img = (img - img.min()) / (img.max() - img.min())
    # get cav_mat
    print('[1] Loading cavities...')
    cav_mat = utils.get_reg(cntpath, cavpath)
    # get mark image
    print('[2] Generating mask image...')
    img_mark = utils.get_mark_multi(betapath, cav_mat,
                                    thrs=ext_thrs, logflag=logflag,
                                    savepath=markpath)
    img_mark = misc.imread(markpath)
    # get split samples
    print('[3] Generating split samples...')
    data, label = utils.gen_sample_multi(
        img, img_mark, rate=0.3, boxsize=10, px_over=5)
    # get recovered image
    print('[4] Getting recovered images...')
    img_re = utils.img_recover_multi(data, label, imgsize=img.shape, px_over=5)
    # get edge detection
    print('[5] Doing edge detection...')
    img_edge = utils.cav_edge(img_re, sigma=1)
    # locate cavities
    print('[6] Locating cavities...')
    utils.cav_locate(img_edge, obspath=obspath, rate=0.6)

    # save result
    print('[5] Saving results...')
    savemat(samplepath, {'data': data, 'label': label})
    misc.imsave(imgrecover, img_re)
    misc.imsave(imgedge, img_edge * 255)

if __name__ == "__main__":
    main()
