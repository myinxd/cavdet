# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
A script that generates splited images from the single observations.

Steps
=====
[1] Generate samples of this observation
[2] Generate recovered image by the true mask image
[3] Do edge detection
[4] Loacate cavities by the connected regions
[5] Save results

"""

import os
import argparse
from astropy.io import fits
from scipy.io import savemat
import scipy.misc as misc

import utils_pro as utils

'''
def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(
        description="Get recovered cavity regions.")
    # Parameters
    parser.add_argument("obspath", help="path of the observation")
    parser.add_argument("matname", help="Name of the saved sample data mat.")
    parser.add_argument("px_slide", help="Number of overlapped pixels")
    args = parser.parse_args()

    obspath = args.obspath
    matname = args.matname
    px_slide = int(args.px_slide)

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    markpath = os.path.join(obspath, 'img_mark.png')
    imgpath = os.path.join(obspath, 'img_smooth.fits')
    samplepath = os.path.join(obspath, matname)

    h = fits.open(imgpath)
    img = h[0].data
    img = (img - img.min()) / (img.max() - img.min())
    # load marked image.
    img_mark = misc.imread(markpath)
    # get split samples
    print('[1] Generating split samples...')
    data, label = utils.gen_sample_multi(
        img, img_mark, rate=0.3, boxsize=10, px_over=px_slide)

    # save result
    print('[2] Saving results...')
    savemat(samplepath, {'data': data, 'label': label})
'''

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(
        description="Get recovered cavity regions.")
    # Parameters
    parser.add_argument("inpath", help="path of the observations")
    parser.add_argument("matname", help="Name of the saved sample data mat.")
    parser.add_argument("px_slide", help="Number of overlapped pixels")
    args = parser.parse_args()

    inpath = args.inpath
    matname = args.matname
    px_slide = int(args.px_slide)

    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    samples = os.listdir(inpath)
    for obs in samples:
        obspath = os.path.join(inpath,obs)
        print("Processing on observation %s" % obs)
        # Judge existance of the path
        if not os.path.exists(obspath):
            print("The observation %s does not exist." % obspath)
            return

        markpath = os.path.join(obspath, 'img_mark.png')
        imgpath = os.path.join(obspath, 'img_smooth.fits')
        samplepath = os.path.join(obspath, matname)

        h = fits.open(imgpath)
        img = h[0].data
        img = (img - img.min()) / (img.max() - img.min())
        # load marked image.
        img_mark = misc.imread(markpath)
        # get split samples
        print('[1] Generating split samples...')
        data, label = utils.gen_sample_multi(
            img, img_mark, rate=0.3, boxsize=10, px_over=px_slide)

        # save result
        print('[2] Saving results...')
        savemat(samplepath, {'data': data, 'label': label})

if __name__ == "__main__":
    main()
