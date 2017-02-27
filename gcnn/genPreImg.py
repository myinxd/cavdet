# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qqq.com>

"""
The preprocessing process

Detect center point, subtract the galaxy center regions,
detect point sources, and fill them with backgrounds.

Note
====
The python environment of CIAO does not have astropy, which may lead errors.

"""

import os
import argparse

import utils_pre as utils

'''
def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Preprocessing on the raw images.")
    # Parameters
    parser.add_argument("obspath", help="path of the input")
    parser.add_argument("boxsize", default='100', help="semi size of cut image (default is 100)")
    args = parser.parse_args()

    obspath = args.obspath
    # box_size = int(args.boxsize)

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    # get cnt
    # utils.get_cnt(obspath, box_size=box_size, imgname='evt2_sub.fits')
    # get cut image
    utils.get_img(obspath, cntpath='cnt.reg',evtname='evt2_sub.fits',cutname='evt2_cut.fits')
    # detect point sources
    utils.get_ps(obspath, pspath='wavd', evtname='evt2_cut.fits')
    # fill ps regions
    utils.fill_ps(obspath, pspath='wavd', evtname='evt2_cut.fits', fillname='img_cut.fits')
    # smooth (optional)
    utils.get_smooth(obspath, sigma=2,imgname='img_cut.fits',smoothname='img_smooth.fits')

'''


def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(
        description="Preprocessing on the raw images.")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    parser.add_argument("boxsize", default='100',
                        help="semi size of cut image (default is 100)")
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
            # get cnt
            utils.get_cnt(obspath, box_size=box_size, imgname='evt2_sub.fits')
            # get cut image
            utils.get_img(obspath, cntpath='cnt.reg',
                          evtname='evt2_sub.fits', cutname='evt2_cut.fits')
            # detect point sources
            utils.get_ps(obspath, pspath='wavd', evtname='evt2_cut.fits')
            # fill ps regions
            utils.fill_ps(obspath, pspath='wavd',
                          evtname='evt2_cut.fits', fillname='img_cut.fits',
                          regname='wavd_man.reg')
            # smooth (optional)
            utils.get_smooth(obspath, sigma=2,
                             imgname='img_cut.fits', smoothname='img_smooth.fits')

if __name__ == "__main__":
    main()
