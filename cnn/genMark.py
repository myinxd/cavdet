# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Get marked images, pixels in cavity regions are set as one.
"""

import os
import argparse

import cnn_utils as utils

'''
def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Generate marked images.")
    # Parameters
    parser.add_argument("obspath", help="path of the observation")
    args = parser.parse_args()

    obspath = args.obspath

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    cntpath = os.path.join(obspath, 'cnt.reg')
    cavpath = os.path.join(obspath, 'cav_man.reg')
    markpath = os.path.join(obspath, 'img_mark.png')
    imgpath = os.path.join(obspath, 'img_cut.fits')
    # get cav_mat
    cav_mat = utils.get_reg(cntpath, cavpath)
    # get mark_image
    utils.get_mark(imgpath, cav_mat, savepath=markpath)

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
            cntpath = os.path.join(obspath, 'cnt.reg')
            cavpath = os.path.join(obspath, 'cav_man.reg')
            markpath = os.path.join(obspath, 'img_mark.png')
            imgpath = os.path.join(obspath, 'img_cut.fits')
            # get cav_mat
            cav_mat = utils.get_reg(cntpath, cavpath)
            # get mark_image
            utils.get_mark(imgpath, cav_mat, savepath=markpath)

if __name__ == "__main__":
    main()
