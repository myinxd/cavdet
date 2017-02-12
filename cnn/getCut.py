# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qqq.com>

"""
Preprocessing to get the sample image after manually filtered point
sources detected by wavdetect.
"""

import os
import argparse

import utils_pre as utils

'''
def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Preprocessing on the raw images: step 2.")
    # Parameters
    parser.add_argument("obspath", help="path of the input")
    parser.add_argument("boxsize", default='400', help="size of cut image (default is 400)")
    args = parser.parse_args()

    obspath = args.obspath
    box_size = int(args.boxsize)

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    # fill ps
    utils.fill_ps(obspath, pspath='wavd')
    # get center region
    utils.get_cnt(obspath, box_size=box_size)
    # get cut image
    utils.get_img(obspath, cntpath='cnt.reg')
'''

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Preprocessing on the raw images: step 2.")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    parser.add_argument("boxsize", default='400', help="size of cut image (default is 400)")
    args = parser.parse_args()

    inpath = args.inpath
    box_size = int(args.boxsize)

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
            # fill ps
            utils.fill_ps(obspath, pspath='wavd')
            # get center region
            utils.get_cnt(obspath, box_size=box_size)
            # get cut image
            utils.get_img(obspath, cntpath='cnt.reg')

if __name__ == "__main__":
    main()
