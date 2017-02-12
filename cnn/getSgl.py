# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qqq.com>

"""
Get single ccd image from the obseration and do point sources detect,
the first step of processing.

"""

import os
import argparse

import utils_pre as utils

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Preprocessing on the raw images.")
    # Parameters
    parser.add_argument("obspath", help="path of the input")
    parser.add_argument("ccd_id", default='7', help="ccd_id (default is 7)")
    args = parser.parse_args()

    obspath = args.obspath
    ccd_id = args.ccd_id

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    # get sub
    utils.get_sub(obspath, ccd_id)
    # get point sources
    utils.get_ps(obspath, pspath='wavd')

'''
def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Preprocessing on the raw images.")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    parser.add_argument("ccd_id", default='7', help="ccd_id (default is 7)")
    args = parser.parse_args()

    inpath = args.inpath
    ccd_id = args.ccd_id

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
            # get sub
            utils.get_sub(obspath, ccd_id)
            # get point sources
            utils.get_ps(obspath, pspath='wavd')
'''
if __name__ == "__main__":
    main()
