# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qqq.com>

"""
Draw circles on the image to observe the cavities, radii are from Shin et al's 2016
"""

import os
import argparse

import utils_pre as utils

'''
def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Get circular regions.")
    # Parameters
    parser.add_argument("obspath", help="path of the input")
    args = parser.parse_args()

    obspath = args.obspath

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    # get regions
    utils.cmp_cav(obspath)
'''

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Get circular regions.")
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
            # get regions
            print("Processing on sample %s ..." % obs)
            utils.cmp_cav(obspath)


if __name__ == "__main__":
    main()
