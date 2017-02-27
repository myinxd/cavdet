# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Get unsharp masked image.

1. convolve with 2 pixel gaus
2. convolve with 10 pixel gaus
3. divide 2 by 10
"""

import os
import argparse

from astropy.io import fits

'''
def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Generate unsharp mask images.")
    # Parameters
    parser.add_argument("obspath", help="path of the observation")
    args = parser.parse_args()

    obspath = args.obspath

    # Judge existance of the path
    if not os.path.exists(obspath):
        print("The observation %s does not exist." % obspath)
        return

    path_small = os.path.join(obspath, 'img_small.fits')
    path_large = os.path.join(obspath, 'img_large.fits')
    imgpath = os.path.join(obspath, 'img_cut.fits')
    imgumpath = os.path.join(obspath,'img_um.fits')
    # gen the small scale smoothed image
    print("aconvolve %s %s 'lib:gaus(2,5,1,2,2)'" % (imgpath, path_small))
    os.system("aconvolve %s %s 'lib:gaus(2,5,1,2,2)'" % (imgpath, path_small))
    # gen the large scale smoothed image
    print("aconvolve %s %s 'lib:gaus(2,5,1,10,10)'" % (imgpath, path_large))
    os.system("aconvolve %s %s 'lib:gaus(2,5,1,10,10)'" % (imgpath, path_large))
    # gen um image
    """
    h_small = fits.open(path_small)
    h_large = fits.open(path_large)
    img_um = h_small[0].data / h_large[0].data
    h = h_small
    h[0].data = img_um
    h.writeto(imgumpath)
    """
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
            path_small = os.path.join(obspath, 'img_small.fits')
            path_large = os.path.join(obspath, 'img_large.fits')
            imgpath = os.path.join(obspath, 'img_cut.fits')
            imgumpath = os.path.join(obspath, 'img_um.fits')
            '''
            # gen the small scale smoothed image
            print("aconvolve %s %s 'lib:gaus(2,5,1,2,2)' clobber=yes" % (imgpath, path_small))
            os.system("aconvolve %s %s 'lib:gaus(2,5,1,2,2)' clobber=yes" % (imgpath, path_small))
            # gen the large scale smoothed image
            print("aconvolve %s %s 'lib:gaus(2,5,1,10,10)' clobber=yes" % (imgpath, path_large))
            os.system("aconvolve %s %s 'lib:gaus(2,5,1,10,10)' clobber=yes" % (imgpath, path_large))
            # gen um image
            '''
            h_small = fits.open(path_small)
            h_large = fits.open(path_large)
            img_um = h_small[0].data / h_large[0].data
            h = h_small
            h[0].data = img_um
            h.writeto(imgumpath, clobber=True)


if __name__ == "__main__":
    main()
