# Copyright (C) 2016 Zhixian MA <zxma_sjtu@qq.com>

"""
Transform cavities regions of samples selected from paper
arXiv-1610.03487.

Reference
---------
[1] J. Shin, J. Woo, and, J. Mulchaey
    "A systematic search for X-ray cavities in galaxy clusters,
    groups, and elliptical galaxies"
    arXiv-1610.03487
"""
import os
import argparse

import cnn_utils as utils

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Region transform")
    # parameters
    parser.add_argument("inpath", help="path holding the samples.")
    args = parser.parse_args()

    # Judge existance of the paths
    try:
        samples = os.listdir(args.inpath)
    except IOError:
        print("Inpath does not exist.")
        return

    # Transform region
    fp = open("sample_z.log",'a')
    for s in samples:
        print("Processing on %s..." % s)
        # get redshift
        z = utils.get_redshift(s)
        if z != -1:
            # calc rate
            rate = utils.calc_rate(z)
            fp.write("%s\t%f\t%f\n" % (s, z, rate))
            # region exchange
            sample_path = args.inpath + '/' + s + '/'
            regpath = os.path.join(sample_path, 'cavities.reg')
            print(regpath)
            utils.reg_exchange(regpath, rate, unit='kpc')
        else:
            pass

if __name__ == "__main__":
    main()
