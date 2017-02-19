# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Copy samples.
"""

import os
import argparse

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(description="Copy samples.")
    # Parameters
    parser.add_argument("inpath", help="path of the folder saving samples")
    parser.add_argument("outpath", help="path to save copied samples")
    args = parser.parse_args()

    inpath = args.inpath
    outpath = args.outpath

    if not os.path.exists(inpath):
        print("The inpath does not exist.")
        return

    samples = os.listdir(inpath)
    for obs in samples:
        obspath = os.path.join(inpath, obs)
        obsout = os.path.join(outpath, obs)
        # Judge existance of the path
        if not os.path.exists(obspath):
            print("The observation %s does not exist." % obspath)
        else:
            print("Processing on sample %s ..." % obs)
            evtin = os.path.join(obspath, "img_beta.fits")
            evtout = os.path.join(obsout,"img_beta.fits")
            cavin = os.path.join(obspath,"beta.fits")
            cavout = os.path.join(obsout,"beta.fits")
            #cntin = os.path.join(obspath, "cnt.reg")
            #cntout = os.path.join(obsout, "cnt.reg")
            # copy
            # os.system("mkdir %s" % (obsout))
            os.system("cp %s %s" % (evtin, evtout))
            os.system("cp %s %s" % (cavin, cavout))
            # os.system("cp %s %s" % (cntin, cntout))

if __name__ == "__main__":
    main()
