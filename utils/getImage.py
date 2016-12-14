# !/usr/bin/python3
# Copyright (C) 2016 Zhixian MA <zxma_sjtu@qq.com>
"""
    A tool to fetch images of radio astro-objects from the NED
    NASA/IPAC Extragalactic Database
    http://ned.ipac.caltech.edu/
"""

from astroquery.ned import Ned
import os
import wget
import argparse


def main():
    """
    Fetch images of astro-objects from NED

        References
        ----------
        [1] astroquery
            http://astroquery.readthedocs.io/en/latest/
    """
    # Init
    parser = argparse.ArgumentParser(
        description='A tool to fetch images of radio astro-objects from the NED')
    # parameters
    parser.add_argument("objname", help="file path of object list.")
    parser.add_argument("folderpath", help="path to save images.")
    parser.add_argument("imglist", help="file path to save images.")
    parser.add_argument("errlist", help="file path to save object names with errors.")
    args = parser.parse_args()

    # Init
    objname = args.objname
    folderpath = args.folderpath
    imglist = args.imglist
    errlist = args.errlist

    # fetch data
    f = open(objname, 'r')
    fs = open(imglist, 'w')
    fn = open(errlist, 'w')

    for sample in f:
        sample = sample[:-1]
        print("Sample name: %s\n" % sample)
        # fetch table
        img_url = Ned.get_image_list(sample)
        # save images
        # make direction
        sample_folder = os.path.join(folderpath, sample)
        sample_folder = sample_folder.replace(' ', '_')
        print(sample_folder)
        if not os.path.exists(sample_folder):
            os.mkdir(sample_folder)
        # save url info
        fs.write('%s' % sample)
        for url in img_url:
            print("Saving image %s\n" % url)
            wget.download(url, sample_folder)
            fs.write('%s\n' % (url))
        # unzip
        gzpath = os.path.join(sample_folder, '*.gz')
        os.system('gunzip -d ' + gzpath)
    f.close()
    fs.close()
    fn.close()

if __name__ == "__main__":
    main()
