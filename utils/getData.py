# !/usr/bin/python3
# Copyright (C) 2016 Zhixian MA <zxma_sjtu@qq.com>
"""
    A tool to query photometry of radio astro-objects from the NED
    NASA/IPAC Extragalactic Database
    http://ned.ipac.caltech.edu/
"""

from astroquery.ned import Ned
import numpy as np
import astroquery
import argparse


def main():
    """
    Fetch radio infos from NED

        References
        ----------
        [1] astroquery
            http://astroquery.readthedocs.io/en/latest/
    """
    # Init
    parser = argparse.ArgumentParser(
        description='A tool to query photometry of radio astro-objects from the NED')
    # parameters
    parser.add_argument("objname", help="file path of object list.")
    parser.add_argument("dataname", help="file path to save query result.")
    parser.add_argument("freq", help="Upper limit of frequency.")
    parser.add_argument("errlist", help="file path to save object names with errors.")
    args = parser.parse_args()

    # get arguments
    objname = args.objname
    dataname = args.dataname
    freq = float(args.freq)
    errlist = args.errlist

    # fetch data
    f = open(objname, 'r')
    fs = open(dataname, 'w')
    fn = open(errlist, 'w')
    fs.write("Name\tBand\tFlux\tUncertainty\tUnits\tRefcode\n")
    for sample in f:
        print("Sample name: %s" % sample[:-1])
        # fetch table
        try:
            obj_table = Ned.get_table(sample, table='photometry')
        except astroquery.exceptions.RemoteServiceError:
            fn.write('%s\n' % sample)
            continue
        # find radio info
        freq_list = np.array(obj_table['Frequency'])
        freq_idx = np.where(freq_list <= freq)[0]
        print(freq_idx)
        if len(freq_idx) >= 1:
            # Judge measurements and uncertainties
            uncer = obj_table['Uncertainty'][freq_idx]
            if len(uncer) == 1:
                freq_str = obj_table['Observed Passband'][freq_idx[0]]
                flux = obj_table['Photometry Measurement'][freq_idx[0]]
                unit = obj_table['Units'][freq_idx[0]]
                ref = obj_table['Refcode'][freq_idx[0]]
                # bytes to str
                freq_str = str(freq_str, encoding='utf-8')
                unit = str(unit, encoding='utf-8')
                ref = str(ref, encoding='utf-8')
                fs.write("%s\t%s\t%f\t%s\t%s\t%s\n" %
                         (sample[:-1], freq_str, flux, str(uncer[0], encoding='utf-8'), unit, ref))
            else:
                for i in range(len(uncer)):
                    if len(uncer[i]):
                        freq_str = obj_table['Observed Passband'][freq_idx[i]]
                        flux = obj_table['Photometry Measurement'][freq_idx[i]]
                        unit = obj_table['Units'][freq_idx[i]]
                        ref = obj_table['Refcode'][freq_idx[i]]
                        # bytes to str
                        freq_str = str(freq_str, encoding='utf-8')
                        unit = str(unit, encoding='utf-8')
                        ref = str(ref, encoding='utf-8')
                        fs.write("%s\t%s\t%f\t%s\t%s\t%s\n" %
                                 (sample[:-1], freq_str, flux, str(uncer[i], encoding='utf-8'),
                                  unit, ref))
    f.close()
    fs.close()
    fn.close()

if __name__ == "__main__":
    main()
