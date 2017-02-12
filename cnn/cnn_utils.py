# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Some utils for preprocessing on the samples for CNN input.

References
----------
[1] Angular diameter distance,
    https://en.wikipedia.org/wiki/Angular_diameter_distance
[2] astroquery.Ned
    http://astroquery.readthedocs.io/en/latest/ned/ned.html

Methods
-------
get_redshift:
    Fetch redshift of the sample from the NED.
calc_rate:
    Calculate the rate, kpc/px.
reg_exchange:
    Transform units of the region data.
gen_sample:
    Generate sample set for CNN training, and testing.
get_reg:
    Get image coordinates of cavities.
get_mark:
    Mark the pixels with label of cavity and background.
"""

import astroquery
from astroquery.ned import Ned
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au
import numpy as np
import os
from astropy.io import fits
from scipy.misc import imsave


def get_redshift(objname):
    """
    Fetch redshift of the sample from the NED.

    Input
    -----
    objname: str
        Name of the object (sample).

    Output
    ------
    redshift: float64
        The redshift.
    """
    try:
        obj_table = Ned.query_object(objname)
        redshift = obj_table['Redshift'].data
        # get redshift
        redshift = float(redshift.data)
    except astroquery.exceptions.RemoteServiceError:
        print("Something wrong when feching %s." % objname)
        return -1

    return redshift

def calc_rate(redshift):
    """Calculate the rate, kpc/px."""
    # Init
    # Hubble constant at z = 0
    H0 = 71.0
    # Omega0, total matter density
    Om0 = 0.27
    # Cosmo
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    # Angular diameter distance, [Mpc]
    dA = cosmo.angular_diameter_distance(redshift)
    # Resolution of Chandra
    rate_px = 0.492 * au.arcsec #[arcmin/px]
    rate_px_rad = rate_px.to(au.rad)
    # rate
    rate = dA.value * rate_px_rad.value # [Mpc/px]

    return rate

def reg_exchange(regpath, rate, unit='kpc'):
    """Transform units of the region data.

    Input
    -----
    regpath: str
        Path of the region file.
    rate: double
        Mpc/px
    unit: str
        Unit of the params in the region, default as kpc.
    """
    # Init
    regname = regpath[:-4]
    regchg = regname + '_chg.reg'

    # Open files
    f_raw = open(regpath, 'r')
    if os.path.exists(regchg):
        os.remove(regchg)
    f_chg = open(regchg,'a')

    if unit == 'kpc':
        unit_rate = 0.001
    else:
        unit_rate = 1
    # read lines
    lines = f_raw.readlines()
    for l in lines:
        l.replace('\n','')
        params = l.split(' ')
        maj_axis = float(params[0]) * unit_rate / rate
        min_axis = float(params[1]) * unit_rate / rate
        dist = float(params[2]) * unit_rate / rate
        # Input to regchg
        f_chg.write("%f\t%f\t%f\n" % (maj_axis, min_axis, dist))

def reg_recover(regpath, rate, unit='kpc'):
    """Recover the region data from px to Mpc.

    Input
    -----
    regpath: str
        Path of the region file.
    rate: double
        Mpc/px
    unit: str
        Unit of the params in the region, default as kpc.
    """
    # Init
    regname = regpath[:-8]
    regchg = regname + '.reg'

    # Open files
    f_raw = open(regpath, 'r')
    if os.path.exists(regchg):
        os.remove(regchg)
    f_chg = open(regchg,'a')

    if unit == 'kpc':
        unit_rate = 0.001
    else:
        unit_rate = 1
    # read lines
    lines = f_raw.readlines()
    for l in lines:
        l.replace('\n','')
        params = l.split('\t')
        maj_axis = float(params[0]) / unit_rate * rate
        min_axis = float(params[1]) / unit_rate * rate
        dist = float(params[2]) / unit_rate * rate
        # Input to regchg
        f_chg.write("%f %f %f\n" % (maj_axis, min_axis, dist))

def gen_sample(img, img_mark, rate=0.5, boxsize=10,px_over=5):
    """
    Generate samples by splitting the pixel classified image with provided     boxsize

    Input
    -----
    img: np.ndarray
        The 2D raw image
    img_mark: np.ndarray
        The 2D marked image
    rate: float
        The rate of cavity pixels in the box, belongs to (0,1), default as 0.5
    boxsize: integer
        Size of the box, default as 10
    px_over: integer
        Overlapped pixels, default as 5

    Output
    ------
    data: np.ndarray
        The matrix holding samples, each column represents one sample
    label: np.ndarray
        Labels with respect to samples, could be 0, 1, and 2.
    """
    # Init
    rows,cols = img.shape
    px_diff = boxsize - px_over
    # Number of boxes
    box_rows = int(np.floor(rows/px_diff)) - 1
    box_cols = int(np.floor(cols/px_diff)) - 1
    # init data and label
    data = np.zeros((box_rows*box_cols, boxsize**2))
    label = np.zeros((box_rows*box_cols, 1))

    # Split
    for i in range(box_rows):
        for j in range(box_cols):
            sample = img[i*px_diff:i*px_diff+boxsize,
                         j*px_diff:j*px_diff+boxsize]
            label_mat = img_mark[i*px_diff:i*px_diff+boxsize,
                                 j*px_diff:j*px_diff+boxsize]
            data[i*box_rows+j,:] = sample.reshape((boxsize**2, ))
            rate_box = len(np.where(label_mat.reshape((boxsize**2,)) == 1)[0])
            # label[i*box_rows+j,0] = np.where(hist==hist.max())[0][0]
            rate_box = rate_box / boxsize**2
            if rate_box >= rate:
                label[i*box_rows+j,0] = 1
            else:
                label[i*box_rows+j,0] = 0

    return data,label

def get_reg(cntpath,cavpath):
    """
    Get image coordinates of cavities

    Input
    =====
    cntpath: str
        Path of the center region
    cavpath: str
        Path of the cavity region
    """
    # Get center phycial coordinate
    fp_cnt = open(cntpath,'r')
    cnt_reg = fp_cnt.readline()
    cnt_reg = cnt_reg[4:-1]
    cnt_list = cnt_reg.split(',')
    cnt_x = float(cnt_list[0])
    cnt_y = float(cnt_list[1])
    cnt_box = float(cnt_list[2])
    fp_cnt.close()

    # Comp coordinates
    cmp_x = cnt_x - cnt_box / 2
    cmp_y = cnt_y - cnt_box / 2

    # Transform cavity regions
    fp = open(cavpath,'r')
    cav_lines = fp.readlines()
    cav_mat = []

    for l in cav_lines:
        l1 = l.replace('\n', '')
        l1 = l1[8:-1]
        param = l1.split(',')
        cav_x = float(param[0]) - cmp_x
        cav_y = float(param[1]) - cmp_y
        cav_maj = float(param[2])
        cav_min = float(param[3])
        cav_rot = float(param[4])
        # push into cav_mat
        cav_mat.append([int(cav_x)-1, int(cav_y)-1, cav_maj, cav_min, cav_rot])

    return cav_mat

def get_mark(imgpath, cav_mat, savepath=None):
    """
    Mark the image with cavity region

    Input
    =====
    imgpath: str
        Path of the image, .fits file
    cav_mat: list
        List of the cavities

    Output
    ======
    img_mark: np.ndarray
        The marked image, pixels in cavities are marked as one,
        the rest are set as zero.
    """
    # load raw image
    hdulist = fits.open(imgpath)
    img = hdulist[0].data
    rows,cols = img.shape

    # Init mark image
    img_mark = np.zeros((rows,cols))

    # Fill
    for cav in cav_mat:
        # focus points
        ang = cav[4] / 180 * np.pi
        maj = cav[2]
        min = cav[3]
        c = np.sqrt(maj**2 - min**2)
        f1 = np.array([c,0])
        f2 = np.array([-c,0])
        # rotate
        rot_mat = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
        f1 = np.dot(rot_mat, f1) + np.array([cav[0], cav[1]])
        f2 = np.dot(rot_mat, f2) + np.array([cav[0], cav[1]])
        # mark
        for i in range(cols):
            for j in range(rows):
                d = (np.sqrt((i-f1[0])**2 + (j-f1[1])**2) +
                        np.sqrt((i-f2[0])**2 + (j-f2[1])**2))
                if d <= 2*maj:
                    img_mark[j,i] = 1

    if savepath != None:
        imsave(savepath, np.flipud(img_mark))

    return img_mark

