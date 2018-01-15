# Copyright (C) 2017 Zhixian MA<zx@mazhixian.me>

"""
Utilities for levelset based cavitiy detection.
"""

import os
import re
import pickle
import numpy as np

def load_params(filepath):
    """Load parameters from user defined config file.

    input
    =====
    filepath: str
        Path of the file saving parameters

    output
    ======
    params_dict: dict
        A dictionary of the parameters

    format
    ======
    group:xxx
    param1:xxx1
    param2:xxx2
    param3:xxx3

    group:yyy
    param1:yyy1
    param2:yyy2
    param3:yyy3

    Note
    ====
    We do not support indent at this moment in the config file.

    Reference
    =========
    [1] Strip '''\n'''
    http://blog.csdn.net/jfkidear/article/details/7532293
    """
    params_dict = {}
    try:
        with open(filepath,"r") as fp:
            lines = fp.readlines()
    except:
        print("Something wrong when reading the config file.")
        return None
    # load params
    for l in lines:
        # strip
        l = l.strip("\n")
        # group
        if re.findall("group",l):
            grp_key = l.split(":")[-1]
            params_dict[grp_key] = {}
        elif l == '':
            continue
        else:
            param_key = l.split(":")[0]
            param = l.split(":")[-1]
            param = param.split(" ")
            if len(param) == 1:
                params_dict[grp_key][param_key] = param[0]
            else:
                params_dict[grp_key][param_key] = param

    return params_dict

def save_params(params_dict, savepath):
    """Save the loaded parameters

    input
    =====
    params_dict: dictionary
        A dictionary of the parameters
    savepath: str
        Path to save the file
    """
    with open(savepath, 'w') as fp:
        pickle.dump(fp,params_dict)

def genBetaModel(matshape, cen, betaparam):
    """
    Generate beta model with given parameters

    inputs
    ======
    matshape: tuple or list
        Shape of the matrix
    cen: tuple or list
        Location of the center pixel
    betaparam: dict
        Parameters of the beta function
        { "A": float,
          "r0": float,
          "theta": float,
          "beta": float,
          "majaxis": float,
          "minaxis": float,
        }

    output
    ======
    matbeta: np.ndarray
        The matrix with modelled beta values
    """
    if len(betaparam) != 6:
        print("There might be short of parameter.")
        return None
    # Init matbeta
    matbeta = np.zeros(matshape)

    # load paramters
    A = betaparam['A']
    r0 = betaparam['r0']
    theta = betaparam['theta']
    beta = betaparam['beta']
    majaxis = betaparam['majaxis']
    minaxis = betaparam['minaxis']
    ecc = majaxis / minaxis # eccentricity

    # Generate meshgrids
    X = np.linspace(1, matshape[0], matshape[0])
    Y = np.linspace(1, matshape[1], matshape[1])

    # anti-clock
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]])

    # Calc
    for j, x in enumerate(X):
        for i,y in enumerate(Y):
            x_r = x - cen[0]
            y_r = y - cen[1]
            r = np.matmul(rot, np.array([x_r, y_r]))
            r = r[0]**2 + r[1]**2 * ecc**2
            matbeta[i, j] = A * (1 + r/r0**2)**(-np.abs(beta))

    return matbeta

def genCavDepression(matbeta, cen, cavparam, angbeta, deprate):
    """
    Mock cavities on the beta model with depression, default with two caves

    inputs
    ======
    matbeta: np.ndarray
        The beta model matrix
    cen: tuple or list
        Location of the center
    angbeta: double
        Roration angle of the beta model
    cavparam: dict
        Paramters of the cavities
        {"majaxis": float,
         "minaxis": float,
         "theta": float,
         "phi": float,
         "dist": float,
        }
    deprate: float
        Rate of depression of cavities in the beta model

    output
    ======
    matcav: np.ndarray
        The matrix with cavity added
    """
    # params
    majaxis = cavparam["majaxis"] / 2
    minaxis = cavparam["minaxis"] / 2
    theta = cavparam["theta"] + angbeta
    phi = cavparam["phi"]
    dist = cavparam["dist"]

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]])
    rot1 = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi),np.cos(phi)]])
    rot2 = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi),np.cos(phi)]])

    matcav = matbeta.copy()
    matshape = matbeta.shape
    X = np.linspace(1, matshape[0], matshape[0])
    Y = np.linspace(1, matshape[1], matshape[1])
    # cavone
    c = np.sqrt(majaxis**2 - minaxis**2)
    core1 = np.matmul(rot, np.array([dist, 0]))
    core2 = np.matmul(rot, np.array([-dist, 0]))
    print(core1)
    print(core2)
    F1_c1 = core1 + np.matmul(rot1,np.array([c,0]))
    F2_c1 = core1 + np.matmul(rot1,np.array([-c,0]))
    F1_c2 = core2 + np.matmul(rot2,np.array([c,0]))
    F2_c2 = core2 + np.matmul(rot2,np.array([-c,0]))
    print(F1_c1)
    print(F2_c1)

    # get depressions
    for j, x in enumerate(X):
        for i, y in enumerate(Y):
            x_r = x - cen[0] # col
            y_r = y - cen[1] # row
            # in cav one?
            d1_c1 = np.sqrt((x_r - F1_c1[0])**2 + (y_r - F1_c1[1])**2)
            d1_c2 = np.sqrt((x_r - F2_c1[0])**2 + (y_r - F2_c1[1])**2)
            d1 = d1_c1 + d1_c2
            # in cav two?
            d2_c1 = np.sqrt((x_r - F1_c2[0])**2 + (y_r - F1_c2[1])**2)
            d2_c2 = np.sqrt((x_r - F2_c2[0])**2 + (y_r - F2_c2[1])**2)
            d2 = d2_c1 + d2_c2
            # Get indices of pixles in the cavities
            if d1 <= 2*majaxis:
                # rotate
                matcav[i, j] *= (1 - deprate)
            elif d2 <= 2*majaxis:
                matcav[i, j] *= (1 - deprate)

    return matcav,rot1,rot2

def genCavProfile(matbeta, cen, cavparam, angbeta, deprate):
    """
    Generate profile of the cav

    inputs
    ======
    matbeta: np.ndarray
        The beta model matrix
    cen: tuple or list
        Location of the center
    angbeta: double
        Roration angle of the beta model
    cavparam: dict
        Paramters of the cavities
        {"majaxis": float,
         "minaxis": float,
         "theta": float,
         "phi": float,
         "dist": float,
        }
    deprate: float
        Rate of depression of cavities in the beta model

    output
    ======
    matcav: np.ndarray
        The matrix with cavity added
    """
    # params
    majaxis = cavparam["majaxis"] / 2
    minaxis = cavparam["minaxis"] / 2
    theta = cavparam["theta"] + angbeta
    phi = cavparam["phi"]
    dist = cavparam["dist"]

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]])
    rot1 = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi),np.cos(phi)]])
    rot2 = np.array([[np.cos(phi), -np.sin(phi)],
                    [np.sin(phi),np.cos(phi)]])

    matcav = matbeta
    matshape = matbeta.shape
    X = np.linspace(1, matshape[0], matshape[0])
    Y = np.linspace(1, matshape[1], matshape[1])
    # cavone
    c = np.sqrt(majaxis**2 - minaxis**2)
    core1 = np.matmul(rot, np.array([dist, 0]))
    core2 = np.matmul(rot, np.array([-dist, 0]))
    print(core1)
    print(core2)
    F1_c1 = core1 + np.matmul(rot1,np.array([c,0]))
    F2_c1 = core1 + np.matmul(rot1,np.array([-c,0]))
    F1_c2 = core2 + np.matmul(rot2,np.array([c,0]))
    F2_c2 = core2 + np.matmul(rot2,np.array([-c,0]))
    print(F1_c1)
    print(F2_c1)

    # get depressions
    row = []
    col = []
    height = []
    for j, x in enumerate(X):
        for i, y in enumerate(Y):
            x_r = x - cen[0] # col
            y_r = y - cen[1] # row
            # in cav one?
            d1_c1 = np.sqrt((x_r - F1_c1[0])**2 + (y_r - F1_c1[1])**2)
            d1_c2 = np.sqrt((x_r - F2_c1[0])**2 + (y_r - F2_c1[1])**2)
            d1 = d1_c1 + d1_c2
            # in cav two?
            d2_c1 = np.sqrt((x_r - F1_c2[0])**2 + (y_r - F1_c2[1])**2)
            d2_c2 = np.sqrt((x_r - F2_c2[0])**2 + (y_r - F2_c2[1])**2)
            d2 = d2_c1 + d2_c2
            # Get indices of pixles in the cavities
            if np.abs(d1 - 2*majaxis) <= 1:
                # rotate
                row.append(y)
                col.append(x)
                height.append(matcav[i, j]*(1 - deprate))
            elif np.abs(d2 - 2*majaxis) <= 1:
                row.append(y)
                col.append(x)
                height.append(matcav[i, j]*(1 - deprate))

    return np.array(row),np.array(col),np.array(height)

def genGaussFilter(imgshape,sigma):
    """Generate the Gaussian filter
    
    Inputs
    ======
    imgshape: tuple
        Shape of the Gaussian kernel
    sigma: float
        Sigma of the gaussian function
    
    Output
    ======
    gaussker: np.ndarray
        The gausian kernel
    """
    x_c = imgshape[1] // 2 
    y_c = imgshape[0] // 2
    x = np.arange(-x_c,x_c+1)
    y = np.arange(-y_c,y_c+1)
    X, Y = np.meshgrid(x,y)
    
    gaussker = (1.0 / (np.sqrt(2*np.pi)*sigma) * 
                np.exp(-(X**2)/(2*sigma**2)) * 
                np.exp(-(Y**2)/(2*sigma**2)))
    
    return gaussker

def getUnsharpMasking(img, sigma1=2.0, sigma2=10.0, kershape=(15, 15)):
    from scipy.signal import convolve2d
    """Do unsharp masking on the image"""
    # generate Gaussian kernels
    if kershape is None:
        kershape1 = (np.fix(sigma1*5), np.fix(sigma1*5))
        kershape2 = (np.fix(sigma2*5), np.fix(sigma2*5))
    else:
        kershape1 = kershape
        kershape2 = kershape
    gker1 = genGaussFilter(kershape1,sigma1)
    gker2 = genGaussFilter(kershape2,sigma2)
        
    # convlve
    img_c1 = convolve2d(img, gker1, mode='same')
    img_c2 = convolve2d(img, gker2, mode='same')
    
    if sigma1 >= sigma2:
        img_um = img_c1 - img_c2
    else:
        img_um = img_c2 - img_c1
    
    return img_um
