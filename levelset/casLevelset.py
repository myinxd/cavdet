# Copyright (C) 2018 Zhixian MA <zx@mazhixian.me>

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.misc import imsave

from levelset import LevelSet

class printException(Exception):  
    """
    Ref: http://blog.csdn.net/kwsy2008/article/details/48468345
    """
    pass  

class CasLevelSet(LevelSet):
    """
    Piecewise constant levelset: cascaded case
    
    Inputs
    ======
    phi_pre: np.ndarray
        The previous levelset function
    imgshape: tuple 
        shape of the image to be segmented
    mu: float
        coefficient of the boundry
    nu: float
        coefficient of the segmented region
    lambda1: float
        coefficient of the internal region
    lambda2: float
        coefficient of the external region
    dt: float
        time interval
        
    Reference
    =========
    [1] Getreuer. P., "Chan-Vese Segmentation"
        http://dx.doi.org/10.5201/ipol.2012.g-cv
    """
    
    def __init__(self, phi_pre, imgshape, mu=1.0, nu=1.0, 
                 lambda1=1.0, lambda2=1.0, dt=0.1,
                 init_mode=None, radius=None, lev=0.0):
        """
        The initializer
        """
        self.phi_pre = phi_pre
        self.lev = lev
        super().__init__(imgshape,mu,nu,lambda1,lambda2,dt,init_mode,radius)
        
        # Init phi
        if self.init_mode is None:
            self.initPhi()
        elif self.init_mode == "cir":
            self.initPhi_cir(radius=self.radius)
        else:
            raise printException("InitModeError")
        
        
    def initPhi_cir(self, radius=None):
        """
        Init the phi function, i.e., the level set, circle case     
        """
        rows,cols = self.imgshape
        if radius is None:
            radius = min(rows, cols) // 4
        # Init
        self.phi = np.ones((rows, cols))
        y = np.arange(-rows//2, rows//2)
        x = np.arange(-cols//2, cols//2)
        X,Y = np.meshgrid(x,y)
        z = np.sqrt(X**2+Y**2)
        
        id_row,id_col = np.where(z > radius)
        self.phi[id_row, id_col] = -1
        
        # mask
        id_pre_row, id_pre_col = np.where(self.phi_pre > self.lev)
        self.phi[id_pre_row, id_pre_col] = np.nan

     
    def initPhi(self):
        """
        Init the phi function, i.e., the level set
        
        Reference
        =========
        [1] Getreuer. P., "Chan-Vese Segmentation"
        http://dx.doi.org/10.5201/ipol.2012.g-cv
        """
        rows,cols = self.imgshape
        # Init
        x = np.arange(0, cols, 1)
        y = np.arange(0, rows, 1)
        X, Y = np.meshgrid(x,y)
        self.phi = np.sin(X*np.pi/5.0) * np.sin(Y*np.pi/5.0)
        
        # mask
        id_pre_row, id_pre_col = np.where(self.phi_pre > self.lev)
        self.phi[id_pre_row, id_pre_col] = np.nan    

        
    def calcCentroids(self, img):
        """Calculate centroids of the internal and external regions
           segmented by the levelset function.
        """
        phi = self.phi
        phi[np.isnan(self.phi)] = 0
        idx_c1r, idx_c1c = np.where(phi > 0)
        idx_c2r, idx_c2c = np.where(phi < 0)
        c1 = np.sum(img[idx_c1r, idx_c1r]) / (len(idx_c1r)+self.yita)
        c2 = np.sum(img[idx_c2r, idx_c2r]) / (len(idx_c2r)+self.yita)
        
        return c1,c2
    
    def calcSegmentation(self, img, niter=100, phi_total=1.0,
                         normflag=True, logflag=False):
        """Do segmentation"""
        if normflag:
            img = self.getNormalization(img, logflag=logflag)
        # calc the region centroids as constands
        self.c1, self.c2 = self.calcCentroids(img)
        # shrink the region
        id_pre_row, id_pre_col = np.where(self.phi_pre < self.lev)
        self.phi_margin = [id_pre_row.min(), id_pre_row.max(), 
                           id_pre_col.min(), id_pre_col.max()]
        #phi = self.phi[self.phi_margin[0]:self.phi_margin[1]+1,
        #               self.phi_margin[2]:self.phi_margin[3]+1]
        phi_mask = np.isnan(self.phi)
        # Iterate to optimize phi
        for it in range(niter):
            phidiffnorm = 0.0
            for j in range(self.phi_margin[0], self.phi_margin[1]+1):
                # top margin
                if j == 0:
                    idu = 0
                else:
                    idu = -1
                # bottom margin
                if j == self.imgshape[0] - 1:
                    idd = 0
                else:
                    idd = 1
                for i in range(self.phi_margin[2], self.phi_margin[3]+1):
                    # left margin
                    if i == 0:
                        idl = 0
                    else:
                        idl = -1
                    # right margin
                    if i == self.imgshape[1]-1:
                        idr = 0
                    else:
                        idr = 1
                    # if nan
                    if phi_mask[j+idu,i] or phi_mask[j+idd,i] or phi_mask[j,i+idl] or phi_mask[j,i+idr]:
                        continue
                    else:
                        # main body
                        Delta = self.dt/(np.pi*(1+self.phi[j,i]*self.phi[j,i]))
                        phi_x = self.phi[j,i+idr]-self.phi[j,i]
                        phi_y = (self.phi[j+idd,i]-self.phi[j+idu,i])/2.0
                        IDivR = 1.0/np.sqrt(self.yita+phi_x**2+phi_y**2)
                        phi_x = self.phi[j,i]-self.phi[j,i+idl]
                        IDivL = 1.0/np.sqrt(self.yita+phi_x**2 + phi_y**2)
                        phi_x = (self.phi[j,i+idr] - self.phi[j,i+idl])/2.0
                        phi_y = self.phi[j+idd,i] - self.phi[j,i]
                        IDivD = 1.0/np.sqrt(self.yita + phi_x**2 + phi_y**2)
                        phi_y = self.phi[j,i] - self.phi[j+idu,i]
                        IDivU = 1.0/np.sqrt(self.yita + phi_x**2 + phi_y**2)
                    
                        # Distances
                        dist1 = (img[j,i] - self.c1)**2
                        dist2 = (img[j,i] - self.c2)**2
                    
                        # Update phi at current point j,i
                        phi_last = self.phi[j,i]
                        self.phi[j,i] = ((self.phi[j,i] + 
                                          Delta*(self.mu*
                                                (self.phi[j,i+idr]*IDivR +
                                                 self.phi[j,i+idl]*IDivL + 
                                                 self.phi[j+idd,i]*IDivD + 
                                                 self.phi[j+idu,i]*IDivU
                                                )- 
                                                self.nu - self.lambda1 * dist1 + 
                                                self.lambda2 * dist2)
                                         ) / 
                                         (1.0 + Delta*self.mu*(IDivR+IDivL+IDivD+IDivU)))
                        phidiff = self.phi[j,i] - phi_last
                        phidiffnorm += phidiff ** 2
                    
            if phidiffnorm <= phi_total and it >= 2:
                break
                    
            # update c1 and c2 
            self.c1,self.c2 = self.calcCentroids(img)
            
            if np.mod(it, 5) == 0:
                t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time())) 
                imsave("./tmp/phi2_%d.png" % it, self.phi)
                print("[%s] Iter: %d     PhiDiffNorm: %.5f" % (t,it,phidiffnorm))
  
    def drawResult(self,img,normflag=True,logflag=False):
        """draw the segmentation curve"""
        if normflag:
            img = self.getNormalization(img, logflag=logflag)
         
        plt.rcParams["figure.figsize"] = [10.0, 4.0]
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1])

        ax0 = plt.subplot(gs[0])
        ax0 = plt.imshow(img)
        ax0 = plt.contour(self.phi,levels=[0.0]);
        ax1 = plt.contour(self.phi_pre, levels=[self.lev])
        plt.xlabel("horizontal")
        plt.ylabel("vertical")
        
        img_seg = np.zeros(img.shape)
        img_seg[self.phi>0] = 1
        ax0 = plt.subplot(gs[1])
        # ax1 = plt.contour(self.phi)
        ax1 = plt.imshow(img_seg)
        plt.xlabel("horizontal")
        plt.ylabel("vertical")
