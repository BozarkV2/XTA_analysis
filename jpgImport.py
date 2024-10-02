# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:54:31 2024

@author: Samth
"""

from PIL import Image
import numpy as np
import os.path as pth
import lmfit
import matplotlib.pyplot as plt

def jpgImport(file):
    
    img = Image.open(pth.abspath(file))
    img.load()
    
    tempArr = np.asarray(img, dtype="int32")
    return tempArr

def findCentroid(img,mag=1):
    
    pxl_to_um=3.75 * mag #um/pixel
    
    z_axis = np.mean(img[:,:1024,:],axis=2)-np.mean(img[0:100,0:100,:])
    x_axis = pxl_to_um*np.arange(0,z_axis.shape[1],1)
    y_axis = pxl_to_um*np.arange(0,z_axis.shape[0],1)
    
    x_grid, y_grid = np.meshgrid(x_axis,y_axis) 
    
    x_model = np.ravel(x_grid)
    y_model = np.ravel(y_grid)
    z_model = np.ravel(z_axis)
    
    model = lmfit.models.Gaussian2dModel()
    params = model.guess(z_model,
                         x=x_model,
                         y=y_model)
    
    # params['amplitude'].value=255
    params['sigmax'].value=100
    params['sigmay'].value=100
    
    result = model.fit(z_model,
                       x=x_model,
                       y=y_model,
                       params=params)
    
    lmfit.report_fit(result)
    
    centerx = result.params['centerx'].value
    centery = result.params['centery'].value
    
    idxx=np.argwhere(x_axis==(min(x_axis, key=lambda x:abs(x-centerx))))
    idxy=np.argwhere(y_axis==(min(y_axis, key=lambda x:abs(x-centery))))
    
    fig1,(ax_x,ax_y) = plt.subplots(1,2)
    
    ax_x.plot(x_axis,result.eval(x=x_axis,y=centery))
    ax_x.plot(x_axis,z_axis[idxy[0,0],:])
    
    ax_y.plot(y_axis,result.eval(x=centerx,y=y_axis))
    ax_y.plot(y_axis,z_axis[:,idxx[0,0]])
    
    fig2,ax = plt.subplots()
    
    ax.imshow(img)
    ax.plot(centerx/pxl_to_um,centery/pxl_to_um,'ro')
    
    fig3,ax2 = plt.subplots()
    
    ax2.imshow(np.reshape(result.eval(x=x_model,y=y_model),
                          z_axis.shape))
    
    return result

def overlap(pumpFile,probeFile,mag=12):
    
    pumpImg = jpgImport(pumpFile)
    probeImg = jpgImport(probeFile)
    
    pumpCent = findCentroid(pumpImg,mag)
    probeCent = findCentroid(probeImg,mag)
    
    pxl_to_um=3.75 * mag #um/pixel
    
    pumpcenterx = pumpCent.params['centerx'].value
    pumpcentery = pumpCent.params['centery'].value
    
    probecenterx = probeCent.params['centerx'].value
    probecentery = probeCent.params['centery'].value
    
    z_axis = np.mean(pumpImg[:,:1024,:],axis=2)
    x_axis = pxl_to_um*np.arange(0,z_axis.shape[1],1)
    y_axis = pxl_to_um*np.arange(0,z_axis.shape[0],1)
    
    fig1,(ax_x,ax_y) = plt.subplots(1,2)
    
    ax_x.plot(x_axis,pumpCent.eval(x=x_axis,y=pumpcentery))
    ax_x.plot(x_axis,probeCent.eval(x=x_axis,y=probecentery))
    
    ax_y.plot(y_axis,pumpCent.eval(x=pumpcenterx,y=y_axis))
    ax_y.plot(y_axis,probeCent.eval(x=probecenterx,y=y_axis))
    
    print('Pump Centroid located at: '+ str(pumpcenterx)+ ', ' 
          + str(pumpcentery))
    
    print('Probe Centroid located at: '+ str(probecentery)+ ', ' 
          + str(probecenterx))
    
    return 
    