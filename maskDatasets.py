# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:58:20 2024

@author: Samth
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker 

def trimDataset(dataset,quiet=False):
    
    meanCnts = np.mean(dataset,axis=1)
    maskPos = getMask(meanCnts)
    
    mask = np.ones(meanCnts.shape,dtype=bool)
    if maskPos.shape[0]>0:
        mask[maskPos] = False
    
    if quiet:   
        fig,ax = plt.subplots()
        ax.plot(meanCnts)
        ax.plot(np.where(mask,meanCnts,0),'ro')
    
    return mask

def onclick(event):
    global offset
    offset = 1
    return 

def getMask(Xdata,title = 'Select measurements to omit'):
    """"Function to use to extract points for GVDcorrection. Only call on it's own if you want the points used for GVDcorrection.
    Returns a tuple of GVD points in energy and time"""
    fig,axs = plt.subplots()
    axs.plot(Xdata)
    plt.title(title)
    klicker = clicker(axs,['Scans'],markers=["x"],linestyle="-",colors=["red"])
    plt.show()
        
    cid = fig.canvas.mpl_connect('close_event', onclick)
    while not 'offset' in globals():
        plt.pause(2)
    
    if 'offset' in globals():
        global offset
        del offset

    fig.canvas.mpl_disconnect(cid)
    
    scan = klicker.get_positions()['Scans']
    if len(scan)>0:
        idx = np.array(np.round(scan[0],0),dtype='int')
    else:
        idx=np.empty((0))
    
    return idx