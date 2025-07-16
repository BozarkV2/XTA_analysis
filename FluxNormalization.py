# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:53:51 2024

@author: Samth
"""

import numpy as np
from numpy import invert, broadcast_to
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker 

def normOddHarms(spectrum,energy=None,quiet=True):
    
    if energy is None:
        energy = np.arange(0,1024,1)
    
    if len(spectrum.shape)>1:
        Eaxis = getHarmPos(spectrum[50], energy,'Select Harmonic Valleys for Flux Normalization')
    else:
        Eaxis = getHarmPos(spectrum, energy,'Select Harmonic Valleys for Flux Normalization')
        
    bins = np.digitize(energy, Eaxis)
    
    mask = np.where(np.mod(bins,2)==0,True,False)
    arrMask = broadcast_to(mask, spectrum.shape)
    
    if len(spectrum.shape)<2:
        evenInt = np.sum(spectrum[mask])
        oddInt = np.sum(spectrum[invert(mask)])
    else:
        evenCnts = np.sum(spectrum[:,mask],axis=1)
        oddCnts = np.sum(spectrum[:,invert(mask)],axis=1)
        
        evenInt = np.broadcast_to(evenCnts[:,np.newaxis],
                                  spectrum.shape)
        oddInt = np.broadcast_to(oddCnts[:,np.newaxis],
                                 spectrum.shape)
        
    normSpec = np.ones(spectrum.shape)
        
    np.divide(spectrum,
              evenInt,
              out = normSpec,
              where = arrMask)
                        
    np.divide(spectrum,
              oddInt,
              out = normSpec,
              where = invert(arrMask))
    
    if not quiet:
        fig,ax = plt.subplots()
        ax.plot(energy,normSpec)
        plt.vlines(Eaxis, np.min(normSpec), np.max(normSpec),'r-')
    
    return normSpec

def normHarms(spectrum,energy=None,quiet=True):
    
    if energy is None:
        energy = np.arange(0,1024,1)
        
    if len(spectrum.shape)<2:
        CntsArr = np.sum(spectrum)
    else:
        TotalCnts = np.sum(spectrum,axis=1)
        CntsArr = np.broadcast_to(TotalCnts[:,np.newaxis],
                                  spectrum.shape)
        
    normSpec = np.divide(spectrum,
                         CntsArr)
    
    if not quiet:
        fig,ax = plt.subplots()
        ax.plot(energy,normSpec)
    
    return normSpec

def normHalfHarms(spectrum,energy=None,quiet=True):
    
    if energy is None:
        energy = np.arange(0,1024,1)
        
    if len(spectrum.shape)>1:
        Eaxis = getHarmPos(spectrum[50], energy,title='select ROI(s), low x then High x, to integrate for Flux Normalization')
    else:
        Eaxis = getHarmPos(spectrum, energy,title='select ROI(s), low x then High x, to integrate for Flux Normalization')
    
    mask = np.where((energy>Eaxis[0]) & (energy<Eaxis[1]),
                    True,False)
    
    if len(spectrum.shape)<2:
        normFact = np.sum(spectrum,where=mask)
    else:
        TotalCnts = np.sum(spectrum,axis=1,where=mask)
        normFact= np.broadcast_to(TotalCnts[:,np.newaxis],
                                  spectrum.shape)
        
    normSpec = np.divide(spectrum,
                         normFact)
    
    if not quiet:
        fig,ax = plt.subplots()
        ax.plot(energy,normSpec)
    
    return normSpec

def normHalfOddHarms(spectrum,energy=None,quiet=True):
    
    if energy is None:
        energy = np.arange(0,1024,1)
    
    if len(spectrum.shape)>1:
        Eaxis = getHarmPos(spectrum[50], energy)
    else:
        Eaxis = getHarmPos(spectrum, energy)
        
    if len(spectrum.shape)>1:
        ROI = getHarmPos(spectrum[50], energy,title='select ROI(s), low x then High x, to exclude for Flux Normalization')
    else:
        ROI = getHarmPos(spectrum, energy,title='select ROI(s), low x then High x, to exclude for Flux Normalization')
    
    bins = np.digitize(energy, Eaxis)
    
    mask = np.where(np.mod(bins,2)==0,True,False)
    maskROI = np.where((energy>Eaxis[0]) & 
                       (energy<Eaxis[1]),False,True)
    arrMask = broadcast_to(mask, spectrum.shape)
    
    if len(spectrum.shape)<2:
        evenInt = np.sum(spectrum[mask & maskROI])
        oddInt = np.sum(spectrum[invert(mask) & maskROI])
    else:
        evenCnts = np.sum(spectrum[:,mask & maskROI],axis=1)
        oddCnts = np.sum(spectrum[:,invert(mask) & maskROI],axis=1)
        
        evenInt = np.broadcast_to(evenCnts[:,np.newaxis],
                                  spectrum.shape)
        oddInt = np.broadcast_to(oddCnts[:,np.newaxis],
                                 spectrum.shape)
        
    normSpec = np.ones(spectrum.shape)
        
    np.divide(spectrum,
              evenInt,
              out = normSpec,
              where = arrMask)
                        
    np.divide(spectrum,
              oddInt,
              out = normSpec,
              where = invert(arrMask))
    
    if not quiet:
        fig,ax = plt.subplots()
        ax.plot(energy,normSpec)
        plt.vlines(Eaxis, np.min(normSpec), np.max(normSpec),'r-')
    
    return normSpec

def normBackSub(spectrum, bck, 
                energy=None, Eaxis=None, quiet=True):
    
    if energy is None:
        energy = np.arange(0,1024,1)
        
    if len(spectrum.shape)>1 and Eaxis is None:
        Eaxis = getHarmPos(spectrum[:,0], energy,title='select ROI(s), low x then High x, to integrate for Background Normalization')
    elif Eaxis is None or len(Eaxis)<2:
        Eaxis = getHarmPos(spectrum, energy,title='select ROI(s), low x then High x, to integrate for Background Normalization')
    
    mask = np.where((energy>Eaxis[0]) & (energy<Eaxis[1]),
                    True,False)
    
    if len(spectrum.shape)<2:
        backFact = np.sum(spectrum,where=mask)
    else:
        TotalSpecCnts = np.sum(spectrum, axis=0, where=mask[:,np.newaxis])
        TotalBackCnts = np.sum(bck,where=mask)
        backFact = TotalSpecCnts/TotalBackCnts
        
    normSpec = spectrum - backFact[np.newaxis,:] * bck[:,np.newaxis]
    
    if not quiet:
        fig,ax = plt.subplots()
        ax.plot(energy,normSpec)
    
    return normSpec

def getHarmPos(Xdata,energy,title='Select harmonic peaks'):
    """"Function to use to extract points for GVDcorrection. Only call on it's own if you want the points used for GVDcorrection.
    Returns a tuple of GVD points in energy and time"""
    fig,axs = plt.subplots()
    axs.plot(energy,Xdata)
    plt.title(title)
    klicker = clicker(axs,['pos'],markers=["x"],linestyle="-",colors=["red"])
    plt.show(blocked=True)
    
    pos = klicker.get_positions()['pos']
    idx = np.array(pos[:,0])
    
    return idx
