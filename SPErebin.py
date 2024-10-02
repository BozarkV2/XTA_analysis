# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:52:32 2024

@author: Samth
"""

from numpy import histogram,digitize,divide,bincount
import numpy as np
from configparser import ConfigParser
from numpy.lib.scimath import sqrt
from FluxNormalization import getHarmPos

def pixelBin(speImage,cropROI = []):
    
    if len(cropROI)>1:
        tempBin = np.sum(speImage[np.min(cropROI):
                                  np.max(cropROI),:], axis=0)
    else:
        tempBin = np.sum(speImage, axis=0)
    
    return tempBin

def energyBin(spectrum, binParams,Eaxis=None):
    
    if type(binParams)==dict:
        binW = float(binParams['Ebins']['binW'])
    else: #assume bin width is being passed in directly
        binW = binParams
    
    if Eaxis is None:
        pixel = np.linspace(0, 1023,num=1024)
        harmPos = getHarmPos(spectrum[0], pixel,'Select high then low energy harmonic')
        harmMaxE = harmPos[0]
        harmMinE = harmPos[1]
        slope = (17.1-34.2)/(harmMaxE-harmMinE)
        wavelengths = pixel*slope + (17.1 - slope*harmMaxE) 
        energyCalib = 1239/wavelengths
    else:
        pixel = np.linspace(0, 1023,num=1024)
        slope = (Eaxis[-1]-Eaxis[0])/(pixel[0] - pixel[-1])
        energyCalib = pixel*slope + Eaxis[-1]  
    
    minE = np.min(energyCalib)
    maxE = np.max(energyCalib)

    if binW == 0:
        Eaxis = energyCalib  
        Binaxis = energyCalib
    else:
        Eaxis = np.arange(start=minE,
                          stop=maxE,
                          step=binW)
    
        Binaxis = np.arange(start=minE+binW/2,
                          stop=maxE+binW/2,
                          step=binW)
    
    bins = digitize(energyCalib,Binaxis)
    
    if len(spectrum.shape)>1:
        spectraBin=np.zeros((Eaxis.shape[0],spectrum.shape[0]))
        stdBin=np.zeros((Eaxis.shape[0],spectrum.shape[0]))
    else:
        spectraBin = np.zeros(Eaxis.shape)
        stdBin=np.zeros(Eaxis.shape)
    
    for idx,i in enumerate(Eaxis):
        if len(spectrum.shape)>1:
            spectraBin[idx]= np.sum(spectrum[:,bins == idx],axis=1)
            stdBin[idx]= np.std(spectrum[:,bins == idx],axis=1)
        else:
            spectraBin[idx]= np.sum(spectrum[0,bins == idx])
            stdBin[idx]= np.std(spectrum[0,bins == idx])
            
    return spectraBin, stdBin, Eaxis
