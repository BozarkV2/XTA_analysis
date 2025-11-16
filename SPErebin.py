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

def energyBin(spectrum, binParams, fundamental = 1030, Eaxis=None):
    
    if type(binParams)==dict:
        binW = float(binParams['Ebins']['binW'])
        if binParams['Ebins']['fundamental']:
            fundamental = binParams['Ebins']['fundamental']
    else: #assume bin width is being passed in directly
        binW = binParams
    
    if Eaxis is None:
        pixel = np.arange(0, spectrum.shape[0], step=1)
        harmPos = getHarmPos(spectrum[spectrum.shape[0]/2], pixel,'Select 72 eV, high to low energy harmonics, then 36 eV')
        harmMaxE = harmPos[0]
        harmMinE = harmPos[-1]
        harmPeaks = harmPos[1:-1]
        
        energy = np.arange(start = len(harmPeaks)* 2*fundamental, stop = 0 , step = -2*fundamental)
        energyCalib = np.polyfit(pixel[harmPeaks],energy,3)
        energyCalib[-1] = 72 - np.polyval(energyCalib, harmMaxE) + energyCalib[-1]
        
        # slope = (17.1-34.2)/(harmMaxE-harmMinE)
        # wavelengths = pixel*slope + (17.1 - slope*harmMaxE) 
        # energyCalib = 1239/wavelengths
    else:
        pixel = np.linspace(0, 1023,num=1024)
        slope = (Eaxis[-1]-Eaxis[0])/(pixel[0] - pixel[-1])
        energyCalib = pixel*slope + Eaxis[-1]  
    
    minE = np.polyval(energyCalib, pixel[0])
    maxE = np.polyval(energyCalib, pixel[-1])

    if binW == 0:
        Eaxis = np.polyval(energyCalib, pixel) 
        Binaxis = np.polyval(energyCalib, pixel) 
    else:
        Eaxis = np.arange(start=minE,
                          stop=maxE,
                          step=binW)
    
        Binaxis = np.arange(start=minE+binW/2,
                          stop=maxE+binW/2,
                          step=binW)
    
    bins = digitize(np.polyval(energyCalib, pixel) ,Binaxis)
    
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

