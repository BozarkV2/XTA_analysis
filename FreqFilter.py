# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:52:00 2024

@author: Samth
"""

import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker 

def freqFilter(data,harmonics,maxF=65,minF=45):
    
    dataFreq = fft(data)
    harmFreq = fft(harmonics)
    
    Freqfilt = np.where((harmFreq >minF) & (harmFreq <maxF),
                        0, dataFreq)
    
    Freqfilt = np.where((harmFreq >2*minF) & (harmFreq <2*maxF),
                        0, Freqfilt)
    
    filtData =np.real(ifft(Freqfilt))
    
    return filtData
    
def freqFilterChoose(data,harmonics,quiet=False):
    
    dataFreq = fft(data)
    harmFreq = np.abs(fft(harmonics))
    
    omitFreq = getMask(harmFreq,
                       np.abs(dataFreq),
                       title=""""Select Frequencies to Exclude:
                           must be in pairs to denote a range""")
    
    lowf= omitFreq[0::2]
    highf = omitFreq[1::2]
    OmitFreq = np.zeros(dataFreq.shape,dtype='csingle')
    for x,y in zip(lowf,highf):
        tempMask = np.where((harmFreq> x) & (harmFreq<y) ,
                            True,False)
        
        OmitFreq[tempMask] = dataFreq[tempMask]
        dataFreq[tempMask]= 0
    
    filtData =np.real(ifft(dataFreq))
    
    if not quiet and omitFreq.shape[0]>0:
        OmitData = np.real(ifft(OmitFreq))
        fig,(ax1,ax2) = plt.subplots(1,2,sharex=True)
        ax1.plot(harmonics,OmitData,label='Omitted Frequencies')
        ax2.plot(harmonics,filtData,label='Filtered Data')
        ax1.plot(harmonics,data,label='Original Data')
        ax1.legend()
        ax2.legend()
    
    return filtData

def onclick(event):
    global offset
    offset = 1
    return 

def getMask(Xdata,Ydata,title = 'Select measurements to omit'):
    """"Function to use to extract points for GVDcorrection. Only call on it's own if you want the points used for GVDcorrection.
    Returns a tuple of GVD points in energy and time"""
    fig,axs = plt.subplots()
    axs.plot(Xdata,Ydata,'bo-')
    plt.title(title)
    klicker = clicker(axs,['xaxis'],markers=["x"],linestyle="-",colors=["red"])
    plt.show()
        
    cid = fig.canvas.mpl_connect('close_event', onclick)
    while not 'offset' in globals():
        plt.pause(2)
    
    if 'offset' in globals():
        global offset
        del offset

    fig.canvas.mpl_disconnect(cid)
    
    freq = klicker.get_positions()['xaxis']
    if len(freq)>0:
        idx = np.array(freq[:,0])
    else:
        idx=np.empty((0))
    
    return idx