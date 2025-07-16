# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:50:53 2021

@author: Bozark
"""

from XUVclass import XASdata,XTAdata
import XUVplotter as Xplt
import matplotlib.pyplot as plt
import os,time
import os.path as pth
import numpy as np
from mpl_point_clicker import clicker 
import read_SPE as spe
from SPErebin import energyBin,pixelBin
from FluxNormalization import normHarms,normOddHarms,normHalfHarms,normHalfOddHarms
from statMethods import fitPCA,airPCA

def init(DIRECTORY):
    
    workspace_dict={
    'DIRECTORY':DIRECTORY, #Where is your data stored? Must be file path
    'all_subdirs':True, #Whether you want to import all directories in a folder
    'CropTF':False, #do you want to crop the y-axis of the images
    'Eaxis':None, #give an energy array if you already have an energy axis on hand
    'binW':0.2, #the energy bin width in eVs
    'AlignE':False, #do you want to dynamically adjust the energy axis, primarily for ground states
    'NormFlux':False, #Do you want to normalize the flux between scans?
    'normBckTF':False, #Do you want to scale the background spectra to the data?
    'normMethod':normHalfHarms, #what method to use during normalization, generally stick with normHalfHarms
    'dataOrg':'Alt', #how was the data collected? 'Alt' for alternating (data/ref/data/ref...), or 'NSSN' (data/ref/ref/data...)
    'pcaTF':False, #Do you want to do principal component analysis?
    'pcaMethod':fitPCA, #what method do you want to use for PCA?
    'pcaArgs':{'components':20, 
               'maxIter':40, #only for airPCA
               'threshold':0.95}
    }
    
    return workspace_dict

def mainStatic(initDict,XUVlist = None,refBack=None,dataBack=None, **kwargs):
    """"This is the main function for importing and processing XUV data. 
    
    If you set all_subdirs to True, then the function will loop through all subfolders,
    and return a list of lists with all the data in each subfolder.
    """
    
    if XUVlist is None:
        XUVlist = importDir(initDict)
        
    XAS = XASdata(XUVlist,options=initDict,
                  dataBck=dataBack,refBck=refBack)
    
    Xplt.plotXAS(XAS)
    Xplt.plotStats(XAS)
    
    return XAS

def mainTrans(initDict,time,XUVlist = None,refBack=None,dataBack=None,
             **kwargs):
    """"This is the main function for importing and processing XUV data. It will automatically find
    files with XMCD
    
    If you set all_subdirs to True, then the function will loop through all subfolders,
    and return a list of lists with all the data in each subfolder.
    """
    plt.close('all')
    
    if XUVlist is None:
        XUVlist = importDir(initDict)
    # dataMask = trimDataset(pumpON)    
    
    XTA = XTAdata(XUVlist,time,options=initDict,
                  dataBck=dataBack,refBck=refBack)
    
    Xplt.plotXTAdata(XTA,
                     color_max=np.max(XTA.trans2D),
                     color_min=np.min(XTA.trans2D))
    
    return XTA

def mainMCD(initDict, XUVlist = None, refBack=None, dataBack=None,
             **kwargs):
    """"This is the main function for importing and processing XUV data. It will automatically find
    files with XMCD
    
    If you set all_subdirs to True, then the function will loop through all subfolders,
    and return a list of lists with all the data in each subfolder.
    """
    plt.close('all')
    
    if XUVlist is None:
        XUVlist = importDir(initDict)
    # dataMask = trimDataset(pumpON)    
    
    XMCD = XTAdata(XUVlist,[0],options=initDict,
                  dataBck=dataBack,refBck=refBack)
    
    Xplt.plotXAS(XMCD)
    Xplt.plotStats(XMCD)
    
    return XMCD

def importDir(initDict,subdirs=False):
    
    if subdirs is True:
        rawImgs=[]
        if type(initDict) is dict:
            for path in os.scandir(initDict['DIRECTORY']):
                if path.is_dir():
                    rawImgs.append(importDir(path))
        else:
            for path in os.scandir(initDict['DIRECTORY']):
                if path.is_dir():
                    rawImgs.append(importDir(path))
        
    if type(initDict) is dict:
        fileGen= (file for file in os.scandir(initDict['DIRECTORY']) 
                  if file.path.endswith('.spe'))
        if initDict['CropTF']:
            cropList = []
            cropTF = True
        else:
            cropList = [[],[]]
            cropTF = False
    else: 
        fileGen = (file for file in os.scandir(initDict) 
                   if file.path.endswith('.spe'))
        cropList = [[],[]]
        cropTF = False
        
    rawImgs = []
    idx=0
    
    for file in fileGen:
        print(file.name)
        for data in spe.loadGen(file):
            if cropTF and idx<2:
                cropROI = getCropPos(data)
                if idx ==0:
                    cropList.append(cropROI)
                elif idx ==1:
                    cropList.append(cropROI)
            
            rawImgs.append(pixelBin(data,
                                    cropList[np.mod(idx,2)]))
            idx+=1
        
    return rawImgs

def getCropPos(Xdata,title='Select harmonic peaks'):
    """"Function to use to extract points for GVDcorrection. Only call on it's own if you want the points used for GVDcorrection.
    Returns a tuple of GVD points in energy and time"""
    fig,axs = plt.subplots()
    axs.imshow(Xdata)
    plt.title(title)
    klicker = clicker(axs,['pos'],markers=["x"],linestyle="-",colors=["red"])
    plt.show(block=True)
        
    #cid = fig.canvas.mpl_connect('close_event', onclick)
    #while not 'offset' in globals():
    #    plt.pause(2)
    
    #if 'offset' in globals():
    #    global offset
    #    del offset

    #fig.canvas.mpl_disconnect(cid)
    
    pos = klicker.get_positions()['pos']
    idx = np.array(pos[:,1],dtype='int')
    
    return idx
