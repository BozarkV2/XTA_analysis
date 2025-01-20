# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:50:53 2021

@author: Bozark
"""

from XUVclass import XASdata,XTAdata
import XUVplotter as Xplt
from XUVImport import dataImporter, load_Generator
import matplotlib.pyplot as plt
# import KineticFitFunctions as kff
# import SpectralFitFunctions as sff
import os,time
import os.path as pth
import numpy as np
from scipy.interpolate import interp1d
from lmfit import Parameters
from mpl_point_clicker import clicker 
import read_SPE as spe
from SPErebin import energyBin,pixelBin
from configparser import ConfigParser
from alignTime import alignTime
from normAbs import normAbs
from AbsCalcs import XUVtransAbs,XUVabs
from maskDatasets import trimDataset
from CrossECorrection import energyCorrect
from FluxNormalization import normHarms,normOddHarms,normHalfHarms,normHalfOddHarms
from FreqFilter import freqFilter,freqFilterChoose
from statMethods import fitPCA,airPCA

def init(DIRECTORY):
    
    workspace_dict={
    'DIRECTORY':DIRECTORY,
    'all_subdirs':True,
    'CropTF':False,
    'Eaxis':None,
    'binW':0.2,
    'AlignE':False,
    'NormFlux':False,
    'normBckTF':False,
    'normMethod':normHalfHarms,
    'dataOrg':'Alt',
    'pcaTF':False,
    'pcaMethod':fitPCA,
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

def mainMCD(initDict='',**kwargs):
    """"This is the main function for importing and processing XUV data. It will automatically find
    files with XMCD
    
    If you set all_subdirs to True, then the function will loop through all subfolders,
    and return a list of lists with all the data in each subfolder.
    """
    if initDict == '':
        initDict = init()
    
    energy = np.loadtxt(initDict['ENERGY_DIR'])[:,1]
    #load data, binned by pixel
    XUVlist = importDir(initDict)
    binParams = ConfigParser()
    try:
        binParams.read(initDict['Bin_params'])
    except FileNotFoundError():
        binParams={'Ebins':{'binW':0.2}}
        
    #align energy axis, normalize the flux
    if initDict['CorrectE']:
        Ealigned = energyCorrect(XUVlist)
    else:
        Ealigned = XUVlist
    #mask bad data
    dataMask = trimDataset(np.asarray(Ealigned[0::2]))
    #normalize Flux
    if initDict['normMethod'] == 'Odd':
        normSpectra = normOddHarms(np.asarray(Ealigned), '', energy)
    elif initDict['normMethod'] == 'Half': 
        normSpectra = normHalfHarms(np.asarray(Ealigned), '', energy)
    else:
        normSpectra=normHarms(np.asarray(Ealigned))
    #bin over energy
    ebin,ebinstd,Eaxis = energyBin(normSpectra, energy, binParams)
    
    #separate out datasets, plot
    sample = ebin[:,1::2]
    sampleStd = ebinstd[:,1::2]
    reference = ebin[:,0::2]
    referenceStd = ebinstd[:,0::2]
    
    plotStep = int(sample.shape[0]/40)
    fig1,ax1 = plt.subplots()
    ax1.plot(np.transpose(sample[:,::plotStep]),'r')
    ax1.plot(np.transpose(reference[:,::plotStep]),'b')
    
    #calulate absorbtion
    MCD = sample-reference
    #average data
    meanMCD = np.nanmean(MCD,axis=1,where=dataMask)
    #frequency filter the data
    freqFilt = freqFilterChoose(meanMCD,Eaxis)
    
    return meanMCD #XUVdata(freqFilt, meanStd, Eaxis)

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

def onclick(event):
    global offset
    offset = 1
    return 

def getCropPos(Xdata,title='Select harmonic peaks'):
    """"Function to use to extract points for GVDcorrection. Only call on it's own if you want the points used for GVDcorrection.
    Returns a tuple of GVD points in energy and time"""
    fig,axs = plt.subplots()
    axs.imshow(Xdata)
    plt.title(title)
    klicker = clicker(axs,['pos'],markers=["x"],linestyle="-",colors=["red"])
    plt.show()
        
    cid = fig.canvas.mpl_connect('close_event', onclick)
    while not 'offset' in globals():
        plt.pause(2)
    
    if 'offset' in globals():
        global offset
        del offset

    fig.canvas.mpl_disconnect(cid)
    
    pos = klicker.get_positions()['pos']
    idx = np.array(pos[:,1],dtype='int')
    
    return idx