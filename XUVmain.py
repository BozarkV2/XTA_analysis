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
import re, json
from mpl_point_clicker import clicker 
import read_SPE as spe
from SPErebin import energyBin,pixelBin, pixelRotBin
from FluxNormalization import normHarms,normOddHarms,normHalfHarms,normHalfOddHarms
from statMethods import fitPCA,airPCA

def init(DIRECTORY):
    
    workspace_dict={
    'DIRECTORY':DIRECTORY, #Where is your data stored? Must be file path
    'all_subdirs':True, #Whether you want to import all directories in a folder
    'CropTF':False, #do you want to crop the y-axis of the images
    'rotTF':False, #rotate the image to correct for harmonic angle
    'rotD':0,
    'Eaxis':None, #give an energy array if you already have an energy axis on hand
    'Ebins':{'binW':0.2, #the energy bin width in eVs
             'fundamental':1030},
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
    
    if type(initDict) is dict:
        directory = initDict['DIRECTORY']
        cropTF = initDict['CropTF']
        rotation=initDict['rotTF']
    else:
        directory = initDict
        cropTF = False
        rotation = False
    
    rawImgs = []
    
    if subdirs is True:
        for path in os.scandir(directory):
            if path.is_dir():
                rawImgs.append(importDir(path))
    
    if cropTF:
        cropList = []
    
    fileList=[]
    
    for file in os.scandir(directory):
        if file.name.endswith('.spe') or file.name.endswith('.dat'):
            fileList.append(file.path)
              # if file.path.endswith('.spe')) #Removed for compatibility with NeXUS dat format
    fileList.sort()
    idx=0
    
    for file in fileList:
        print(file)
        if file.endswith('.spe'):
            #spe contains multiple images per file
            for data in spe.loadGen(file):
                if cropTF and idx<2:
                    cropROI = get2DPos(data, "Select Image Bounds")
                    cropList.append(cropROI)
            
                if cropTF:
            #choose croped image based on sample or ref image
                    rawImgs.append(pixelBin(data,
                                            cropList[np.mod(idx,2)])) 
                elif rotation:
                    rawImgs.append(pixelRotBin(data,-4))  
                else:
                    rawImgs.append(pixelBin(data))  
                idx+=1
        
        elif file.endswith('.dat'):
            data = np.loadtxt(file, skiprows=6, max_rows=1300)
            if cropTF and idx<2:
                cropROI = get2DPos(data, "Select Image Bounds")
                cropList.append(cropROI)
            
            if cropTF:
        #choose croped image based on sample or ref image
                rawImgs.append(pixelBin(data,
                                        cropList[np.mod(idx,2)])) 
            elif rotation:
                rawImgs.append(pixelRotBin(data.reshape(900,2048),initDict['rotD']))  
                
            else:
               rawImgs.append(pixelBin(data))  
            
            idx+=1
        
    return rawImgs

def importNexus(initDict, subdirs = False):
    
    if type(initDict) is dict:
        directory = initDict['DIRECTORY']
        cropTF = initDict['CropTF']
        rotation=initDict['rotTF']
    else:
        directory = initDict
        cropTF = False
        rotation = False
    
    rawImgs = []
    
    if subdirs is True:
        for path in os.scandir(directory):
            if path.is_dir():
                rawImgs.append(importNexus(path))
    
    if cropTF:
        cropList = []
    
    fileList=[]
    
    for file in os.scandir(directory):
        if file.name.endswith('.dat'):
            fileList.append(file.path)
              
    fileList.sort()
    header = []
    xpos = []
    ypos = []
    delaypos = []
    bckgDict = {}
    
    for file in fileList:
        print(file)
        
        with open(file, 'r') as data:
            img=[]
            idx=1
            for line in data:
                if idx == 2:
                    hdrDict = json.loads(line)
                    header.append(hdrDict)
                    xpos.append(hdrDict['Sample_Stage_xPosition'])
                    ypos.append(hdrDict['Sample_Stage_yPosition'])
                    delaypos.append(hdrDict['Delay_Stage_Position'])
                    bckg = hdrDict['Corresponding_Background']
                        
                if idx >6 and re.match('\n', line):
                    
                    tempArr = np.asarray(img, np.float64)
                    if bckg not in bckgDict:
                        bckgDict[bckg] = pixelBin(np.loadtxt
                                                  (os.path.split(os.path.split(file)[0])[0]+
                                                   '\Background\\'+bckg, 
                                                   skiprows=6,
                                                   max_rows=tempArr.shape[0]))
                        
                    rawImgs.append(pixelBin(tempArr) - bckgDict[bckg])
                    break
                
                if idx >6:    
                    img.append(line.split('\t'))
                
                idx+=1
        
    return rawImgs, xpos, ypos, delaypos

def get2DPos(Xdata,title='Select harmonic peaks'):
    fig,axs = plt.subplots()
    axs.imshow(Xdata)
    plt.title(title)
    klicker = clicker(axs,['pos'],markers=["x"],linestyle="-",colors=["red"])
    plt.show(block=True)
    
    pos = klicker.get_positions()['pos']
    idx = np.array(pos[:,1],dtype='int')
    
    return idx

def get1DPos(Xdata,title='Select harmonic peaks', xaxis=None):
    fig,axs = plt.subplots()
    if xaxis is None:
        axs.plot(Xdata)
    else:
        axs.plot(xaxis, Xdata)
        
    plt.title(title)
    klicker = clicker(axs,['pos'],markers=["x"],linestyle="-",colors=["red"])
    plt.show(block=True)
    
    pos = klicker.get_positions()['pos']
    idx = np.array(pos[:,0])
    
    return idx