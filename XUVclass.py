# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 20:46:34 2021

@author: Bozark
"""
#Things To do for OTA class: store wavelength, time, intensity, statistics for
#given wavlengths? What about noise filtering? Anything that can be done with 
#a single dataset, so no dataset or std dev, but GVD correction, t0 alignment,
#bad time point removal, kinetic and spectral traces
import numpy as np
import math, re
from numpy.lib.scimath import sqrt
import KineticFitFunctions as kff
from copy import deepcopy
import read_SPE as spe
import os
from SPErebin import pixelBin,energyBin
from CrossECorrection import energyCorrect
from FluxNormalization import normHalfHarms,normHalfOddHarms,normOddHarms,normBackSub
from configparser import ConfigParser
from AbsCalcs import XUVabs,XUVtransAbs
from FreqFilter import freqFilterChoose
from statMethods import fitPCA, airPCA

class XUVimg():
    def __init__(self, Intensity, enrgBin, std, Eaxis,name=''):
        self.Intensity = Intensity
        self.StdDev = std
        self.pxlBin = pixelBin(Intensity)
        self.enrgBin = enrgBin
        self.Eaxis = Eaxis
        self.name = name

class XASdata():
    def __init__(self,XUVlist,dataBck=None,refBck=None,name=None,
                 options=None, template=None):
        self.raw = np.asarray(XUVlist)
        self.name = name
        if template is not None:
            options = template.options
            options['Eaxis']=template.energy
        self.options = options
        self.pipe(XUVlist,dataBck,refBck,options)
        
    def pipe(self, XUVlist,dataBck=None,refBck=None, options=None):
        
        spectrum=None
        
        if options is not None:
            if options['AlignE']:
                self.alignE(XUVlist)
                spectrum=self.aligned
            
            if options['NormFlux']:
                self.normFlux(options['normMethod'],spectrum=spectrum)
                spectrum = self.normed
        
        self.energyBin(spectrum=spectrum, binW = options['binW'], 
                       dataBck=dataBck, refBck=refBck)
        self.separateData()
        self.calcAbs()
        self.averageXAS()
        self.scanStats()
        
    def energyBin(self, binW=0.2,spectrum=None,
                  dataBck=None,refBck=None):
        
        if spectrum is not None:
            self.binned, self.binstd, self.energy = energyBin(spectrum,
                                                          binW)
        else:
            self.binned, self.binstd, self.energy = energyBin(self.raw,
                                                          binW)
            
        if dataBck is not None and refBck is not None:
            bckbin = energyBin(np.asarray([dataBck,refBck]),
                               binW, self.energy)
            self.dataBck = bckbin[0][:,0]
            self.refBck = bckbin[0][:,1]
        else:
            self.dataBck = None
            self.refBck = None
            
    def separateData(self,method='Alt',data=None,ref=None):
        if data is not None and ref is not None:
            self.data = data
            self.ref = ref
        else:
            if method == 'Alt':
                self.data = self.binned[:,1::2]
                self.ref = self.binned[:,0::2]
            else:
                self.data = np.concatenate((self.binned[:,0,np.newaxis],
                                           self.binned[:,2::4],
                                           self.binned[:,3::4]),
                                           axis=1)
                self.ref = np.concatenate((self.binned[:,1,np.newaxis],
                                          self.binned[:,4::4],
                                          self.binned[:,5::4]),
                                          axis=1)
                
        if self.dataBck is not None and self.refBck is not None:
            if self.options['NormFlux']:
                self.data = normBackSub(self.data, self.dataBck, self.energy)
                self.ref = normBackSub(self.ref, self.refBck, self.energy)
            elif self.options['normBckTF']:
                self.data = normBackSub(self.data, self.dataBck, self.energy)
                self.ref = normBackSub(self.ref, self.refBck, self.energy)
            else:
                self.data -= self.dataBck[:,None]
                self.ref -= self.refBck[:,None]
        
    def scanStats(self):
        self.intCnts = np.sum(self.binned,axis=0)
        self.dataBase = np.nanmean(XUVabs(self.data[:,0::2],self.data[:,1::2]),axis=1)
        self.refBase = np.nanmean(XUVabs(self.ref[:,0::2],self.ref[:,1::2]),axis=1)
        self.dataStd = np.nanstd(self.data,axis=1)
        self.refStd = np.nanstd(self.ref,axis=1)
        
    def alignE(self,XUVlist):
        self.aligned = np.asarray(energyCorrect(XUVlist))
    
    def normFlux(self,method,spectrum=None):
        if spectrum is not None:
            self.normed = method(spectrum)
        else:
            self.normed = method(self.raw)
    
    def calcAbs(self,data=None,ref=None):
        if data is not None:
            self.abs = XUVabs(data,ref)
        else:
            self.abs = XUVabs(self.data,self.ref)
    
    def freqFilt(self, spectrum=None):
        if spectrum is not None:
            self.freq = freqFilterChoose(spectrum,self.energy)
        else:
            self.freq = freqFilterChoose(self.aveXAS,self.energy)
            
    def averageXAS(self):
        
        self.aveXAS = np.nanmean(self.abs,axis=1)
        self.aveXASstd = np.nanstd(self.abs,axis=1)
        self.aveHarmData = np.nanmean(self.data,axis=1)
        self.aveHarmRef = np.nanmean(self.ref,axis=1)
        
    def harmonicAlignment(self):
        pass

class XTAdata():
    def __init__(self, XUVlist, time, options,
                 dataBck=None, refBck=None,
                 name = None, template = None):
        
        if type(XUVlist) == tuple:
            self.raw = np.asarray(XUVlist[0])
            self.xpos = XUVlist[1]
            self.ypos = XUVlist[2]
            self.delaypos = XUVlist[3]
        else:
            self.raw = np.asarray(XUVlist)
            self.xpos = None
            self.ypos = None
            self.delaypos = None
            
        self.name = name
        self.time = np.array(time)
        if template is not None:
            options = template.options
            options['Eaxis']=template.energy
        self.options = options
        self.Kinetic = {}
        self.KineticStd = {}
        self.T_slice = {}
        self.T_sliceStd = {}
        self.FitResults = {}
        self.KineticFit = {}
        self.pipe(XUVlist,dataBck,refBck,options)
        
    def pipe(self, XUVlist, dataBck,refBck,options=None):
        spectrum=None
        
        if options is not None:
            if options['AlignE']:
                self.alignE(XUVlist)
                spectrum = self.aligned
                
            if options['NormFlux']:
                self.normFlux(options['normMethod'],spectrum=spectrum)
                spectrum = self.normed
        
            self.energyBin(options, spectrum=spectrum,
                           dataBck=dataBck,
                           refBck=refBck)
            self.separateData(method=options['dataOrg'])
        
            self.calcTrans(pca = options['pcaTF'],
                           pcaMethod=options['pcaMethod'],
                           pcaArgs = options['pcaArgs'])
            
        else:
            self.energyBin(options, dataBck=dataBck, refBck=refBck)
            self.separateData()
            self.calcTrans()
            
        if len(self.time) >1:
            self.bckgSub()
        self.averageXTA()
        self.scanStats()
        
    def energyBin(self, options,spectrum=None,
                  dataBck=None,refBck=None):
        
        binW = options['Ebins']['binW']
        Eaxis = options['Eaxis']
        if spectrum is None:
            self.binned, self.binstd, self.energy = energyBin(self.raw,
                                                          binW, Eaxis=Eaxis)
        else:
            self.binned, self.binstd, self.energy = energyBin(spectrum,
                                                          binW, Eaxis=Eaxis)
            
        if dataBck is not None and refBck is not None:
            bckbin = energyBin(np.asarray([dataBck,refBck]),
                               binW,self.energy)
            self.dataBck = bckbin[0][:,0]
            self.refBck = bckbin[0][:,1]
        else:
            self.dataBck = None
            self.refBck = None
            
    def alignE(self,XUVlist):
        self.aligned = np.asarray(energyCorrect(XUVlist))
            
    def normFlux(self,method,spectrum=None):
        if spectrum is not None:
            self.normed = method(spectrum)
        else:
            self.normed = method(self.raw)
        
    def separateData(self,method='Alt',data=None,ref=None):
        if data is None or ref is None:
            if method == 'Alt':
                self.data = self.binned[:,0::2]
                self.ref = self.binned[:,1::2]
            else:
                self.data = np.concatenate((self.binned[:,0::4],
                                           self.binned[:,3::4]),
                                           axis=1)
                
                self.ref = np.concatenate((self.binned[:,1::4],
                                          self.binned[:,2::4]),
                                          axis=1) 
        else:
            self.data = data
            self.ref = ref
            
        if self.dataBck is not None and self.refBck is not None:
            self.data -= self.dataBck[:,None]
            self.ref -= self.refBck[:,None]
        
    def calcTrans(self,data=None,ref=None,
                  pca=False, pcaMethod=None, pcaArgs=None):
        if data is not None:
            trans = XUVtransAbs(data, ref)
        elif pca:
            trans = pcaMethod(self.data, self.ref,
                              self.energy, time=self.time, 
                              **pcaArgs)
        else:
            trans = XUVtransAbs(self.data, self.ref)
        
        self.trans3D = np.reshape(trans, (len(self.energy),
                                          int(self.data.shape[1]/len(self.time)),
                                          len(self.time)))
        
    def averageXTA(self,outliers=False):
        self.trans2D = np.nanmean(self.trans3D,axis=1)
        self.trans2Dstd = np.nanstd(self.trans3D,axis=1)/np.sqrt(self.trans3D.shape[1])
        
        if outliers:
            tempMean = np.nanmean(self.trans3D,axis=0)
            mask = np.ma.masked_where(np.logical_and(
                self.trans3D < (self.trans3D-self.trans2Dstd[:,None,:]),
                self.trans3D > (self.trans3D+self.trans2Dstd[:,None,:])),
                self.trans3D)
            self.trans2D = np.ma.mean(mask,axis=1)
            self.trans2Dstd = np.ma.std(mask,axis=1)
            
        self.aveHarmData = np.nanmean(self.data,axis=1)
        self.aveHarmRef = np.nanmean(self.ref,axis=1)
        
    def scanStats(self):
        self.intCnts = np.sum(self.binned,axis=0)
        self.harmStd = np.nanstd(self.data,axis=1)
        self.harm2Ddiff = self.data - self.aveHarmData[:,None,None]
        self.dataBase = np.nanmean(XUVabs(self.data[:,0::2],self.data[:,1::2]),axis=1)
        self.refBase = np.nanmean(XUVabs(self.ref[:,0::2],self.ref[:,1::2]),axis=1)
        self.dataStd = np.nanstd(self.data,axis=1)
        self.refStd = np.nanstd(self.ref,axis=1)
        
    def bckgSub(self):
        
        temptime = np.where(self.time<-0.1,True,False)
        TmpArr = self.trans3D[:,:,temptime]
        TmpMean = np.nanmean(TmpArr,axis=(1,2))
        
        self.background = TmpMean
        self.trans3D = self.trans3D - TmpMean[:,None,None]
        
    def KineticTrace(self,energy,binW):
        idx = []
        label  = str(energy)
        idx = [i for i,x in enumerate(self.energy) if math.isclose(x, energy,rel_tol=(binW/energy))]
        self.Kinetic[label]=np.sum(self.trans2D[idx,:].copy(),axis=0)/len(idx)
        self.KineticStd[label]=np.sqrt(np.sum(np.square(self.trans2Dstd[idx,:].copy()),axis=0))/sqrt(len(idx))
    
    def SpectralTrace(self,Tpoint,binW):
        idx = []
        label = str(Tpoint)
        idx = [i for i,x in enumerate(self.time) if math.isclose(x, Tpoint,rel_tol=(np.abs(binW/Tpoint)))]
        self.T_slice[label] = np.sum(self.trans2D[:,idx].copy(),axis=1)/len(idx)
        self.T_sliceStd[label]=np.sqrt(np.sum(np.square(self.trans2Dstd[:,idx].copy()),axis=1))/sqrt(len(idx))

class XMCDdata():
    def __init__(self, XUVlist, dataBck=None, refBck=None,
                 name = None, options = None, template = None):
        self.raw = np.asarray(XUVlist)
        self.name = name
        self.options = options
        self.Kinetic = {}
        self.KineticStd = {}
        self.T_slice = {}
        self.T_sliceStd = {}
        self.FitResults = {}
        self.KineticFit = {}
        self.pipe(XUVlist,dataBck,refBck,options)
        
    def pipe(self, XUVlist, dataBck,refBck,options=None):
        spectrum=None
        
        if options is not None:
            if options['AlignE']:
                self.alignE(XUVlist)
                spectrum = self.aligned
                
            if options['NormFlux']:
                self.normFlux(options['normMethod'],spectrum=spectrum)
                spectrum = self.normed
        
            self.energyBin(spectrum=spectrum,dataBck=dataBck,refBck=refBck)
            self.separateData(method=options['dataOrg'])
        
            self.calcTrans(pca = options['pcaTF'],
                           pcaMethod=options['pcaMethod'],
                           pcaArgs = options['pcaArgs'])
            
        else:
            self.energyBin(dataBck=dataBck,refBck=refBck)
            self.separateData()
            self.calcTrans()
            
        self.averageMCD()
        self.scanStats()
        
    def energyBin(self, binW=0.2, spectrum=None,
                  dataBck=None, refBck=None):
        if spectrum is None:
            self.binned, self.binstd, self.energy = energyBin(self.raw,
                                                          binW)
        else:
            self.binned, self.binstd, self.energy = energyBin(spectrum,
                                                          binW)
            
        if dataBck is not None and refBck is not None:
            bckbin = energyBin(np.asarray([dataBck,refBck]),
                               binW,self.energy)
            self.dataBck = bckbin[0][:,0]
            self.refBck = bckbin[0][:,1]
        else:
            self.dataBck = None
            self.refBck = None
            
    def alignE(self,XUVlist):
        self.aligned = np.asarray(energyCorrect(XUVlist))
            
    def normFlux(self,method,spectrum=None):
        if spectrum is not None:
            self.normed = method(spectrum)
        else:
            self.normed = method(self.raw)
        
    def separateData(self,method='Alt', data=None, ref=None):
        if data is None or ref is None:
            if method == 'Alt':
                self.data = self.binned[:,0::2]
                self.ref = self.binned[:,1::2]
            else:
                self.data = np.concatenate((self.binned[:,0::4],
                                           self.binned[:,3::4]),
                                           axis=1)
                
                self.ref = np.concatenate((self.binned[:,1::4],
                                          self.binned[:,2::4]),
                                          axis=1) 
        else:
            self.data = data
            self.ref = ref
            
        if self.dataBck is not None and self.refBck is not None:
            self.data -= self.dataBck[:,None]
            self.ref -= self.refBck[:,None]
        
    def calcTrans(self, data=None, ref=None,
                  pca=False, pcaMethod=None, pcaArgs=None):
        if data is not None:
            trans = XUVtransAbs(data, ref)
        elif pca:
            trans = pcaMethod(self.data, self.ref,
                              self.energy, **pcaArgs)
        else:
            trans = XUVtransAbs(self.data, self.ref)
        
        self.trans2D = np.reshape(trans, (len(self.energy),
                                          int(self.data.shape[1])))
        
    def averageMCD(self,outliers=False):
        self.MCD = np.nanmean(self.trans2D,axis=1)
        self.MCDstd = np.nanstd(self.trans2D,axis=1)/np.sqrt(self.trans2D.shape[1])
            
        self.aveHarmData = np.nanmean(self.data,axis=1)
        self.aveHarmRef = np.nanmean(self.ref,axis=1)
        
    def scanStats(self):
        self.intCnts = np.sum(self.binned,axis=0)
        self.harmStd = np.nanstd(self.data,axis=1)
        self.harm2Ddiff = self.data - self.aveHarmData[:,None,None]