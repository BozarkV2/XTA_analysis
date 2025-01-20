# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:55:05 2024

@author: Samth
"""

import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters,fit_report
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.stats import lognorm
from scipy.integrate import quad
from numpy.lib.scimath import sqrt,log
from numpy import correlate
from maskDatasets import getMask

def energyCorrect(data, centerPxl=512):
    
    params = Parameters()
    params.add('m', value= 1,min=0.9,max=1.1, vary=True)
    
    dataArr = np.asarray(data)
    meanCnts = np.mean(dataArr,axis=1)
    refIdx = getMask(meanCnts, title = 'Select reference scan')[0]
    
    refPxl = np.linspace(1,1024,num=1024)
    eCorrected=[]

    for idx,dataset in enumerate(data):
        if idx is not refIdx:
            shift = np.argmax(correlate(dataset, data[refIdx],mode='same'))-centerPxl
            
            normRef = data[refIdx]/np.max(data[refIdx])
            normData = dataset/np.max(dataset)
            
            results = minimize(crossResidual,params,
                               args=(normRef,normData,shift,centerPxl,refPxl),
                               nan_policy='omit')
            
            slope = results.params['m'].value
            intercept = centerPxl*(1-slope)-shift
            dataInterp = interp1d(intercept+slope*refPxl,dataset,fill_value='extrapolate')
            tempCorr = dataInterp(refPxl)
            eCorrected.append(tempCorr)
        else:
            eCorrected.append(data[idx])
    
    return eCorrected

def crossResidual(params,ref,data,shift,centerPxl,refPxl):
    
    parvals = params.valuesdict()
    slope = parvals['m']
    intercept = centerPxl*(1-slope)-shift
    
    dataInterp = interp1d(slope*refPxl+intercept,data,fill_value='extrapolate')
    ecorrData = dataInterp(refPxl)
    model = np.max(correlate(ref,ecorrData,mode='same'))
    
    return np.abs(ref-ecorrData)/model

def testenergyCorrect(data):
    
    params = Parameters()
    params.add('m', value= 1,min=0.8,max=1.2, vary=True)
    params.add('shift', value= 0,min=-2,max=2, vary=True)
    
    dataArr = np.asarray(data)
    meanCnts = np.mean(dataArr,axis=1)
    refIdx = getMask(meanCnts, title = 'Select reference scan')[0]
    
    refPxl = np.linspace(1,1024,num=1024)
    eCorrected=[]

    for idx,dataset in enumerate(data):
        if idx is not refIdx:
            
            normRef = data[refIdx]/np.sum(data[refIdx])
            normData = dataset/np.sum(dataset)
            
            results = minimize(testcrossResidual,params,
                               args=(normRef,normData,refPxl),
                               nan_policy='omit')
            
            slope = results.params['m'].value
            intercept = results.params['shift'].value
            dataInterp = interp1d(intercept+slope*refPxl,dataset,fill_value='extrapolate')
            tempCorr = dataInterp(refPxl)
            eCorrected.append(tempCorr)
        else:
            eCorrected.append(data[idx])
    
    return eCorrected

def testcrossResidual(params,ref,data,refPxl):
    
    parvals = params.valuesdict()
    slope = parvals['m']
    shift = parvals['shift']
    
    dataInterp = interp1d(slope*refPxl+shift,data,fill_value='extrapolate')
    ecorrData = dataInterp(refPxl)
    model = np.max(correlate(ref,ecorrData,mode='same'))
    
    return np.abs(ref-ecorrData)/model