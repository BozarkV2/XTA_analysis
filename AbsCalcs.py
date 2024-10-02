# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:20:32 2024

@author: Samth
"""
import numpy as np

def XUVabs(data, ref):
    
    if ref.shape != data.shape:
        ref = np.ones(data.shape)
        
    Xabs = -np.log10(np.abs(data/ref))
    
    return Xabs

def XUVstd(data,Std,ref,refStd):
    
    Xabs = -np.log10(np.abs(data/ref))
    Xstd = np.abs(Xabs)*np.sqrt((np.divide(Std,data)**2+
                                 np.divide(refStd,ref)**2))
    
    return Xstd

def XUVtransAbs(data, ref):
    
    transient = -np.log10(np.abs(data/ref))
    # bckg = np.mean([transient[x] for x,t in enumerate(time) if t<-5], axis=1)
    # bckg = np.zeros(transient.shape)
    
    return transient

def XUVtransstd(data,Std,ref,refStd):
    
    transient = -np.log10(np.abs(data/ref))
    bckg = np.zeros(data.shape)
    Xstd = np.abs(transient)*np.sqrt((np.divide(Std,data)**2+
                                      np.divide(refStd,ref)**2))
    
    return Xstd
