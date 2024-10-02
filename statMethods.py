# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:48:32 2024

@author: Samth
"""

import numpy as np
from AbsCalcs import XUVabs
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import PCA
import numpy.ma as ma
from lmfit import minimize, Parameters,fit_report
from mpl_point_clicker import clicker 
from FreqFilter import onclick,getMask

def PearsonsCov(data):
    #https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-22-35135&id=460498
    ref1 = data[0::2].T
    ref2 = data[1::2].T
    
    pairAbs = -np.log(np.abs(ref1/ref2))
    # pairAbs = np.where(np.isnan(rawAbs),rawAbs,0)
    
    meanAbs = np.nanmean(pairAbs,axis=1)
    pairStd = np.nanstd(pairAbs,axis=1)
    
    covar = np.ma.cov(np.ma.masked_invalid(pairAbs))
    covStd = pairStd*pairStd.T
    
    Pcorr= covar/covStd
    
    return Pcorr

def svdCorr(data,comp = 10,threshold=0.95):

    leftU, singVals,rightW = svd(data)
    pca = PCA(comp,svd_solver='auto').fit(np.transpose(data))
    
    #number of components needed to account for 95% of variance
    n_comp=0
    i=0
    while i<threshold:
       i += pca.explained_variance_ratio_[n_comp]
       n_comp+=1
       
    return n_comp,pca
    
def fitPCA(data,energy,components = 20,transAbs=None,
           threshold=0.95,time=None,pca=None,quiet=False):
    
    pumpON = np.where(data[:,0::2] == 0,  1, data[:,0::2])
    pumpOFF = np.where(data[:,1::2] == 0,  1, data[:,1::2])
    
    pumpOffAbs = -np.log10(np.abs(pumpOFF[:,0::2]/pumpOFF[:,1::2]))
    
    if transAbs==None:
        transAbs = -np.log10(np.abs(pumpON/pumpOFF))
    
    if pca is None:
        pca = PCA(components).fit(np.transpose(pumpOffAbs))

    OD_PC = pca.components_.T    
    
    ROI = getMask(energy,pumpON[:,0],
                  title=""""Select energies to exclude for PCA fit,
                      must be in pairs to denote a range""")
    
    lowE = ROI[0::2]
    highE = ROI[1::2]
    
    for x,y in zip(lowE,highE):
        tempMask = np.where(np.logical_or((energy< x), (energy>y)),
                            1,0)
        # maskData[tempMask] = data[tempMask]
        W_k = np.diag(tempMask)    

    C = np.linalg.inv(OD_PC.T.dot(W_k.dot(OD_PC))).dot(
        OD_PC.T).dot(W_k).dot(transAbs)
    
    # result = minimize(PCAresid,params,args = (maskData,n_comps,pca,ROI))
    model = OD_PC.dot(C)
    
    if quiet is False:
        fig1,ax = plt.subplots(5,1,sharex=True)
        for idx,axis in enumerate(ax):
            axis.plot(energy,OD_PC[:,idx])
            
            for x,y in zip(lowE,highE):
                axis.fill_betweenx([np.min(OD_PC[:,idx]),np.max(OD_PC[:,idx])],
                                   x,y,color='r',alpha=0.2)
       
        if time is not None:
            tempTrans = np.reshape(transAbs, (len(energy),
                                           int(pumpON.shape[1]/len(time)),
                                           len(time)))
            tempModel = np.reshape(model, (len(energy),
                                           int(pumpON.shape[1]/len(time)),
                                           len(time)))
            trans2D = np.nanmean(tempTrans,axis=1)
            model2D = np.nanmean(tempModel,axis=1)
            fig2,bx = plt.subplots(5,1,sharex=True)
            for idx,axis in enumerate(bx):
                axis.plot(energy,trans2D[:,idx],label=str(time[idx]))
                axis.plot(energy,model2D[:,idx])
                
                for x,y in zip(lowE,highE):
                    axis.fill_betweenx([np.min(trans2D[:,idx]),np.max(trans2D[:,idx])],
                                      x,y,color='r',alpha=0.2)
                axis.legend()
        
    return transAbs-model

def airPCA(data,energy,components = 10,maxIter=80,
           threshold=0.95,
           pca=None,quiet=False):
    
    pumpON = data[:,0::2]
    pumpOFF = data[:,1::2]
    
    pumpOffAbs = -np.log10(pumpOFF[:,0::2]/pumpOFF[:,1::2])
    transAbs = -np.log10(pumpON/pumpOFF)
    
    k_vect = np.arange(maxIter)
    vector = [1+0.2]**k_vect
    
    if pca is None:
        pca = PCA(components).fit(np.transpose(pumpOffAbs))

    OD_PC = pca.components_.T    
    
    pcaWeights = np.ones(len(pumpON))
    
    W_k = np.diag(pcaWeights)    

    C = np.linalg.inv(OD_PC.T.dot(W_k.dot(OD_PC))).dot(
        OD_PC.T).dot(W_k).dot(transAbs)
    
    # result = minimize(PCAresid,params,args = (maskData,n_comps,pca,ROI))
    model = OD_PC.dot(C)
    
    if not quiet:
        plt.imshow((transAbs-model).T)
    
    return transAbs-model

def NPRpca(pcaSub,data):
    NPR = (np.std(data,1)**2)/(np.std(pcaSub,1)**2)
    
    return NPR

def PCAresid(params,data,n_comp,pca):
    
    OD_PC = params['PCA'].value
    OD_PC_T = params['delOD'].value
    W_k = params['fitPCA'].value
    # delPC = 
    
    C = np.linalg.inv(OD_PC.T.dot(W_k.dot(OD_PC))).dot( 
        OD_PC.T).dot(W_k).dot(data)
    
    model = pca.components_(n_comp).dot(C)
    
    return (data-model)**2
    
    