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
    
def fitPCA(data, ref, energy, components = 20,transAbs=None,
           threshold=0.95, maxIter=None,
           time=None, pca=None, quiet=False):
    
    if time is not None:
        ref3D = np.reshape(ref, (len(energy),
                                  int(ref.shape[1]/len(time)),
                                  len(time)))
        tempOffAbs = -np.log10(np.abs(ref3D[:,0::2,:]/ref3D[:,1::2,:]))
        pumpOffAbs = np.reshape(tempOffAbs, (len(energy),
                                       int(ref.shape[1]/2)))
    else:
        pumpOffAbs = -np.log10(np.abs(ref[:,0::2]/ref[:,1::2]))
    
    pcaOffAbs = np.where(np.logical_or(np.isnan(pumpOffAbs), 
                         np.isinf(pumpOffAbs)),
                         0, pumpOffAbs)
    
    if transAbs==None:
        tempAbs = -np.log10(np.abs(data/ref))
        transAbs = np.where(np.logical_or(np.isnan(tempAbs), 
                             np.isinf(tempAbs)),
                             0, tempAbs)
    
    if pca is None:
        pca = PCA(components).fit(np.transpose(pcaOffAbs))

    OD_PC = pca.components_.T    
    
    ROI = getMask(energy,data[:,0],
                  title=""""Select energies to exclude for PCA fit,
                      must be in pairs to denote a range""")
    
    lowE = ROI[0::2]
    highE = ROI[1::2]
    
    for x,y in zip(lowE,highE):
        tempMask = np.where(np.logical_or((energy< x), (energy>y)),
                            1,0)
        W_k = np.diag(tempMask)    

    C = np.linalg.inv(OD_PC.T.dot(W_k.dot(OD_PC))).dot(
        OD_PC.T).dot(W_k).dot(transAbs)
    
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
                                           int(data.shape[1]/len(time)),
                                           len(time)))
            tempModel = np.reshape(model, (len(energy),
                                           int(data.shape[1]/len(time)),
                                           len(time)))
            trans2D = np.nanmean(tempTrans,axis=1)
            model2D = np.nanmean(tempModel,axis=1)
            fig2,bx = plt.subplots(5,1,sharex=True)
            for idx,axis in enumerate(bx):
                axis.plot(energy,trans2D[:,-idx],label=str(time[-idx]))
                axis.plot(energy,model2D[:,-idx])
                
                for x,y in zip(lowE,highE):
                    axis.fill_betweenx([np.min(trans2D[:,-idx]),
                                        np.max(trans2D[:,-idx])],
                                      x,y,color='r',alpha=0.2)
                axis.legend()
        
    return transAbs-model

def airPCA(data, ref, energy, components = 20, maxIter=40, time=None,
           threshold=0.95, transAbs=None,
           pca=None, quiet=True):
    
    pumpOffAbs = -np.log10(np.abs(ref[:,0::2]/ref[:,1::2]))
    
    pcaOffAbs = np.where(np.logical_or(np.isnan(pumpOffAbs), 
                         np.isinf(pumpOffAbs)),
                         0, pumpOffAbs)
    
    if transAbs is None:
        tempAbs = -np.log10(np.abs(data/ref))
        transAbs = np.where(np.logical_or(np.isnan(tempAbs), 
                             np.isinf(tempAbs)),
                             0, tempAbs)
    
    if pca is None:
        pca = PCA(components).fit(np.transpose(pcaOffAbs))

    OD_PC = pca.components_.T    
    
    if time is not None:
        Nt = np.shape(time)[0]
        n, scans = np.shape(transAbs)
        Ns = int(scans/Nt)
        
        tempTrans = np.reshape(transAbs, 
                               (n,Ns,Nt))
        trans2D = np.nanmean(tempTrans,axis=1)
    else:
        Nt=1
        n, scans = np.shape(transAbs)
        Ns = int(scans)
        
        tempTrans = np.reshape(transAbs, 
                               (n,Ns,1))
        trans2D = np.nanmean(tempTrans,axis=1)
    
    k_max = maxIter                             # maximum number of iterations
    c = 0.2                                     # constant c
    k_vect = np.arange(k_max)                   # vector of increasing steps k
    vector = (1+c)**k_vect                  # constant alpha(k)
    k_opt = 45                                  # optimum k where the optimum result is found
    stop_at_opt = False                         # if True, the iteration stops when the optimum k is reached
    
    w = []                                      # array where the effective number of pixels is stored
    d_norm = []                                 # array where the square norm of d is stored
    NPR_test = []                               # array where the NPR (noise power reduction) of the test sample is stored
    NPR_avg = []                                # array where the NPR averaged over the pump-probe delays t is stored
    
    w_k = np.ones(n)                            # initial weights
    
    for k, alpha in enumerate(vector):
        print("k: " + str(k))
        w.append(np.sum(w_k))
        
        # find OD_ref step k (OD_ref_k)
        W_k = np.diag(w_k)
        
        C = np.linalg.inv(OD_PC.T.dot(W_k.dot(OD_PC))).dot(
            OD_PC.T).dot(W_k).dot(transAbs)
        
        model = OD_PC.dot(C)
        OD_ref_k = transAbs - model
        
        # calculate d_norm step k, and append to list d_norm
        OD_mean_k = np.mean(OD_ref_k,1)
        d_abs_k = np.abs(OD_mean_k)
        d_norm_k = np.linalg.norm(d_abs_k)    
        d_norm.append(d_norm_k)
        
        # calculate NPR values for test sample (first delay), and append to list NPR_test
        # average NPR for all the other delays, and append to list NPR_avg
        OD_ref_k_t = np.reshape(OD_ref_k, [n, Ns, Nt])
        NPR_k = NPRpca(OD_ref_k_t,tempTrans)
        NPR_avg_k = np.sum((1-w_k)[:,None]*NPR_k,0)/np.sum(1-w_k)
        NPR_test.append(10*np.log10(NPR_avg_k[0]))
        NPR_avg.append(10*np.log10(np.mean(NPR_avg_k[1:])))
        
        # stop if the optimum is reached if the flag stop_at_optimum is selected
        if k == k_opt and stop_at_opt:
            break
            
        # calculate the weight of the next step
        w_k = np.exp(-alpha*d_abs_k/d_norm_k)  
            
    # convert list of parameters to numpy arrays
    w = np.array(w)
    d_norm = np.array(d_norm)
    NPR_test = np.array(NPR_test)
    NPR_avg = np.array(NPR_avg)

    plt.figure(figsize = (14,4))
    plt.subplot(1,3,1)
    plt.plot(k_vect, w)
    plt.legend(['w'])
    plt.subplot(1,3,2)
    plt.plot(k_vect, d_norm)
    plt.legend(['d'])
    plt.subplot(1,3,3)
    plt.plot(k_vect, NPR_test)
    plt.plot(k_vect, NPR_avg)
    plt.legend(['NPR test', 'NPR avg'])

    if time is not None and quiet is False:
        tempModel = np.reshape(model, (len(energy),
                                       int(data.shape[1]/len(time)),
                                       len(time)))
        model2D = np.nanmean(tempModel,axis=1)

        fig2,bx = plt.subplots(5,1,sharex=True)
        for idx,axis in enumerate(bx):
            axis.plot(energy,trans2D[:,-idx],label=str(time[-idx]))
            axis.plot(energy,model2D[:,-idx])
            axis.legend()
    
    return transAbs - model

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
    
    
