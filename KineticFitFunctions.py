# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:26:57 2022

@author: Bozark
"""
#module holding kinetic fits
import sympy as sym
import sympy.utilities.lambdify as lambdify
import matplotlib.pyplot as plt
import numpy as np
from lmfit import minimize, Parameters,fit_report
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.stats import lognorm
from scipy.integrate import quad
from numpy.lib.scimath import sqrt,log
import io,os
import SpectralFitFunctions as sff
import XUVplotter as Xplt
import inspect as ins

def fitInit(comp=1):
    """
    Parameters
    ----------
    comp : TYPE, optional
        The default is 1, and describes the sum of exponential fitting funct.
        Pass in comp=0 for the t_zero function.

    Returns
    -------
    fncDict : TYPE
        A dictionary of the default values for any given fitting function.
        Can be amended and passed in to manyFitKinetic in place of kwargs with "**fncDict"

    """
    if comp==0:
        signature = ins.signature(t_zero)
    elif comp==1:
        signature = ins.signature(oneExpFit)
    elif comp==2:
        signature = ins.signature(twoExpFit)
    elif comp==3:
        signature = ins.signature(threeExpFit)
    elif comp==4:
        signature = ins.signature(fourExpFit)
    elif comp==5:
        signature = ins.signature(stretchExpFit)

    fncDict =  {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not ins.Parameter.empty
    }
    
    fncDict.pop('quiet')
    
    return fncDict

def manyKinetic(TAdata,wvlngth,binw, norm = True,normT = 500, clean = True,cleanFactor=3):
    """Take a list of TAdata and extract kinetic traces for each, then plot together. Can bin over a wavelength range, along with
    options to normalize the plotting. Clean is an option to remove outliers."""
    
    for data in TAdata:
        data.KineticTrace(wvlngth,binw)
        if clean:
            CleanKin(data,wvlngth,factor=cleanFactor)
            
    Xplt.PlotMK(TAdata, wvlngth, norm = norm,normT = normT)
    
def manySpectral(TAdata,Time,binw,norm = True,normW = 500):
    """Take a list of TAdata and extract spectral traces for each, then plot together. Can bin over a wavelength range, along with
    options to normalize the plotting."""
  
    for data in TAdata:
        data.SpectralTrace(Time,binw)
            
    Xplt.PlotMS(TAdata,Time,norm = norm,normW = normW )

def manyFitKinetic(TAdata,wvlngth,comp,**kwargs):
    """
    Function to take a list of datasets, and fit a kinetic trace to the same exp model.

    Parameters
    ----------
    TAdata : list
        List of TAdata.
    wvlngth : int
        Wavelength to fit for each dataset.
    comp : int between 1 and 5
        Numbers 1 through 4 are # of exponential components to fit. 
        5 components is actually a stretched exponential fit.
    **kwargs : dictionary
         Dictionary of parameters to pass to exponential fitting. Can include amplitude (A1...) or time (C1...).

    Returns
    -------
    results : list
        List of lmfit minimizer results. Can be passed to saveFits function to export results.

    """
    
    results = []
    for data in TAdata:
        if comp ==1:
            results.append(oneExpFit(data, wvlngth,quiet = True,**kwargs))
        elif comp ==2:
            results.append(twoExpFit(data, wvlngth,quiet = True,**kwargs))
        elif comp ==3:
            results.append(threeExpFit(data, wvlngth,quiet = True,**kwargs))
        elif comp ==4:
            results.append(fourExpFit(data, wvlngth,quiet = True,**kwargs))
        elif comp == 5:
            results.append(stretchExpFit(data, wvlngth, quiet=True, **kwargs))    
        else:
            results.append(lognormFit(data, wvlngth, quiet=True, **kwargs))    
            
    for i in results:
        params = i.params.valuesdict()
        print("aic: " + str(i.aic))
        print("bic: " + str(i.bic))
        print("chi2: " + str(i.chisqr))
        print("C1: " + str(params['C1']))
            
    return results

def manyFitSpectral(TAdata,t_slice,params,wvl_mask=[],**kwargs):
    """
    Takes a list of TAdata and extracts and plots a spectral slice. Unfinished functioin

    Parameters
    ----------
    TAdata : List
        List of TAdata.
    t_slice : time
        DESCRIPTION.
    params : Fitting parameters
        DESCRIPTION.
    wvl_mask : list of wavelengths to exclude, optional
        DESCRIPTION. The default is [].
    **kwargs : dictionary 
        Used to pass arguments to the spectral fitting function..

    Returns
    -------
    results : list of lmfit results.

    """
    results=[]
    for i in TAdata:
        results.append(sff.FitSingleSpectral(i, t_slice, params,wvl_mask=wvl_mask,**kwargs))
        
    return results

def CleanKin(TAdata,wvlngth,factor = 3):
    """
    Remove outliers from kinetic trace. Takes a single TAdata and wavelength, along with a factor for the number of standard deviations
    to consider an outlier.

    Parameters
    ----------
    TAdata : TAdata
        A single dataset to correct a kinetic trace for.
    wvlngth : Int
        The wavelength at which to remove outliers from.
    factor : integer, optional
        The number of standard deviations away from the mean to consider a point as an outlier. The default is 3.

    Returns
    -------
    None.

    """
    label = str(wvlngth)
    
    stdMean = np.median(TAdata.KineticStd[label])
    if np.mean(TAdata.KineticStd[label])<stdMean:
        stdMean = np.mean(TAdata.KineticStd[label])

    for i,pnt in enumerate(TAdata.Kinetic[label]):
        if i ==0:
            mean = 0
        elif i > (len(TAdata.Kinetic[label])-3):
            mean = sum(TAdata.Kinetic[label][i-4:i-1])/4
        else:
            mean = (TAdata.Kinetic[label][i-1]+
                    TAdata.Kinetic[label][i+1]+
                    TAdata.Kinetic[label][i-2]+
                    TAdata.Kinetic[label][i+2])/4
        if abs(pnt) > (abs(mean)+stdMean*factor):
            TAdata.Kinetic[label][i] = mean
            TAdata.KineticStd[label][i] = stdMean


def oneExpFit(TAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,A1 = -0.0005,
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True, quiet = False,
              time_mask=[]):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(TAData.time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = TAData.time[time_index]
    else:
        time_index = range(0,len(TAData.time))
        time = TAData.time[time_index]
    
    if label in TAData.Kinetic:
        data = TAData.Kinetic[label][time_index]
    else:
        TAData.KineticTrace(wvlngth,0.5)
        data = TAData.Kinetic[label][time_index]
    
    fit_std = TAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    data = data[nan_idx]
    fit_std = fit_std[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value=A1, vary=varyA1)
    params.add('t0',value=t0, vary=varyt0)
    params.add('rise',value=rise,vary=False)
        
    results = minimize(oneExpResidual,
                       params,
                       args=(time,data,fit_std),
                       nan_policy='omit')
    
    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = TAData.time
    if rise:
        TAData.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))
    else:
        TAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time)
    
    TAData.FitResults[label]=results
    if not quiet:
        Xplt.PlotKinetic(TAData, wvlngth)
    
    return results

def oneExpResidual(params,time,data,std=[]):
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model = A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))
    else:
        model = A1*erfexp(C1,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def twoExpFit(TAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,\
              A1 = -0.0005,C2 = 10,A2 = -0.0005, time_mask = [],
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True,
              varyC2=True,varyA2=True,quiet = False):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(TAData.time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = TAData.time[time_index]
    else:
        time_index = range(0,len(TAData.time))
        time = TAData.time[time_index]
    
    if label in TAData.Kinetic:
        data = TAData.Kinetic[label][time_index]
    else:
        TAData.KineticTrace(wvlngth,0.5)
        data = TAData.Kinetic[label][time_index]
    
    fit_std = TAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('t0',value= t0, vary=varyt0)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value= A1, vary=varyA1)
    params.add('C2', value= C2, min=0.1, vary=varyC2)
    params.add('A2',value= A2, vary=varyA2)
    params.add('rise',value=rise,vary=False)
        
    results = minimize(twoExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    
    if not quiet:
       print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = TAData.time
    if rise:
        TAData.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))
    else:
        TAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)
    
    TAData.FitResults[label]=results
    if not quiet:
        Xplt.PlotKinetic(TAData, wvlngth)
    
    return results

def twoExpResidual(params,time,data,std = []):
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model = A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))+A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))
    else:
        model = A1*erfexp(C1,IRF,t0,time)+A2*erfexp(C2,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def threeExpFit(TAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,\
              A1 = -0.0005,C2 = 10,A2 = -0.0005, C3 = 500,A3 = -0.0005,time_mask = [], 
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True,
              varyC2=True,varyA2=True,varyC3=True,varyA3=True,quiet=False):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(TAData.time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = TAData.time[time_index]
    else:
        time_index = range(0,len(TAData.time))
        time = TAData.time[time_index]
    
    if label in TAData.Kinetic:
        data = TAData.Kinetic[label][time_index]
    else:
        TAData.KineticTrace(wvlngth,0.5)
        data = TAData.Kinetic[label][time_index]
    
    fit_std = TAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('t0',value= t0, vary=varyt0)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('C2', value= C2, min=0.1, vary=varyC2)
    params.add('C3', value= C3, min=0.1, vary=varyC3)
    if  (type(A1)==str):
        params.add('A1',expr= A1, vary=varyA1)
    else:
        params.add('A1',value= A1, vary=varyA1)
    if  (type(A2)==str):
        params.add('A2',expr= A2, vary=varyA2)
    else:
        params.add('A2',value= A2, vary=varyA2)
    if  (type(A3)==str):
        params.add('A3',expr= A3, vary=varyA3)
    else:
        params.add('A3',value= A3, vary=varyA3)
    
    params.add('rise',value=rise,vary=False)
        
    results = minimize(threeExpResidual,params,args=(time,data,fit_std))
    
    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    C3 = parvals['C3']
    A3 = parvals['A3']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = TAData.time
    if rise:
        TAData.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time)) 
    else:
        TAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time)
    
    TAData.FitResults[label]=results
    if not quiet:
        Xplt.PlotKinetic(TAData, wvlngth)
    
    return results

def threeExpResidual(params,time,data,std = []):
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    C3 = parvals['C3']
    A3 = parvals['A3']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time))
    else:
        model= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data
    
def fourExpFit(TAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,
              A1 = -0.0005,C2 = 10,A2 = -0.0005, C3 = 500,A3 = -0.0005,
              A4 = -0.001,C4 = 1000, time_mask = [], quiet=False):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(TAData.time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = TAData.time[time_index]
    else:
        time_index = range(0,len(TAData.time))
        time = TAData.time[time_index]
    
    if label in TAData.Kinetic:
        data = TAData.Kinetic[label][time_index]
    else:
        TAData.KineticTrace(wvlngth,0.5)
        data = TAData.Kinetic[label][time_index]
    
    fit_std = TAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=False)
    params.add('t0',value= t0, vary=False)
    params.add('C1', value= C1, min=0.1, vary=False)
    params.add('A1',value= A1,max=0, vary=True)
    params.add('C2', value= C2, min=0.1, vary=True)
    params.add('A2',value= A2,max=0, vary=True)
    params.add('C3', value= C3, min=0.1, vary=True)
    params.add('A3',value= A3,max=0, vary=True)
    params.add('C4', value= C4, min=0.1, vary=True)
    params.add('A4',value= A4,min=0, vary=True)
    params.add('rise',value=rise,vary=False)
        
    results = minimize(fourExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    
    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    C3 = parvals['C3']
    A3 = parvals['A3']
    C4 = parvals['C4']
    A4 = parvals['A4']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = TAData.time
    if rise:
        TAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time)\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time))\
                    +A4*(erfexp(1e6,IRF,t0,time) -erfexp(C4, IRF, t0, time))
    else:
        TAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time) +A4*erfexp(C4,IRF,t0,time)
    
    TAData.FitResults[label]=results
    if not quiet:
        Xplt.PlotKinetic(TAData, wvlngth)
    
    return results

def fourExpResidual(params,time,data,std = []):
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    C2 = parvals['C2']
    A2 = parvals['A2']
    C3 = parvals['C3']
    A3 = parvals['A3']
    C4 = parvals['C4']
    A4 = parvals['A4']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model= A1*erfexp(C1,IRF,t0,time)\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time))\
                    +A4*(erfexp(1e6,IRF,t0,time) -erfexp(C4, IRF, t0, time))
    else:
        model= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time) +A4*erfexp(C4,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def stretchExpFit(TAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,A1 = -0.0005,B = 1,
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True, quiet = False,varyB=True,
              time_mask=[]):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(TAData.time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = TAData.time[time_index]
    else:
        time_index = range(0,len(TAData.time))
        time = TAData.time[time_index]
    
    if label in TAData.Kinetic:
        data = TAData.Kinetic[label][time_index]
    else:
        TAData.KineticTrace(wvlngth,0.5)
        data = TAData.Kinetic[label][time_index]
    
    fit_std = TAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value=A1, vary=varyA1)
    params.add('t0',value=t0, vary=varyt0)
    params.add('B',value=B, min=0.01,vary=varyB)
    params.add('rise',value=rise,vary=False)
        
    results = minimize(stretchExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    
    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    time = TAData.time
    if rise:
        TAData.KineticFit[label] = np.where(time<0,A1*(erfexp(1e6, IRF, t0, time)-erfexp(C1,IRF,t0,time)),
                         A1*(stretcherfexp(1e6,IRF,t0,B,time) - stretcherfexp(C1,IRF,t0,B,time)))
    else:
        TAData.KineticFit[label] = np.where(time<0,A1*erfexp(C1,IRF,t0,time),
                         A1*stretcherfexp(C1,IRF,t0,B,time))
    
    TAData.FitResults[label]=results
    if not quiet:
        Xplt.PlotKinetic(TAData, wvlngth)
    
    return results

def stretchExpResidual(params,time,data,std=[]):
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    B = parvals['B']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    if rise:
        model = np.where(time<0,A1*(erfexp(1e6, IRF, t0, time)-erfexp(C1,IRF,t0,time)),
                         A1*(stretcherfexp(1e6,IRF,t0,B,time) - stretcherfexp(C1,IRF,t0,B,time)))
    else:
        model = np.where(time<0,A1*erfexp(C1,IRF,t0,time),
                         A1*stretcherfexp(C1,IRF,t0,B,time))

    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def distExpFit(TAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0,
               varyIRF=False,varyt0=False,quiet = False,time_mask=[]):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(TAData.time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = TAData.time[time_index]
    else:
        time_index = range(0,len(TAData.time))
        time = TAData.time[time_index]
    
    if label in TAData.Kinetic:
        data = TAData.Kinetic[label][time_index]
    else:
        TAData.KineticTrace(wvlngth,0.5)
        data = TAData.Kinetic[label][time_index]
    
    fit_std = TAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    fit_std = fit_std[nan_idx]
    data = data[nan_idx]
    
    C_arr = np.logspace(0.5, 5000,num=20)
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    for idx in range(len(C_arr)):
        params.add('A'+str(idx),value=1/len(C_arr),vary=True)
    params.add('t0',value=t0, vary=varyt0)
    params.add('rise',value=rise,vary=False)
        
    results = minimize(distExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    
    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    model = np.zeros(time.shape)
    
    time = TAData.time
    TAData.KineticFit[label] = np.zeros(time.shape)
    if rise:
        for idx,tau in enumerate(C_arr):
            TAData.KineticFit[label].add(parvals['A'+str(idx)]*(erfexp(1e6,IRF,t0,time) - erfexp(tau,IRF,t0,time)))
    else:
        for idx,tau in enumerate(C_arr):
            TAData.KineticFit[label].add(parvals['A'+str(idx)]*erfexp(tau,IRF,t0,time))
    
    TAData.FitResults[label]=results
    if not quiet:
        Xplt.PlotKinetic(TAData, wvlngth)
    
    return results

def distExpResidual(params,time,data,std=[]):
    parvals = params.valuesdict()
    
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    C_arr = np.logspace(0.5, 5000,num=20)
    model = np.zeros(time.shape)
    
    if rise:
        for idx,tau in enumerate(C_arr):
            model.add(parvals['A'+str(idx)]*(erfexp(1e6,IRF,t0,time) - erfexp(tau,IRF,t0,time)))
    else:
        for idx,tau in enumerate(C_arr):
            model.add(parvals['A'+str(idx)]*erfexp(tau,IRF,t0,time))
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(std),1,std)
        return np.divide((model-data),tmpStd)
    else:
        return model-data
    
def lognormFit(TAData,wvlngth, rise = False,IRF = 0.13, t0 = 0, C1 = 1,A1 = -0.0005,
               sigma=1,varysigma=True,plotInit=False,
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True, quiet = False,
              time_mask=[]):
    
    label = str(wvlngth)
    if len(time_mask) >0:
        time_index = np.zeros(0,dtype=int)
        regions = len(time_mask)
        for r in range(regions):
            x,y = time_mask[r]
            time_index = np.concatenate((time_index,
                                        [i for i,z in enumerate(TAData.time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = TAData.time[time_index]
    else:
        time_index = range(0,len(TAData.time))
        time = TAData.time[time_index]
    
    if label in TAData.Kinetic:
        data = TAData.Kinetic[label][time_index]
    else:
        TAData.KineticTrace(wvlngth,0.5)
        data = TAData.Kinetic[label][time_index]
    
    fit_std = TAData.KineticStd[label][time_index]
    
    #clear the data of nan's
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    data = data[nan_idx]
    fit_std = fit_std[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('C1', value= C1, min=0.1, vary=varyC1)
    params.add('A1',value=A1, vary=varyA1)
    params.add('sigma',value=sigma,min=0.1,max=10,vary=varysigma)
    params.add('t0',value=t0, vary=varyt0)
    params.add('rise',value=rise,vary=False)
        
    if plotInit:
        dist_k = np.linspace(0.1,1000.1,5000)
        dist_tau = lognorm.pdf(dist_k,sigma,loc=C1)
        norm_k=dist_tau.sum()
        dist_tau/=norm_k
        
        if rise:
            model = np.zeros(time.shape)
            for c,k in zip(dist_k,dist_tau):
                tmp_arr = k*(erfexp(1e6,IRF,t0,time)-erfexp(c,IRF,t0,time))
                model += tmp_arr
        else:
            model = np.zeros(time.shape)
            for c,k in zip(dist_k,dist_tau):
                tmp_arr = k*erfexp(c,IRF,t0,time)
                model += tmp_arr
                
        model*=A1
        fig,ax = plt.subplots()
        ax.plot(time,model)
        ax.plot(time,data)
    
    results = minimize(lognormResid,params,args=(time,data,fit_std),nan_policy='raise')
    
    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    sigma=parvals['sigma']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    dist_k = np.linspace(0.1,1000.1,5000)
    dist_tau = lognorm.pdf(dist_k,sigma,loc=C1)
    norm_k=dist_tau.sum()
    dist_tau/=norm_k
    
    time = TAData.time
    
    if rise:
        model = np.zeros(time.shape)
        for c,k in zip(dist_k,dist_tau):
            tmp_arr = k*(erfexp(1e6,IRF,t0,time)-erfexp(c,IRF,t0,time))
            model += tmp_arr
    else:
        model = np.zeros(time.shape)
        for c,k in zip(dist_k,dist_tau):
            tmp_arr = k*erfexp(c,IRF,t0,time)
            model += tmp_arr
            
    TAData.KineticFit[label] = A1*model
    TAData.FitResults[label]=results
    
    if not quiet:
        Xplt.PlotKinetic(TAData, wvlngth)
    
    fig,ax = plt.subplots()
    ax.plot(dist_k,dist_tau)
    
    return results

def lognormResid(params,time,data,std=[]):
    
    parvals = params.valuesdict()
    
    C1 = parvals['C1']
    A1 = parvals['A1']
    # k = parvals['k']
    sigma = parvals['sigma']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals['rise']
    
    dist_k = np.linspace(0.1,1000.1,5000)
    dist_tau = lognorm.pdf(dist_k,sigma,loc=C1)
    norm_k=dist_tau.sum()
    dist_tau/=norm_k
    
    if rise:
        model = np.zeros(time.shape)
        for c,k in zip(dist_k,dist_tau):
            tmp_arr = k*(erfexp(1e6,IRF,t0,time)-erfexp(c,IRF,t0,time))
            model += tmp_arr
    else:
        model = np.zeros(time.shape)
        for c,k in zip(dist_k,dist_tau):
            tmp_arr = k*erfexp(c,IRF,t0,time)
            model += tmp_arr
            
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(tmpStd),1,tmpStd)
        return np.divide((A1*model-data),tmpStd)
    else:
        return A1*model-data

def t_zero(TAData,wvlngth,quiet=False,A1=-0.005):
    
    label = str(wvlngth)
    time = TAData.time
    data = TAData.Kinetic[label]
    
    fit_std = TAData.KineticStd[label]
    
    nan_idx = np.where(np.isnan(data),False,True)
    time = time[nan_idx]
    data = data[nan_idx]
    fit_std = fit_std[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=0.13,vary=True)
    params.add('C1', value= 0.1, min=0.1, vary=True)
    params.add('A1',value= -0.005, vary=True)
    params.add('t0',value= 0, vary=True)
    params.add('rise',value= False, vary=False)
    
    results = minimize(oneExpResidual,params,args=(time,data,fit_std),nan_policy='raise')
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    C1 = parvals['C1']
    A1 = parvals['A1']
    IRF = parvals['IRF']
    t0= parvals['t0']
    
    single_fit = A1*erfexp(C1,IRF,t0,time)
    
    return t0

def stretcherfexp(tau,IRF,t0, B,t):
    
    val = np.where((t-t0)>-5, np.exp((IRF/1.65511/2/tau)**2-((t-t0)/tau)**B)*\
           (erf((t-t0)/IRF*1.65511-(IRF/1.65511/2/tau))+1)/2,0)
    return val

def erfexp(tau,IRF,t0, t):
    
    val = np.where((t-t0)>-5, np.exp((IRF/1.65511/2/tau)**2-(t-t0)/tau)*\
           (erf((t-t0)/IRF*1.65511-(IRF/1.65511/2/tau))+1)/2,0)
    return val