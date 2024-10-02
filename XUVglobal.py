# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:24:12 2023

@author: Samth
"""

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
from numpy.lib.scimath import sqrt,log
import io
import TAplotter as TAplt
from glotaran.io import load_model 
from glotaran.io import load_dataset

DATA_PATH = "C:/Users/Samth/Documents/Samples/OTA"
MODEL_PATH = "C:/Users/Samth/Documents/Samples/OTA"
PARAM_PATH = "C:/Users/Samth/Documents/Samples/OTA"



def oneExpGlobal(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,A1 = -0.0005,
              varyC1=True,varyIRF=False,varyt0=False,varyA1=True, quiet = False):
    
    compiled_data=[]
    compiled_time=[]
    compiled_std=[]
    
    for data in OTAData:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
        compiled_time.append(time)
    
    for data,idx in zip(OTAData,wvlngth):
        label = str(wvlngth)
        if label in data.Kinetic:
            trace = data.Kinetic[label][time_index]
            compiled_data.append(trace)
        else:
            data.KineticTrace(wvlngth,0.5)
            compiled_data.append(data.Kinetic[label][time_index])
        
        if OTAData.Ave:
            compiled_std.append(data.KineticStd[label][time_index])
    
    #clear the data of nan's
    for time,std,data in zip(compiled_time,compiled_std,compiled_data):
        nan_idx = np.where(np.isnan(data),False,True)
        time = time[nan_idx]
        std = std[nan_idx]
        data = data[nan_idx]
    
    params = Parameters()
    params.add('IRF',value=IRF,vary=varyIRF)
    params.add('t0',value=t0, vary=varyt0)
    for x in range(len(OTAData)):
        params.add(f'C1_{x}', value= C1, min=0.1, vary=varyC1)
        params.add('A1_{x}',value=A1, vary=varyA1)
        params.add('rise_{x}',value=rise,vary=False)
    
    for x in range(len(OTAData)):
        params[f'C1_{x}'].expr='C1_0'
        
    if OTAData.Ave:
        results = minimize(flatResid,params,args=(compiled_time,compiled_data,compiled_std))
    else: 
        results = minimize(flatResid,params,args=(compiled_time,compiled_data))

    if not quiet:
        print(fit_report(results))
    
    fit_params = results.params
    parvals = fit_params.valuesdict()
    
    for i,data in enumerate(OTAData):
        C1 = parvals[f'C1_{i}']
        A1 = parvals[f'A1_{i}']
        IRF = parvals['IRF']
        t0= parvals['t0']
        rise = parvals[f'rise_{i}']
        
        time = data.Time
        if rise:
            data.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))
        else:
            data.KineticFit[label]= A1*erfexp(C1,IRF,t0,time)
        
        data.FitResults[label]=results
        
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def flatResid(params,time,data,std):
    resid = np.zeros((1))
    
    for i,x in enumerate(data):
        resid.append(globalOneExpRes(params, i, x, time[i],std[i]))
    
    return resid

def globalOneExpRes(params,i,time,data,std=[]):
    parvals = params.valuesdict()
    
    # time_arr = np.linspace(-50,100,9001)
    # arr_0 = np.where(time_arr==0)[0][0]
    C1 = parvals[f'C1_{i}']
    A1 = parvals[f'A1_{i}']
    IRF = parvals['IRF']
    t0= parvals['t0']
    std_FWHM_const=2*sqrt(2*log(2))
    rise = parvals[f'rise_{i}']
    
    if rise:
        model = A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))
    else:
        model = A1*erfexp(C1,IRF,t0,time)
    
    if len(std)!=0:
        tmpStd = np.where(np.isclose(std,np.zeros(np.shape(std))),1,std)
        tmpStd = np.where(np.isnan(std),1,std)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def twoExpFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,\
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
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
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
        
    #resid = Residual(params,A,B,C, ZnOkinetic,UVTime,QDkinetic,VisTime)
    if OTAData.Ave:
        results = minimize(twoExpResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(twoExpResidual,params,args=(time,data),nan_policy='omit')

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
    
    time = OTAData.Time
    if rise:
        OTAData.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))
    else:
        OTAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def twoExpResidual(params,time,data,std = []):
    parvals = params.valuesdict()
    
    # time_arr = np.linspace(-50,100,9001)
    # arr_0 = np.where(time_arr==0)[0][0]
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
        tmpStd = np.where(np.isnan(std),1,std)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def threeExpFit(OTAData,wvlngth, rise = False,IRF = 0.13 , t0 = 0, C1 = 1,\
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
                                        [i for i,z in enumerate(OTAData.Time) if z < x or z>y]))
        
        idx,cnt = np.unique(time_index,return_counts=True)
        time_index = idx[cnt == regions] 
        time = OTAData.Time[time_index]
    else:
        time_index = range(0,len(OTAData.Time))
        time = OTAData.Time[time_index]
    
    if label in OTAData.Kinetic:
        data = OTAData.Kinetic[label][time_index]
    else:
        OTAData.KineticTrace(wvlngth,0.5)
        data = OTAData.Kinetic[label][time_index]
    
    if OTAData.Ave:
        fit_std = OTAData.KineticStd[label][time_index]
    
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
    params.add('C3', value= C3, min=0.1, vary=varyC3)
    params.add('A3',value= A3, vary=varyA3)
    params.add('rise',value=rise,vary=False)
        
    if OTAData.Ave:
        results = minimize(threeExpResidual,params,args=(time,data,fit_std))
        # results = minimize(twoplusTPAResidual,params,args=(time,data,fit_std),nan_policy='omit')
    else: 
        results = minimize(threeExpResidual,params,args=(time,data),nan_policy='omit')

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
    
    time = OTAData.Time
    if rise:
        OTAData.KineticFit[label]= A1*(erfexp(1e6,IRF,t0,time) - erfexp(C1,IRF,t0,time))\
            +A2*(erfexp(1e6,IRF,t0,time) - erfexp(C2,IRF,t0,time))\
                +A3*(erfexp(1e6,IRF,t0,time) - erfexp(C3,IRF,t0,time)) 
    else:
        OTAData.KineticFit[label]= A1*erfexp(C1,IRF,t0,time) +A2* erfexp(C2,IRF,t0,time)\
            +A3* erfexp(C3,IRF,t0,time)
    
    OTAData.FitResults[label]=results
    if not quiet:
        TAplt.PlotKinetic(OTAData, wvlngth)
    
    return results

def threeExpResidual(params,time,data,std = []):
    parvals = params.valuesdict()
    
    # time_arr = np.linspace(-50,100,9001)
    # arr_0 = np.where(time_arr==0)[0][0]
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
        tmpStd = np.where(np.isnan(std),1,std)
        return np.divide((model-data),tmpStd)
    else:
        return model-data

def erfexp(tau,IRF,t0, t):
    
    val = np.where((t-t0)>-5, np.exp((IRF/1.65511/2/tau)**2-(t-t0)/tau)*\
           (erf((t-t0)/IRF*1.65511-(IRF/1.65511/2/tau))+1)/2,0)
    return val