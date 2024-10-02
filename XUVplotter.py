# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:53:05 2021

@author: Bozark
"""
import XUVclass,matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import math

def plotXAS(XUVdata):
    
    data = XUVdata.aveHarmData
    ref = XUVdata.aveHarmRef
    absData = XUVdata.aveXAS
    std = XUVdata.aveXASstd
    energy = XUVdata.energy
    
    fig,(ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=False)
    # plt.ylabel("Counts")
    ax1.plot(energy,data)
    ax1.plot(energy,ref)
    ax1.set_xlabel(xlabel="Pixel",fontsize=18)
    ax1.set_ylabel(ylabel="Counts",fontsize=18)
    ax1.legend()
    
    ax2.plot(energy,absData,label = "Average Absorbance")
    ax2.set_xlabel(xlabel="Energy",fontsize=18)
    ax2.set_ylabel(ylabel="Counts",fontsize=18)
    ax2.fill_between(energy,
                     absData-std,
                     absData+std,
                     alpha=0.5,color='r')
    ax2.plot(energy,std,label = 'Standard Deviation')
    ax2.legend()

def plotStats(XUVdata):
    
    fig,(ax1,ax2) = plt.subplots(1,2)
    
    Eaxis = XUVdata.energy
    
    ax1.hist(XUVdata.aveXASstd,bins=100)
    ax1.set_title('XAS standard deviation Distribution')
    ax1.set_xlabel('Standard Deviation Bin')
    ax1.set_ylabel('Frequency')
    
    ax2.plot(Eaxis, XUVdata.dataBase,label='data abs baseline')
    ax2.plot(Eaxis, XUVdata.refBase,label = 'ref abs baseline')
    ax2.plot(Eaxis,XUVdata.aveXASstd,label='XAS std dev')
    ax2.set_xlabel('Energy (eV)')
    ax2.set_ylabel('Abs')
    ax2.legend()
    
    ax3 = ax2.twinx()
    ax3.plot(Eaxis,XUVdata.aveHarmData,label='Data harmonic')
    ax3.set_ylabel('Intensity (counts)')
    ax3.legend()
    
    fig2,ax4 = plt.subplots()
    ax4.plot(XUVdata.intCnts)
    
def plotXTAdata(Xdata,color_min=-0.05,color_max=0.05,fromGUI=False):
    """
    Plots a single XTA with variable color. If its an averaged dataset, will also plot standard deviation.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdataset.
    color_min : float, optional
        Minimum to use for color map. The default is -0.0005.
    color_max : float, optional
        Maximum to use for color map. The default is 0.0005.

    Returns
    -------
    fig : pyplot fig class
        Can be used for further alterations to figure.

    """
    energy = Xdata.energy
    TimeAxis = Xdata.time
    Intensity = Xdata.trans2D
    std = Xdata.trans2Dstd/np.sqrt(Xdata.trans3D.shape[1])
    
    # xticks = range(0,len(TimeAxis))
    # yticks = range(0,len(energy),20)
    # xticklabel = ["{:6.2f}".format(i) for i in TimeAxis[xticks]]
    # yticklabel = ["{:6.2f}".format(i) for i in energy[yticks]]

    fig1,ax1 = plt.subplots(1,1)
    # ax1.set_xticks(xticks)
    # ax1.set_xticklabels(xticklabel)
    # ax1.set_yticks(yticks)
    # ax1.set_yticklabels(yticklabel)
    
    # if not fromGUI:
    cax = ax1.imshow(Intensity,interpolation=None,
               vmin=color_min,vmax=color_max,
               extent=[TimeAxis[0],TimeAxis[-1],energy[-1],energy[0]],
               cmap='bwr',aspect='auto')
    
    fig1.colorbar(cax,ax=ax1)
    
    fig2,bx = plt.subplots(1,1,sharex=True)
    for idx,t in enumerate(TimeAxis):
        bx.errorbar(energy,Intensity[:,idx],yerr=std[:,idx],
                    capsize=3,capthick=2,
                    label=str(t)+' ps')
    
    bx.legend()
    fig2.tight_layout()
    
    return 
    
def plotTimeMap(Xdata,color_min=-0.05,color_max=0.05,fromGUI=False):
    """
    Plots a single XTA with variable color. If its an averaged dataset, will also plot standard deviation.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdataset.
    color_min : float, optional
        Minimum to use for color map. The default is -0.0005.
    color_max : float, optional
        Maximum to use for color map. The default is 0.0005.

    Returns
    -------
    fig : pyplot fig class
        Can be used for further alterations to figure.

    """
    energy = Xdata.energy
    TimeAxis = Xdata.time
    Intensity = Xdata.trans3D
    # std = Xdata.trans2Dstd
    
    # xticks = range(0,Xdata.trans3D.shape[1],20)
    # yticks = range(0,len(energy),20)
    # yticklabel = ["{:6.2f}".format(i) for i in energy[yticks]]

    plotNum = round(np.sqrt(len(TimeAxis)))

    fig1,ax = plt.subplots(plotNum,plotNum,sharey=True)
    # ax.set_xticks(xticks)
    # ax1.set_xticklabels(xticklabel)
    
    # if not fromGUI:
    for idx,axis in enumerate(ax.ravel()):
        if idx > (Intensity.shape[2]-1):
            axis.plot(Xdata.aveHarmData,energy)
            axis.set_ylim(energy[-1],energy[0])
            # axis.set_yticks(yticks)
            # axis.set_yticklabels(yticklabel)
            break
        
        cax = axis.imshow(Intensity[:,:,idx],interpolation=None,
               vmin=color_min,vmax=color_max,
               extent=[0,Intensity.shape[1],energy[-1],energy[0]],
               cmap='bwr',aspect='auto')
        # axis.set_yticks(yticks)
        axis.set_xlabel(str(TimeAxis[idx])+' ps')
        # axis.set_yticklabels(yticklabel)
    
    fig1.colorbar(cax,ax=axis)
    fig1.tight_layout()
    
    return 

def plotHarms(XUVdata):
    
    data = XUVdata.aveHarmData
    ref = XUVdata.aveHarmRef
    energy = XUVdata.energy
    
    fig,ax1 = plt.subplots(1,1)
    # plt.ylabel("Counts")
    ax1.plot(energy,data,label='Data')
    ax1.plot(energy,ref,label='Ref')
    ax1.set_xlabel(xlabel="Energy",fontsize=18)
    ax1.set_ylabel(ylabel="Intensity",fontsize=18)
    ax1.legend()
    
    return
    
def PlotKinetic(XTAdata,energy,plotFit = True, norm = False,normT = 500,plotError = True):
    """
    Takes a single TAdata, or a list of TAdata, and plots a kinetic trace. Plots any fit and error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    wvlngth : Integer
        specific wavelength to plot kinetics for.
    plotFit : Boolean, optional
        Whether to plot a fitted line, if available. The default is True.
    norm : Boolean, optional
        Whether to normalize the kinetic line to 1, at some time. The default is False.
    normT : Integer, optional
        The time in ps to normalize a kinetic trace to. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.

    """
    
    if type(XTAdata)==list:
        gcf = __PlotMK__(XTAdata, energy,plotFit,norm,normT,plotError)
    else:
        label= str(energy)
        
        if label in XTAdata.Kinetic:
            Intensity = XTAdata.Kinetic[label]
        else:
            XTAdata.KineticTrace(energy,1)
            Intensity = XTAdata.Kinetic[label]
        time = XTAdata.time
        
        if norm:
            norm_i = [i for i,x in enumerate(time) if math.isclose(x, normT,rel_tol=(0.2))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
        
        # if label in XTAdata.FitResults and plotFit:
        #     fit_wvl = XTAdata.KineticFit[label]
        
        if plotError:
            error = XTAdata.KineticStd[label]
        
        gcf, ax = plt.subplots()
        if plotError:
            ax.errorbar(time,Intensity/normFact,yerr=error/normFact,capsize=0.5,label = XTAdata.name)
        else:
            ax.plot(time,Intensity/normFact,'.',label = XTAdata.name)
            
        # if label in XTAdata.FitResults and plotFit:
        #     ax.plot(time[0:len(fit_wvl)],fit_wvl/normFact,label = XTAdata.name+"fit")
        
        plt.legend()
        
    return gcf
    
def PlotSpectral(TAdata,time_slice,norm = False,normW = 500,plotError = False):
    """
    Takes a single TAdata, or a list of TAdata, and plots a spectral trace. Plots error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    time_slice : float
        time to plot spectral traces at.
    norm : Boolean, optional
        Whether to normalize the spectral trace to 1, at some wavelength. The default is False.
    normW : integer, optional
        The wavelength to normalize a spectral trace to 1 at. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.
    """
    if type(TAdata)==list:
        gcf = __PlotMS__(TAdata, time_slice, norm, normW,plotError)
    else:
        label= str(time_slice)
        
        if label in TAdata.T_slice:
            Intensity = TAdata.T_slice[label]
        else:
            TAdata.SpectralTrace(time_slice,time_slice/10)
            Intensity = TAdata.T_slice[label]
        
        if(TAdata.Ave):
            error = TAdata.T_sliceStd[label]
    
        Wavelengths = TAdata.Wavelength
        gcf, ax = plt.subplots()
        
        if norm:
            norm_i = [i for i,x in enumerate(Wavelengths) if math.isclose(x, normW,rel_tol=(0.01))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
            
        if TAdata.Ave and plotError:
            error = TAdata.T_sliceStd[label]
            
        if TAdata.Ave and plotError:
            ax.errorbar(Wavelengths,Intensity/normFact,yerr=error/normFact,capsize=0.5,label =  TAdata.name)
        else:
            ax.plot(Wavelengths,Intensity/normFact,'r.',label = TAdata.name)
            
    return gcf

def __PlotMK__(XTAdata,energy,plotFit = False, norm = False,normT = 500,plotError = False):
    """
    Takes a list of TAdata and plots a kinetic trace. Plots any fit and error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    wvlngth : Integer
        specific wavelength to plot kinetics for.
    plotFit : Boolean, optional
        Whether to plot a fitted line, if available. The default is True.
    norm : Boolean, optional
        Whether to normalize the kinetic line to 1, at some time. The default is False.
    normT : Integer, optional
        The time in ps to normalize a kinetic trace to. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.
    """
    label= str(energy)
    
    gcf, ax = plt.subplots()
    cmap = get_cmap(len(XTAdata)+1)
    
    for i,data in enumerate(XTAdata):
        Intensity = data.Kinetic[label]
        time = data.time
        
        if norm:
            norm_i = [i for i,x in enumerate(time) if math.isclose(x, normT,rel_tol=(0.2))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
        
        if label in data.FitResults and plotFit:
            fit_wvl = data.KineticFit[label]
        
        if data.Ave and plotError:
            error = data.KineticStd[label]
        
        if data.Ave and plotError:
            ax.errorbar(time,Intensity/normFact,yerr=error/normFact,fmt='.',capsize=1,label = data.name,color = cmap(i))
        else:
            ax.plot(time,Intensity/normFact,'.',label = data.name,color = cmap(i))
            
        if label in data.FitResults and plotFit:
            ax.plot(time,fit_wvl/normFact,label = data.name+"fit",c=cmap(i))
        
        plt.legend()
        
    return gcf

def __PlotMS__(TAdata,time_slice,norm = False,normW = 500,plotError = False):
    """
    Takes a list of TAdata, and plots a spectral trace. Plots error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    time_slice : float
        time to plot spectral traces at.
    norm : Boolean, optional
        Whether to normalize the spectral trace to 1, at some wavelength. The default is False.
    normW : integer, optional
        The wavelength to normalize a spectral trace to 1 at. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.
    """
    
    label= str(time_slice)
    
    gcf, ax = plt.subplots()
    cmap = get_cmap(len(TAdata)+1)
    
    for i,data in enumerate(TAdata):
        Intensity = data.T_slice[label]
        Wavelengths = data.Wavelength
        
        if norm:
            norm_i = [i for i,x in enumerate(Wavelengths) if math.isclose(x, normW,rel_tol=(0.01))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
            
        if data.Ave and plotError:
            error = data.T_sliceStd[label]
            
        if data.Ave and plotError:
            ax.errorbar(Wavelengths,Intensity/normFact,yerr=error/normFact,capsize=0.5,label = str(i))
        else:
            ax.plot(Wavelengths,Intensity/normFact,'r.',markevery =None,label = data.name,c=cmap(i))
            
        plt.legend()
            
    return gcf

def PlotMSS(TAdata,time_points,norm = False,normW = 500,plotError = False):
    """
    Takes a single TAdata and plots a range of spectral traces. Can plot error along with it.

    Parameters
    ----------
    TAdata : TAdata
        A single TAdata structure.
    time_points : List
        Plots multiple spectral traces at these times.
    norm : Boolean, optional
        Whether to normalize the spectral trace to 1, at some wavelength. The default is False.
    normW : integer, optional
        The wavelength to normalize a spectral trace to 1 at. The default is 500.
    plotError : Boolean, optional
        Whether you also want to plot the standard deviation for the kinetic trace. The default is True.

    Returns
    -------
    gcf : pyplot figure class
        A structure to use if you want to make further changes to the figure.
    """
    gcf, ax = plt.subplots()
    cmap = get_cmap(len(time_points)+1)
    
    for i,t_pnts in enumerate(time_points):
        label=str(t_pnts)
        
        if label in TAdata.T_slice:
            data = TAdata.T_slice[label][:]
        else:
            TAdata.SpectralTrace(t_pnts,t_pnts/10)
            data = TAdata.T_slice[label][:]
        
        Intensity = TAdata.T_slice[label]
        Wavelengths = TAdata.Wavelength
        
        if norm:
            norm_i = [i for i,x in enumerate(Wavelengths) if math.isclose(x, normW,rel_tol=(0.01))]
            normFact = np.mean(Intensity[norm_i])
        else:
            normFact = 1
            
        if TAdata.Ave and plotError:
            error = TAdata.T_sliceStd[label]
            
        if TAdata.Ave and plotError:
            ax.errorbar(Wavelengths,Intensity/normFact,yerr=error/normFact,capsize=0.5,label = label)
        else:
            ax.plot(Wavelengths,Intensity/normFact,'-',markevery =None, label = label,c=cmap(i))
            
        #future todo when a robust spectral fitting is implemented    
        # if plotFit: #label in TAdata.SpectralFit and
        #     fit_data = TAdata.SpectralFit[label]
        #     ax.plot(Wavelengths,fit_data/normFact,label = "fit"+str(i),c=cmap(i))
            
        plt.legend()
            
    return gcf

def get_cmap(i,name = 'hsv'):
    return plt.cm.get_cmap(name,i)
