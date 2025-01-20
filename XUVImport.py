# -*- coding: utf-8 -*-

import os, shutil, re
import numpy as np
import math , pickle
import os.path as pth
import sys
from XUVclass import XASdata,XUVimg
import read_SPE as spe
from SPErebin import pixelBin,energyBin
from configparser import ConfigParser
from AbsCalcs import XUVabs, XUVtransAbs
import XUVplotter as Xplt
import CrossECorrection as Ecorr 

class dataImporter():
    def __init__(self, DICT):
        self.directory = pth.abspath(DICT['DIRECTORY'])
        
        params = ConfigParser()
        params.read(DICT['Bin_params'])
        self.params = params
        self.dict = DICT
        self.rawImg = None
        
    def import_ET(self):
        if self.dict['STATIC_YN']: 
           self.time=[] 
        else: #for transient data, load time
            timeTemp = np.loadtxt(
                pth.join(self.directory, self.dict['TIME_DIR'])) #load raw file
            #convert 2-column time data to 1 column
            self.time = np.where(timeTemp[1]<0,
                                -timeTemp[0],timeTemp[0]) 
        
        tempE = np.loadtxt(
            pth.join(self.directory,self.dict['ENERGY_DIR']))
        self.energyF = tempE[:,1]
    
    def importImage(self,file):
        if type(file) == str:
            self.gen = spe.loadLarge(file)
        else:
            self.gen = spe.loadLarge(file.path)
    
    def harmonic(self,img,file):
        self.importImage(file)
        rawImg = next(self.gen)
        ebin,std,Eaxis = energyBin(img, self.energyF, self.params)
        return XUVimg(rawImg,ebin,std,Eaxis,file.name)
    
    def staticAbs(self, spectrum, Std, Ref):
        return XUVabs(spectrum, Std, Ref)
    
    def transAbs(self,spectrum,Std,Ref):
        return XUVtransAbs(spectrum, Std, Ref)
    
    def pipe(self,giveRaw=False,givePxlBin=False,giveEBin=False,corrEnrg=True):
        #Bring in time and energy axes
        self.import_ET()
        
        #loop through all spe files in directory
        fileGen= (file for file in os.scandir(self.directory) 
                  if file.path.endswith('.spe'))
        
        #loop through files in directory
        for file in fileGen:
            #import one image at a time
            self.importImage(file)
            
            if giveRaw:
                rawImg = [img for img in self.gen]
                yield rawImg
            else:
                rawImg = [pixelBin(img) for img in self.gen]
            
            if givePxlBin:
                return rawImg
            
            if len(rawImg)==1:
                yield self.harmonic(rawImg,file)
            else:
                refData = np.asarray(rawImg[0::2])
                data = np.asarray(rawImg[1::2])
            
            dataBin,dataStd,Eaxis = energyBin(data, self.energyF, self.params)
            refBin,refStd,Eaxis = energyBin(refData, self.energyF, self.params)
            
            #calculate absorbance and return data
            if self.dict['STATIC_YN']:
                Abs,stdDev = self.staticAbs(dataBin, dataStd, 
                                            refBin, refStd)
                
                yield XASdata(Abs, stdDev, Eaxis,
                              name = file.name)
            else:
                transAbs,transDev = self.transAbs(dataBin, dataStd, 
                                            refBin, refStd)
                
                yield XASdata(transAbs, transDev, Eaxis, 
                          self.Time,name = file.name)
        
def load_Generator(DIRECTORY):
    
    for file in os.scan(DIRECTORY):
        rawImage = spe.load(file.path)    
        yield rawImage

def save2D(XTAdata,Directory,FileNameList):
    """
    Utility to save all kinetic traces, either for a list of data or a single dataset.
    Will also save fits to the kinetic data, if available.

    Parameters
    ----------
    TAdata : Single TAdata or list of TAdata
        The data to save kinetic traces from.
    directory : str
        Where to save all the kinetic traces.
    basename : str
        Name to use for header and filename.
    wvlngth : int
        The wavelength to save kinetic traces from.

    Returns
    -------
    None.

    """
    if re.search('/\Z', Directory)==None:
        Directory +='/'
        
    if isinstance(XTAdata,list):
        #XTAdata is not an individual object, but a list of objects
        for data,file in zip(XTAdata,FileNameList):
            np.savetxt(Directory+file+".txt",data.trans2D,header = file)
            np.savetxt(Directory+file+"Std.txt",data.trans2Dstd,header = file)
            np.savetxt(Directory+file+"Time.txt",XTAdata.time,header = file)
            np.savetxt(Directory+file+"Energy.txt",XTAdata.energy,header = file)
    else:
        if isinstance(FileNameList,list):
            file = FileNameList[0]
        else:
            file = FileNameList
            
        np.savetxt(Directory+file+".txt",XTAdata.trans2D,header = file)
        np.savetxt(Directory+file+"Std.txt",XTAdata.trans2Dstd,header = file)
        np.savetxt(Directory+file+"Time.txt",XTAdata.time,header = file)
        np.savetxt(Directory+file+"Energy.txt",XTAdata.energy,header = file)

def saveKin(XTAdata,directory,basename,energy):
    """
    Utility to save all kinetic traces, either for a list of data or a single dataset.
    Will also save fits to the kinetic data, if available.

    Parameters
    ----------
    TAdata : Single TAdata or list of TAdata
        The data to save kinetic traces from.
    directory : str
        Where to save all the kinetic traces.
    basename : str
        Name to use for header and filename.
    energy : int
        The wavelength to save kinetic traces from.

    Returns
    -------
    None.

    """
    label = str(energy)
    if re.search('/\Z', directory)==None:
        directory +='/'
        
    if isinstance(XTAdata,list):
        #TAdata is not an individual object, but a list of objects
        for i,data in enumerate(XTAdata):
            hdr = "Time" +str(i)+", kin" + basename+str(i) + ", Std" + basename +str(i) +", fit" \
                + basename+str(i)
            np.savetxt(directory+basename+str(i)+".txt",
            np.stack([data.Time,data.Kinetic[label],data.KineticStd[label],
                      data.KineticFit[label]],axis=1),header = hdr)
    else:
        hdr = "Time" +", kin" + basename + ", Std" + basename 
        np.savetxt(directory+basename+".txt",
        np.stack([XTAdata.Time,XTAdata.Kinetic[label],
                  XTAdata.KineticStd[label]],axis=1),header = hdr)
        
def saveSpectral(XTAdata,directory,basename,time):
    """
    Utility to save all spectral traces, either for a list of data or a single dataset.

    Parameters
    ----------
    TAdata : Single TAdata or list of TAdata
        The data to save kinetic traces from.
    directory : str
        Where to save all the kinetic traces.
    basename : str
        Name to use for header and filename.
    time : int
        The time to save spectral traces from.

    Returns
    -------
    None.

    """
    
    label = str(time)
    
    if re.search('/\Z', directory)==None:
        directory +='/'
    
    if isinstance(XTAdata,list):
        #TAdata is not an individual object, but a list of objects
        for i,data in enumerate(XTAdata):
            hdr = "Energy" +str(i)+", T_slice" + basename+str(i) + ", Std" + basename +str(i)
            np.savetxt(directory+basename+str(i)+".txt",
            np.stack([data.energy,data.T_slice[label],
                      data.T_sliceStd[label]],axis=1),header = hdr)
    else:
        hdr = "Wavelength" +", T_slice" + basename + ", Std" + basename 
        np.savetxt(directory+basename+".txt",
        np.stack([XTAdata.energy,XTAdata.T_slice[label],
                  XTAdata.T_sliceStd[label]],axis=1),header = hdr)
    
def saveFits(FitList,directory,basename, components):
    """
    Utility to save parameters from fits from multiple fits, such as AIC and time constants/amplitudes.
    Amplitudes will be saved as normalized amplitudes, may not be as meaningful when the amplitudes have different signs.

    Parameters
    ----------
    FitList : List
        This should be a list of fit results, as outputted by manyFitKinetic, for example.
    directory : str
        Where to save the fits.
    basename : str
        Name to use for file name and headers.
    components : int
        How many components are in the fitted results.

    Returns
    -------
    None.

    """
    
    label = str(components)
    AIC = []
    BIC = []
    outArr1 =np.zeros((4*components))
    hdr = "AIC" +basename+", BIC" + basename
    
    if re.search('/\Z', directory)==None:
        directory +='/'
        
    for n in range(1,components+1):
        hdr+= ", C"+str(n) + basename + ", C" +str(n)+"Std_" + basename
        hdr+= ", A"+str(n) + basename + ", A" +str(n)+"Std_" + basename
        
    for i,data in enumerate(FitList):
        params = data.params.valuesdict()
        AIC.append(data.aic)
        BIC.append(data.bic)
        
        normA=1
        # for n in range(1,components+1): 
        #     label= "A"+str(n)
        #     normA += params[label]
            
        tmparr = []
        for n in range(1,components+1):                        
            label = "C"+str(n)
            tmparr.append(params[label])
            if data.params[label].stderr is None:
                tmparr.append(0)
            else:
                tmparr.append(data.params[label].stderr)
            label= "A"+str(n)
            tmparr.append(params[label])
            if data.params[label].stderr is None:
                tmparr.append(0)
            else:
                tmparr.append(data.params[label].stderr/abs(tmparr[-1]))
            tmparr[-2] /= normA
            
        outArr1 = np.vstack((outArr1,tmparr))
    
    outArr2 = np.stack([AIC,BIC])
    outArr = np.delete(outArr1,0,axis=0)
    outArr = np.concatenate([outArr2.transpose(),outArr],axis=1)
    np.savetxt(directory+basename+".txt",outArr,header = hdr)
 
    
def exportAllXTA(Directory,XTAlist,FileNameList):
    """
    Utility to save everything for a list of TAdata sets, including spectral traces,
    kinetic traces, time, wavelengths, 2D matrix, and standard deviation matrix.

    Parameters
    ----------
    Directory : str
        Where to save data.
    XTAlist : list
        List of XTAdata to export.
    FileNameList : List
        List of names as a string to save each dataset to.

    Returns
    -------
    None.

    """
    if re.search('/\Z', Directory)==None:
        Directory +='/'
       
    if isinstance(XTAlist,list):     
        for data,file in zip(XTAlist,FileNameList):
            for kin in data.Kinetic:
                saveKin(data,Directory,file+kin,kin)
                
            for spec in data.T_slice:
                saveSpectral(data,Directory,file+spec+"ps",spec)
                
            np.savetxt(Directory+file+".txt",data.Intensity,header = file)
            np.savetxt(Directory+file+"Std.txt",data.Std,header = file)
            np.savetxt(Directory+file+"Time.txt",data.Time,header = file)
            np.savetxt(Directory+file+"Wavelength.txt",data.Wavelength,header = file)
    else:
        if isinstance(FileNameList,list):
            file = FileNameList[0]
        else:
            file = FileNameList
            
        for kin in XTAlist.Kinetic:
            saveKin(XTAlist,Directory,file+kin,kin)
            
        for spec in XTAlist.T_slice:
            saveSpectral(XTAlist,Directory,file+spec+"ps",spec)
            
        np.savetxt(Directory+file+".txt",XTAlist.Intensity,header = file)
        np.savetxt(Directory+file+"Std.txt",XTAlist.Std,header = file)
        np.savetxt(Directory+file+"Time.txt",XTAlist.Time,header = file)
        np.savetxt(Directory+file+"Wavelength.txt",XTAlist.energy,header = file)
