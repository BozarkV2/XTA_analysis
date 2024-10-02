# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:27:56 2024

@author: Samth
"""

from scipy.interpolate import interp1d
import numpy as np

def alignTime(XUVlist):
    """
    Linearly interpolates the absorbance values for a set of transient 
    XUV data onto a common time axis

    Parameters
    ----------
    XUVlist : list of XUV data structures, which include time and abs values.

    Returns
    -------
    None.

    """
    timeList = [data.Time for data in XUVlist]
    timeArr = np.round(np.mean(timeList),decimals=2)
    
    for data in XUVlist:
        timeInterp = interp1d(data.Time,data.Abs,
                                   kind='linear',fill_value='extrapolate')
        data.Abs = timeInterp(timeArr)
        
        