# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:24:05 2024

@author: Samth
"""

import numpy as np

def normAbs(XUVlist,ROI):
    
    for data in XUVlist:
        Eindex = [idx for idx,x in enumerate(data.Eaxis)
                  if (x>ROI[0] and x<ROI[1])]
        normFactor = np.max(data[Eindex]) 
        
        data.Std /= abs(normFactor)
        data.Abs /= normFactor
        