# XTA_analysis
Scripts and functions for analyzing X-ray transient absorption spectroscopy, or associated spectroscopy, primarily from a High Harmonic source.

These scripts can be run in their most basic form from the XUVmain function. Give the file path to the init function, which returns a dictionary with data processing options. Then give this dictionary to mainStatic/mainTransient/mainXMCD and follow the prompts. 

The init function has a description of each parameter, and the execution of each parameter can be found in the XUVclass function. 

XASdata is where the raw data is written to, as well as the data after each processing step. These data can be accessed through the returned class. There are also options to give the XASdata another XASdata class as a template. In this case, the energy axis and processing is propagated from the template to the new class.

XTAdata is similar to XASdata, but it includes functions for extracting kinetic traces with "KineticTrace", spectral traces with "SpectralTrace", as well as subtracting a background.

Plotting can be done with Xplt (abbreviated XUVplotter):
PlotXAS for ground state spectra
PlotXAShalf for plotting ground state spectra based on which data has the best statistics.
PlotStats for plotting relevant statistics of ground state spectra. 
PlotXTAdata for plotting transient spectra
plotTimeMap for plotting the scan vs. energy map for each time point (before averaging)
plotHarms plots the averaged data and reference harmonic spectra
PlotKinetic for plotting kinetic slices
PlotSpectral for plotting spectral slices

If it's desired to run the scripts without the main functions, a basic implementation is shown below for a case where a set of files includes a reference spectrum.

import XUVmain
dir = 'My\\Example\\Directory"
start = init(dir)
XUVlist = importDir(start)
dataBack = XUVlist.pop()
refBack = XUVlist.pop()
XAS = XASdata(XUVlist, dataBck = dataBack, refBck = refBack, name='Example', options=start)
