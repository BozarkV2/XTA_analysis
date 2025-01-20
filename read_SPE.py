# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:30:43 2024

@author: Samth
"""
#This code is taken from the scipy cookbook: https://scipy-cookbook.readthedocs.io/items/Reading_SPE_files.html?highlight=SPE

#!python numbers=disabled
# read_spe.py
import numpy as N
import time

class File(object):

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = N.int64(self.read_at(42, 1, N.int16)[0])
        self._ydim = N.int64(self.read_at(656, 1, N.int16)[0])
        self._zdim = N.int64(self.read_at(1446, 1, N.int32)[0])

    def _load_date_time(self):
        rawdate = self.read_at(20, 9, N.int8)
        rawtime = self.read_at(172, 6, N.int8)
        strdate = ''
        for ch in rawdate :
            strdate += chr(ch)
        for ch in rawtime:
            strdate += chr(ch)
        self._date_time = time.strptime(strdate,"%d%b%Y%H%M%S")

    def get_size(self):
        return (self._xdim, self._ydim, self._zdim)

    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return N.fromfile(self._fid, ntype, size)

    def load_img(self):
        #this originally assumed an input of unsigned 16 bits, but the image is actually
        #single float CD 01/29/2024
        img = self.read_at(4100, self._xdim * self._ydim * self._zdim, N.single) 
        return img.reshape((self._zdim, self._ydim, self._xdim))

    def close(self):
        self._fid.close()

def load(fname):
    fid = File(fname)
    img = fid.load_img()
    fid.close()
    return img

class FileLarge(object):

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = N.int64(self.read_at(42, 1, N.int16)[0])
        self._ydim = N.int64(self.read_at(656, 1, N.int16)[0])
        self._zdim = N.int64(self.read_at(1446, 1, N.int32)[0])

    def _load_date_time(self):
        rawdate = self.read_at(20, 9, N.int8)
        rawtime = self.read_at(172, 6, N.int8)
        strdate = ''
        for ch in rawdate :
            strdate += chr(ch)
        for ch in rawtime:
            strdate += chr(ch)
        self._date_time = time.strptime(strdate,"%d%b%Y%H%M%S")

    def get_size(self):
        return (self._xdim, self._ydim, self._zdim)

    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return N.fromfile(self._fid, ntype, size)

    def load_img(self, zdim):
        #this originally assumed an input of unsigned 16 bits, but the image is actually
        #single float CD 01/29/2024
        img_size = self._xdim*self._ydim
        
        #read_at needs an absolute byte position. 
        #the factor 4 comes from the image being single float
        readin = 4100+img_size*4*zdim 
        img = self.read_at(readin, img_size, N.single)        
        
        return img.reshape((self._ydim, self._xdim))

    def close(self):
        self._fid.close()

def loadGen(fname):
    fid = FileLarge(fname)
    
    for z in range(fid._zdim):
        # print(z)
        yield fid.load_img(z)
    
    fid.close()

if __name__ == "__main__":
    import sys
    img = load(sys.argv[-1])