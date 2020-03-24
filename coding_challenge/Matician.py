#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: stephenchen
"""

import numpy as np
import time
from scipy.ndimage import convolve1d
np.random.seed(5)

class MyMatrix():
    
    def __init__(self):
        self.rows = 0
        self.cols = 0
    
    def setDimension(self):
        """
        Read rows and columns (cols) as integer arguments from the command line.
        """
        print('Input numbers of rows and cols in the format of:  (rows, cols):')
        r_c_str = (input()) 
        for c in "(),":
            r_c_str = r_c_str.replace(c, " ")
        r_c_str = r_c_str.split()
        try:
            self.rows = int(r_c_str[0])
            self.cols = int(r_c_str[1])
        except ValueError:
            print("Error! Please input positive integer values for rows and cols.")
            raise ValueError ("Error! Please input positive integer values for rows and cols.")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError ("Error! Number of rows and cols must be positive")
            

       
    def createMatrix(self):
        """
        Create an unsigned char matrix M of size [rows x cols]   
        """
        if self.rows <= 0 or self.cols <= 0:
            raise Exception('Please set correct number of rows and cols')
        self.M = np.zeros(shape=(self.rows,self.cols), dtype=np.uint8)
        print('Create {} X {} unsigned char Matrix'.format(self.rows, self.cols))
    
  
    def fillM(self, low = 0, high = 256):
        """
        Fill M with randomly selected non-negative integers.   
        :type low: int Lowest (signed) integer to be drawn from the distribution
        :type high: int one above the largest (signed) integer to be drawn from the distribution
        """
        self.M = (np.random.randint(low, high, size=(self.rows,self.cols))).astype(np.uint8)
        print('Fill M with randomly selected non-negative integers.')
        # print(self.M.itemsize) # verify item type
        # print(self.M)
        # print(self.M.shape)
        
    def convolve1d(self, k):
        """
        pply the filter K=[-1, 0, 1] along the rows axis, then the cols axis (i.e. 
        convolve the matrix M with K along the vertical & horizontal axis respectively).
        :type k: List[int] filter 
        :rtype totalTime: float total time taken by the machine in computing Dx and Dy matrices
        """
        print("Apply the filter K=[-1, 0, 1] along the rows axis, then the cols axis.")
        start_time = time.time()
        temp = self.M.astype(np.int)
        self.Dy = (convolve1d(temp, weights=[-1, 0, 1], axis = 0))
        self.Dx = (convolve1d(temp, weights=[-1, 0, 1], axis = -1))
        totalTime = time.time() - start_time
        print("Store Dy and Dx as instance variables.")
        return totalTime
    
    def minmax(self):
        """
        Compute the min and max values for both Dx & Dy matrices individually
        :rtype: int min and max values for both Dx & Dy
        """
        maxDy = max(map(max, self.Dy)) 
        minDy = min(map(min, self.Dy))
        maxDx = max(map(max, self.Dx)) 
        minDx = min(map(min, self.Dx))
        return maxDy, minDy, maxDx, minDx
        
        
    



matrix = MyMatrix()
matrix.setDimension()
matrix.createMatrix()
matrix.fillM()
k = [-1,0,1]
ttl_time = matrix.convolve1d(k)
print('{} seconds was taken by the machine in computing Dx and Dy matrices'.format(ttl_time))
maxDy, minDy, maxDx, minDx = matrix.minmax()
print('min(Dy) = {}, max(Dy) = {}, min(Dx) = {}, max(Dx) = {}'.format(minDy, maxDy, minDx, maxDx))

