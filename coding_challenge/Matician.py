#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: stephenchen
"""

import numpy as np
import time
from scipy.ndimage import convolve1d

np.random.seed(5)


# # Step 1
# # Read rows and columns (cols) as integer arguments from the command line.
# print('Input an Integer as row number:')
# rows = int(input()) 
# print('Input an Integer as column number:')
# cols = int(input()) 
# print("Your input is {0:d} rows and {1:d} cols.".format(rows, cols)) 

rows = 8
cols = 8

# Step 2
# Create an unsigned char matrix M of size [rows x cols] 
M = np.zeros(shape=(rows,cols), dtype=np.uint8)
# print(M.itemsize) # verify item type
# print(M)

# Step 3
# Fill M with randomly selected non-negative integers.
M = (np.random.randint(255, size=(rows,cols))).astype(np.uint8)
# print(M.itemsize) # verify item type
print(M)
print(M.shape)

# Step 4
# Apply the filter K=[-1, 0, 1] along the rows axis, then the cols axis (i.e. 
# convolve the matrix M with K along the vertical & horizontal axis respectively).
k = np.array([-1, 0, 1]).reshape([3,1])
# print(k.shape)



start_time = time.time()
M = M.astype(np.int)
Dy = (convolve1d(M, weights=[-1, 0, 1], axis = 0))
Dx = (convolve1d(M, weights=[-1, 0, 1], axis = -1))
time = time.time() - start_time
print("Total time taken by the machine in computing Dx and Dy matrices: {:f} s".format(time) )

print(Dy)
print(Dx)


# Step 7
# Compute the min and max values for both Dx & Dy matrices individually 
# Print the computed min & max values.
maxV = max(map(max, Dy)) 
minV = min(map(min, Dy))
print(maxV)
print(minV)




