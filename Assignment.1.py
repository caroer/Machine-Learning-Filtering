# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:54:33 2020

@author: mnbe
"""
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from numpy import asarray
import numpy as np
import pywt
import pywt.data
from collections import OrderedDict
import timeit

""""""""""""""" HAAR transform """""""""""""""

def Transform(size, inputMatrix):
    I=1
    J=2

    temp = np.zeros((size, size))
    
    for i in range(0, size):       # Row wise
        for K in range(0,int(size/2)):
            a1=(inputMatrix[i, K*J]+inputMatrix[i, K*J+I])/2
            c2=(inputMatrix[i, K*J]-inputMatrix[i, K*J+I])/2
            temp[i, K*J]=a1
            temp[i, K*J+I]=c2
         
    for i in range(0, size):       # Column wise
        for K in range(0,int(size/2)):
            a1=(temp[K*J, i]+temp[K*J+I, i])/2
            c2=(temp[K*J, i]-temp[K*J+I, i])/2
            temp[K*J, i]=a1
            temp[K*J+I, i]=c2
             
    return temp


def Inverse(Size, unOrganized):
    I=1
    J=2
    temp = np.zeros((Size, Size))
    
    for i in range(0, Size):       # Column wise
        for K in range(0,int(Size/2)):
            a1=(unOrganized[K*J, i]+unOrganized[K*J+I, i])
            c2=(unOrganized[K*J, i]-unOrganized[K*J+I, i])
            temp[K*J, i]=a1
            temp[K*J+I, i]=c2
             
    for i in range(0, Size):       # Row wise
        for K in range(0,int(Size/2)):
            a1=(temp[i, K*J]+temp[i, K*J+I])/2
            c2=(temp[i, K*J]-temp[i, K*J+I])/2
            temp[i, K*J]=a1
            temp[i, K*J+I]=c2
      
    return temp

def Rearrange_values(size, input_matrix):

    averages = CreateSubMatrix(size, input_matrix, 0, 0)
    horisontal = CreateSubMatrix(size, input_matrix, 0, 1)
    vertical = CreateSubMatrix(size, input_matrix, 1, 0)
    diagonal = CreateSubMatrix(size, input_matrix, 1, 1)

    return AssembleMatrix(size, averages, horisontal, vertical, diagonal)


def CreateSubMatrix(size, inputMatrix, rowOffset, colOffset):
    subMatrix = np.zeros((size, size))
    
    for i in range(0, size):
        for j in range(0, size):
            subMatrix[i, j] = inputMatrix[2*i + rowOffset, 2*j + colOffset]
            
    return subMatrix
 
    
def AssembleMatrix(size, averages, horisontal, vertical, diagonal):
    total = np.zeros((size*2,size*2))
    
    for i in range(0, size):
        for j in range(0, size):
            total[i, j] = averages[i, j]
            
    for i in range(0, size):
        for j in range(size, size*2):
            total[i, j] = horisontal[i, j-size]
            
    for i in range(size, size*2):
        for j in range(0, size):
            total[i, j] = vertical[i-size, j]
            
    for i in range(size, size*2):
        for j in range(size, size*2):
            total[i, j] = diagonal[i-size, j-size]
            
    return total


def main():
    
    img2 = Image.open('data_uppgift1.jpg').convert('LA')
    a = asarray(img2)  #nu är bilden som en array och kan transformeras mha HAAR
    #b=a[0:4096,0:4096,0]
    n_pix = 4096
    b=a[0:n_pix,0:n_pix,0] 
    plt.figure(figsize = (50,15))
    plt.imshow(b, interpolation="nearest", cmap=plt.cm.gray)
    #plt.show()
    plt.savefig('Original_pic.png')
    n_sweeps = 5
    
    s=b.copy()
    n=int(np.log2(len(s[0])))      # n = 7
    size=int(np.power(2,n))
    unOrganized = []
    
    for L in range(1,n_sweeps + 1):      # Number of sweeps        
        unOrganized.append(Transform(size, s))  # 128x128, 64x64
        
        size=int(size/2)    # 64, 32
        organized = Rearrange_values(size,unOrganized[-1])  # 128x128, 64x64
        
        s = organized[0:size, 0:size]   # 64x64, 32x32
        
        figName = 'Transform_sweep_'+str(L)+'.png'
        plt.imshow(s, cmap='gray')
        #plt.show()
        plt.savefig(figName)
        print('Transform sweep ' + str(L) + '/' + str(n_sweeps) + ' completed!')
        
      
    for i in range(1,n_sweeps + 1):
        #r = Reconstruction(2*size, s)
        
        m = unOrganized[-1]     # 64x64
        size = size*2
        inverted = Inverse(size, m)    
        figName = 'Inverse_sweep_'+str(i)+'.png'
        plt.imshow(inverted, cmap='gray')
        #plt.show()
        plt.savefig(figName)
        print('Inverse transform sweep ' + str(i) + '/' + str(n_sweeps) + ' completed!')
        unOrganized.pop()

timer = timeit.timeit(stmt = lambda: main(), number = 1)