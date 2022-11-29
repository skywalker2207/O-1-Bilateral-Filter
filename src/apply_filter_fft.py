import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
from math import exp,sqrt
import time

def applyfilter_fft ( im , kr) : 
    # kernel's fft is scaled to fit image's fft
    m , n = np.shape ( im ) 
    mk, nk = np.shape ( kr ) 

    hpad = (m-mk)//2 
    wpad = (n-nk)//2

    # print ( "hpad" , hpad , "wpad" , wpad ) 

    kr = np.pad(kr, ((hpad , (m-mk) - hpad), (wpad , ( n - nk ) - wpad  )), 'constant', constant_values=(0))
    kr = np.fft.fftshift ( kr )
    # kr = np.pad(kr, ((m-mk,  0 ), (0 ,  n - nk  )), 'constant', constant_values=(0))
    
    appliedF = np.fft.fft2 ( kr  )   *  np.fft.fft2 ( im) 

    return  np.fft.ifft2 ( appliedF ) 