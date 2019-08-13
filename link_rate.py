#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:37:08 2019

@author: daniel
"""

import numpy as np
import math

def dBm2mW(dbm):
    return np.power(10, dbm/10)

def link_rate(d):
    
    # model parameters
    L0 = -50.6;       # power at 1m (dBm)
    n = 2.75;
    sigma_F2 = 40.5;  # variance of noise due to fading, ~N(0,sigma_F^2)
    N0 = -70          # noise at receiver (dBm)

    # link rate
    PRdBm = L0 - 10*n*np.log10(d)                   # receiver power (dBm)
    PRmW = dBm2mW(PRdBm)                            # receiver power (mW)
    PN0mW = dBm2mW(N0)                              # noise at receiver (nW)
    rate = 1 - np.special.erfc(np.sqrt(PRmW/PN0mW)) # normalized rate
    
    # link variance
    var = np.power(math.log(10) / (10*math.sqrt(PN0mW*math.pi)) * np.exp(-PRmW/PN0mW) * np.power(10,PRdBm/20), 2*sigma_F2)
    
    return [rate, var]