# -*- coding: utf-8 -*-
"""
Example 6.1 2D MLP classifier for the Iris dataset

Developed for Machine Learning for Mechanical Engineers at the University of
South Carolina

@author: austin_downey
"""

import IPython as IP
IP.get_ipython().magic('reset -sf')


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


plt.close('all')

data = np.loadtxt('accel_1.csv',skiprows=2, delimiter=',')
tt = data[:,0]
yy = data[:,1]

plt.figure()
plt.plot(tt,yy)
plt.xlabel('time (s)')
plt.ylabel('acceleration ($m/s$)')



plt.figure()


# Number of sample points
N = yy.shape[0]
# sample spacing

T = tt[1]-tt[0]


yf = fft(yy)
xf = fftfreq(N, T)[:N//2]

yyy = 2.0/N * np.abs(yf[0:N//2])

plt.plot(xf, yyy,'.-')
plt.xlabel('frequency (Hz)')
plt.ylabel('Power')
plt.xlim([0,5000])
plt.grid()
plt.tight_layout()


           








































