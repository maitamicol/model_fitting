"""
Model fitting example
"""
import numpy as np
import matplotlib.pyplot as plt
from math import *
from aga import evol

def f1(w,p):
	return p[0] * np.power(w,p[1])

def f2(w,p):
	return p[2] * (1. - np.exp(-p[3] * np.power(w,p[4]))) * np.power(w,3) / (np.exp(h * nu0 * w / (k * p[5])) - 1.)

def f3(w,p):
	return p[6] * (1. - np.exp(-p[7] * np.power(w,p[8]))) * np.power(w,3) / (np.exp(h * nu0 * w / (k * p[9])) - 1.)

def f(w,p):
	return f1(w,p) + f2(w,p) + f3(w,p)

def chi2(p):
	a = np.power(((data[:,1] - f((data[:,0] * 1e9 / nu0),p)) / std), 2)
	return -np.sum(a)

#constants in SI units
h = 6.6260e-34	 #Planck's constant
k = 1.3806e-23   #Boltzmann's constant
c = 299792458    #speed of light
nu0 = c / 7e-3   #characteristic frequency

#data
data = np.genfromtxt('data.txt', dtype=float)				 #data points (n_data x (freq.(GHz), flux(mJy)))
error = np.genfromtxt('error.txt', dtype=float)				 #n_data x (upper limit, lower limit)
std = (error[:,0]-error[:,1])/2                              #standard deviation
lim = np.array([[-2., 1.],[-3., 1.],[-2., 1.],[-2., 1.],     #intervals in parametric logspace
	[-1., 1.],[1., 3.],[-5., -2.],[-2., 1.],[-2., 1.],[1., 3.]])
n_data = len(data[:,0])

n = 10				                                         #number of parameters
chi_red, par, run = evol(chi2, n, n_data, lim)
print("Chi_red:", chi_red)
print("Run:", run)
print("Parameters:\n", par)

#plot
fig = plt.figure()
fig.set_size_inches(4.5, 4)
ax = fig.add_subplot(111)
ax.set(title='Spectral Energy Distribution (SED)', xlim=(1e0,3*1e5), ylim=(0.5,1e5), xscale='log', yscale='log', xlabel='Freq. (GHz)', ylabel='Flux (mJy)')
ax.errorbar(data[:,0], data[:,1], yerr=std, fmt='o', markersize=1.5, color='black', capsize=1.2, elinewidth=0.5)

x = np.logspace(0, 6, 500)
w = x * 1e9 / nu0
f1 = f1(w,par)
f2 = f2(w,par)
f3 = f3(w,par)
ax.plot(x, f1, color='red', linestyle='--', linewidth=.5)
ax.plot(x, f2, color='blue', linestyle='-.', linewidth=.5)
ax.plot(x, f3, color='green', linestyle=':', linewidth=.5)
ax.plot(x, f1+f2+f3, color='black', linewidth=.5)
plt.show()