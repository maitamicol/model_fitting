"""
Model fitting example
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from math import *
from aga import evol
from df import func_df

def f1(x,p):
	w = x * 1e9 / nu0
	return p[0] * np.power(w,p[1])

def f2(x,p):
	w = x * 1e9 / nu0
	return p[2] * (1. - np.exp(-p[3] * np.power(w,p[4]))) * np.power(w,3) / (np.exp(h * nu0 * w / (k * p[5])) - 1.)

def f3(x,p):
	w = x * 1e9 / nu0
	return p[6] * (1. - np.exp(-p[7] * np.power(w,p[8]))) * np.power(w,3) / (np.exp(h * nu0 * w / (k * p[9])) - 1.)

def f(x,p):
	return f1(x,p), f2(x,p), f3(x,p)

def fn(x,p):
	df_1, df_2, df_3 = func_df(np.log10(x), p)
	return 10**f1n.predict(df_1), 10**f2n.predict(df_2), 10**f3n.predict(df_3)

def chi2(p):
	f1n, f2n, f3n = fn(data[:,0], p)
	f = f1n + f2n + f3n 
	a = np.power(((data[:,1] - f) / std), 2)
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

#NN interpolation for each function
f1n = pickle.load(open('MLmodel_f1.sav', 'rb'))
f2n = pickle.load(open('MLmodel_f2.sav', 'rb'))
f3n = pickle.load(open('MLmodel_f3.sav', 'rb'))

n = 10
par = np.array([1.07770827e+00, 4.09242919e-02, 1.60082864e-01, 3.06239230e-01, 2.56571519e-01, 8.91054667e+01, 4.77458373e-05, 2.02035260e-01, 1.32938992e-02, 3.23494150e+02])

#chi_red, par, run = evol(chi2, n, n_data, lim)
#print("Chi_red:", chi_red)
#print("Run:", run)
#print("Parameters:\n", par)

#plot
fig = plt.figure()
fig.set_size_inches(4.5, 4)
ax = fig.add_subplot(111)
ax.set(title='Spectral Energy Distribution (SED)', xlim=(1e0,3*1e5), ylim=(0.5,1e5), xscale='log', yscale='log', xlabel='Freq. (GHz)', ylabel='Flux (mJy)')
ax.errorbar(data[:,0], data[:,1], yerr=std, fmt='o', markersize=1.5, color='black', capsize=1.2, elinewidth=0.5)

x = np.logspace(0, 6, 500)
f1n, f2n, f3n = fn(x, np.log10(par))
f1, f2, f3 = f(x,par)
ax.plot(x, f1, color='red', linewidth=.5)
ax.plot(x, f2, color='blue', linewidth=.5)
ax.plot(x, f3, color='green', linewidth=.5)
ax.plot(x, f1+f2+f3, color='black', linewidth=.5)
#ax.plot(x, f1n, color='red', linestyle='--', linewidth=.5)
#ax.plot(x, f2n, color='blue', linestyle='-.', linewidth=.5)
#ax.plot(x, f3n, color='green', linestyle=':', linewidth=.5)
#ax.plot(x, f1n+f2n+f3n, color='black', linewidth=.5)
plt.show()

