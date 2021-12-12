"""
Asexual genetic algorithm for finding global maximum of a multiparameter function
"""
import numpy as np
from math import *

def evol(f, dim, n_data, lim):
	n = dim			  #number of dimensions
	N0 = 1000         #population
	N1 = 40    		  #parents
	N2 = 24    		  #sons
	p = 0.55     	  #convergency factor

	#initial random positions in n-dimension parametric logspace for N0 individuals 
	pop = np.zeros((n+1, N0))
	for i in range(n):
		pop[i,:] = np.random.uniform(lim[i,0], lim[i,1], N0)
	#initial fitness
	for j in range(N0):
		pop[n,j] = f(pop[0:n,j])
	#sort array by last row (fittest in last column)
	pop = pop[:,pop[n,:].argsort()]

	run = 0
	run_max = 10
	gen_max = 30
	chi_red = 10
	chi_max = 1.6

	while(chi_red > chi_max and run < run_max):
		run += 1
		gen = 0
		print(run)
		print(chi_red)
		while(chi_red > chi_max and gen < gen_max):
			gen += 1
			#s=N2xN1=N0-N1 sons from N1 fittest parents
			s = N0-N1
			for i in range(n):
				l0, l1 = lim[i,0], lim[i,1]
				l = (l1-l0)*pow(p,gen)  #size of the sampling boxes
				for j in range(N1):
					pj = pop[0:n, N0-1-j]
					pop[i,j*N2:(j+1)*N2] = np.random.uniform(max(l0, pj[i] - l), min(l1, pj[i] + l), N2)
			#fitness of sons and sorting of the array
			for j in range(s):
				pop[n,j] = f(pop[0:n,j])
			pop = pop[:,pop[n,:].argsort()]
			chi_red = sqrt(-pop[n, N0-1]/(n_data - n - 1.))
	par = np.power(10,pop[0:n,N0-1])
	
	return(chi_red, par, run)