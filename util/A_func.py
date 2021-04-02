
import numpy as np
from scipy import special


def func_A(x):
	modified_result_of_division = np.divide(special.i1e(x),special.i0e(x))
	return modified_result_of_division

def func_A_Deriv(kappa): #Same as f' referenced in Song
    aTemp =  func_A(kappa)
    return (1-aTemp*(aTemp+1/kappa))

def kEst(outputVal): #determine the estimation step thru slope (Originated with Banerjee, used by Sra, Song)
	return (outputVal*(2-np.power(outputVal, 2))/(1-np.power(outputVal, 2)))

def inv_func_A_sra(x, iterations = 5):  #Reference Sra Paper
	kappa = kEst(x)
	for i in range(iterations):
		kappa = kappa-(func_A(kappa)-x)/func_A_Deriv(kappa) #forgot to subtract by outputVal
	return kappa

def mean_of_cct(kappa_1, kappa_2):
	return inv_func_A_sra(func_A(kappa_1)*func_A(kappa_2))

