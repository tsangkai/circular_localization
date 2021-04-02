

import numpy as np
import math


class LieGroup:
	def __init__(self, _theta, _x, _y):
		self.lg_matrix = np.array([
			[math.cos(_theta), -math.sin(_theta), _x],
			[math.sin(_theta), math.cos(_theta), _y],
			[0, 0, 1]])

	def __mul__(self, other):
		_resulting_matrix = self.lg_matrix @ other.lg_matrix
		return LieGroup(math.atan2(_resulting_matrix[1,0], _resulting_matrix[0,0]), _resulting_matrix[0,2], _resulting_matrix[1,2])

	def __str__(self): 
		return str(self.lg_matrix)

	def toEuclidean(self):
		return [math.atan2(self.lg_matrix[1,0], self.lg_matrix[0,0]), self.lg_matrix[0,2], self.lg_matrix[1,2]]


def Phi(_vector):
	output_matrix = np.zeros((3,3))
	ad_rep = adjoint_rep(_vector)
	ad_rep_prod = np.identity(3)

	for i in range(20):
		output_matrix = output_matrix + (math.pow(-1,i)/math.factorial(i)) * ad_rep_prod
		ad_rep_prod = ad_rep_prod @ ad_rep

	return output_matrix


def Adjoint_rep(input_lg):
	lg_matrix = input_lg.lg_matrix
	_theta = math.atan2(lg_matrix[1,0], lg_matrix[0,0])
	_x = lg_matrix[1,2]
	_y = -lg_matrix[0,2]

	return LieGroup(_theta, _x, _y)

def adjoint_rep(_vector):            # input is in R^3
	return np.array([[0, 0, 0],
		[0, -_vector[2,0], -_vector[0,0]],
		[-_vector[1,0], -_vector[0,0], 0]])


def exp_SE2(_vector):                # input is in R^3
	_theta = _vector[0,0]
	_x = _vector[1,0]
	_y = _vector[2,0]

	_temp = V(_theta) @ np.array([[_x], [_y]])

	return LieGroup(_theta, _temp[0,0], _temp[1,0])


def V(_theta):
	return (1.0/_theta)*np.array([[math.sin(_theta), -1+math.cos(_theta)],
		                           [1-math.cos(_theta), math.sin(_theta)]])

