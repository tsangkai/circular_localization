

import numpy as np
import yaml

from scipy import special
import math
import cmath

from LieGroup import *


### import parameters

with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)

landmark = config['landmark_position']


class GaussianEstimate:
	def __init__(self, _mean=0, _std=0.01):
		self.mean = _mean
		self.cov = math.pow(_std, 2)

	def time_update(self, _input_mean=0, _input_std=1):
		self.mean += _input_mean
		self.cov += math.pow(_input_std, 2)


	def observation_update(self, _input_mean=0, _input_std=1):
		_input_cov = _input_std ** 2

		self.mean = self.mean + (self.cov/(self.cov+_input_cov))*(_input_mean-self.mean)
		self.cov = 1.0 / (1.0/self.cov + 1.0/_input_cov)


class CircularEstimate:
	def __init__(self, _phase=0, _concentration=0):
		self.phase = _phase
		self.concentration = _concentration

	def time_update(self, input_phase, input_cct):
		self.phase = (self.phase + input_phase) % (2*math.pi)
		self.concentration = mean_of_cct(self.concentration, input_cct)

	def observation_update(self, input_phase, input_cct):
		phase_tentative = cmath.phase(self.concentration*cmath.rect(1, self.phase)+input_cct*cmath.rect(1, input_phase))
		self.concentration = self.concentration*math.cos(phase_tentative-self.phase) + input_cct*math.cos(phase_tentative-input_phase)
		self.phase = phase_tentative % (2*math.pi)

	def force_observation_update(self, input_phase, input_cct):
		self.concentration = input_cct
		self.phase = input_phase



###############
class GaussianSpatialState:
	def __init__(self, _mean=np.matrix([[0.0], [0.0], [0.0]]), _cov=np.matrix([[0.01,0,0], [0,0.01,0], [0,0,0.01]])):
		self.mean = _mean
		self.cov = _cov


	def time_update(self, input_w, input_w_std, input_v, input_v_std, dt):
		input_w_cov = math.pow(input_w_std, 2)
		input_v_cov = math.pow(input_v_std, 2)

		self.mean[0,0] = self.mean[0,0] + input_w * dt
		self.mean[1,0] += input_v * dt * math.cos(self.mean[0,0])
		self.mean[2,0] += input_v * dt * math.sin(self.mean[0,0])

		F_state_2_state  = np.matrix([[1, 0, 0],
			                        [-input_v*dt*math.sin(self.mean[0,0]), 1, 0],
			                        [input_v*dt*math.cos(self.mean[0,0]),  0, 1]])

		F_input_2_state = dt * np.matrix([[1, 0],
			                        [0, math.cos(self.mean[0,0])],
			                        [0,  math.sin(self.mean[0,0])]])

		Q = np.matrix([[input_w_cov, 0],[0, input_v_cov]])

		self.cov = F_state_2_state * self.cov * F_state_2_state.getT() + F_input_2_state * Q * F_input_2_state.getT()


	# relative observation model: more linear, for debugging
	def rel_observation_update(self, observation, d_std):
		R = np.matrix([[d_std**2, 0], [0, d_std**2]])

		H = np.matrix([[0, -1, 0], [0, 0, -1]])
		S = H * self.cov * H.getT() + R

		innovation = observation - (np.matrix(landmark).getT() - self.mean[1:3,0])

		self.mean = self.mean + self.cov * H.getT() * S.getI() * innovation
		self.cov = self.cov - self.cov * H.getT() * S.getI() * H * self.cov



	def bd_observation_update(self, bearing, bearing_std, distance, distance_std):

		distance_cov = distance_std ** 2
		bearing_cov = bearing_std ** 2
		
		dx = landmark[0]-self.mean[1,0]
		dy = landmark[1]-self.mean[2,0]

		est_bearing = (math.atan2(dy, dx) - self.mean[0,0]) % (2*math.pi)
		est_distance = math.sqrt(math.pow(dx,2) + math.pow(dy,2))

		observ = np.matrix([[bearing], [distance]])
		est_observ = np.matrix([[est_bearing], [est_distance]])

		H = np.matrix([[-1, dy/(math.pow(dx,2) + math.pow(dy,2)), -dx/(math.pow(dx,2) + math.pow(dy,2))], 
			[0, -dx/est_distance, -dy/est_distance]])

		S = np.matrix([[bearing_cov, 0],[0, distance_cov]]) + H * self.cov * H.getT()

		innovation = observ - est_observ
		innovation[0] = ((innovation[0] + math.pi) % (2*math.pi)) - math.pi

		self.mean = self.mean + self.cov * H.getT() * S.getI() * innovation
		self.cov = self.cov - self.cov * H.getT() * S.getI() * H * self.cov

	def read_estimation(self):
		return [self.mean[0,0], self.mean[1,0], self.mean[2,0]]
	


###	

class HybridSpatialState:
	def __init__(self, _phase=0, _concentration=0.0001, _x=0, _x_std=0.1, _y=0, _y_std=0.1):
		self.theta_est = CircularEstimate(_phase, _concentration)

		self.x_est = GaussianEstimate(_x, _x_std)
		self.y_est = GaussianEstimate(_y, _y_std)

	def time_update(self, input_w, input_w_std, input_v, input_v_std, dt):
		input_w_cov = math.pow(input_w_std, 2)
		input_v_cov = math.pow(input_v_std, 2)

		# estimate update
		self.theta_est.time_update(input_w*dt, 1/(input_w_cov*dt*dt))

		_input_std = math.sqrt(input_v_cov + math.pow(input_v, 2))
		self.x_est.time_update(input_v*dt*func_A(self.theta_est.concentration)*math.cos(self.theta_est.phase), _input_std)
		self.y_est.time_update(input_v*dt*func_A(self.theta_est.concentration)*math.sin(self.theta_est.phase), _input_std)


	def direct_observation_update(self, obs_theta, obs_theta_cct, obs_x, obs_y, obs_std):
		self.theta_est.observation_update(obs_theta, obs_theta_cct)

		self.x_est.observation_update(obs_x, obs_std)
		self.y_est.observation_update(obs_y, obs_std)


	def bd_observation_update(self, bearing, bearing_cct, distance, distance_std):
		distance_cov = distance_std ** 2

		dx = landmark[0]-self.x_est.mean
		dy = landmark[1]-self.y_est.mean
		est_distance = math.sqrt(dx**2 + dy**2)
		landmark_orientation = math.atan2(dy, dx)


		# intermediate terms
		_equ_phase = landmark_orientation - bearing
		# print([est_distance*distance/(2*self.x_est.cov), bearing_cct])
		_equ_phase_cct = mean_of_cct(est_distance*distance/(2*self.x_est.cov), bearing_cct)



		# CHECK THE FOLLOWING TWO LINES
		_equ_x = landmark[0] - distance * func_A(bearing_cct) * func_A(self.theta_est.concentration) * math.cos(self.theta_est.phase + bearing)
		_equ_y = landmark[1] - distance * func_A(bearing_cct) * func_A(self.theta_est.concentration) * math.sin(self.theta_est.phase + bearing)

		_equ_std = math.sqrt(distance_cov + distance**2)

		# estimate update
		self.theta_est.force_observation_update(_equ_phase, _equ_phase_cct)
		self.x_est.observation_update(_equ_x, _equ_std)
		self.y_est.observation_update(_equ_y, _equ_std)

	def read_estimation(self):
		return [self.theta_est.phase, self.x_est.mean, self.y_est.mean]




#########


class GridModule:
	def __init__(self, _scale, _phase=0, _concentration=500):
		self.scale = _scale
		self.estimate = CircularEstimate(_phase, _concentration)

	def time_update(self, input_distance, input_distance_std):
		_phase_cct = math.pow(self.scale/ 2*math.pi*input_distance_std, 2)
		self.estimate.time_update(2*math.pi*input_distance/self.scale, _phase_cct)

	def observation_update(self, input_distance, input_distance_std):
		# _phase_cct = min(math.pow(self.scale / 2*math.pi*input_distance_std, 2), 600)
		_phase_cct = math.pow(self.scale / 2*math.pi*input_distance_std, 2)

		self.estimate.observation_update(2*math.pi*input_distance/self.scale, _phase_cct)



class CircularSpatialState:
	def __init__(self, _phase=0, _concentration=0.0001):
		# estimate


		self.theta_est = CircularEstimate(_phase, _concentration)
		self.module_x_array = []
		self.module_y_array = []

		self.num_of_module = 4
		for i in range(self.num_of_module):   
			self.module_x_array.append(GridModule(0.5*math.pow(1.5,i), 0, 650))
			self.module_y_array.append(GridModule(0.5*math.pow(1.5,i), 0, 650))


		self._x = 0
		self._y = 0




	def time_update(self, input_w, input_w_std, input_v, input_v_std, dt):
		input_w_cov = math.pow(input_w_std, 2)
		input_v_cov = math.pow(input_v_std, 2)

		# estimate update
		self.theta_est.time_update(input_w*dt, 1/(input_w_cov*dt*dt))


		total_distance = input_v * func_A(self.theta_est.concentration) * dt
		total_distance_std = math.sqrt(input_v_cov+math.pow(input_v, 2)) * dt


		for i in range(self.num_of_module):   # 5 modules, the scale grow from 25 cm with spatial ratio 1.5
			self.module_x_array[i].time_update(total_distance*math.cos(self.theta_est.phase), total_distance_std)
			self.module_y_array[i].time_update(total_distance*math.sin(self.theta_est.phase), total_distance_std)




	def direct_observation_update(self, obs_theta, obs_theta_cct, obs_x, obs_y, obs_std):
		self.theta_est.observation_update(obs_theta, obs_theta_cct)

		for i in range(self.num_of_module):   # 5 modules, the scale grow from 25 cm with spatial ratio 1.5
			self.module_x_array[i].observation_update(obs_x, obs_std)
			self.module_y_array[i].observation_update(obs_y, obs_std)


	def bd_observation_update(self, bearing, bearing_cct, distance, distance_std):
		distance_cov = distance_std**2

		_position = self.cartesian_readout()

		# orientation update
		position_cct = self.module_x_array[-1].estimate.concentration * math.pow(2*math.pi/self.module_x_array[-1].scale ,2)
		dx = landmark[0]-_position[0]
		dy = landmark[1]-_position[1]


		est_distance = math.sqrt(dx**2 + dy**2)
		est_phase = math.atan2(dy, dx) - bearing
		_cct = mean_of_cct(bearing_cct, est_distance*distance*position_cct*0.5)
		self.theta_est.force_observation_update(est_phase, _cct)


		# position update
		_equ_x = landmark[0] - distance * func_A(bearing_cct) * func_A(self.theta_est.concentration) * math.cos(self.theta_est.phase + bearing)
		_equ_y = landmark[1] - distance * func_A(bearing_cct) * func_A(self.theta_est.concentration) * math.sin(self.theta_est.phase + bearing)
		_equ_std = math.sqrt(distance_cov+math.pow(distance, 2))


		for i in range(self.num_of_module):   # 5 modules, the scale grow from 25 cm with spatial ratio 1.5
			self.module_x_array[i].observation_update(_equ_x, _equ_std)
			self.module_y_array[i].observation_update(_equ_y, _equ_std)




	def _grid_to_catesian(self, module_array):
		_num = 801

		_space = np.linspace(-1.2, 1.2, num=_num)
		_likelihood = np.zeros_like(_space)

		for i in range(_num):
			_point = _space[i]
			for module in module_array:
				_likelihood[i] += module.estimate.concentration * math.cos(2.0*math.pi*_point/module.scale - module.estimate.phase)

		return _space[np.argmax(_likelihood)]

	def cartesian_readout(self):
		return [self._grid_to_catesian(self.module_x_array), self._grid_to_catesian(self.module_y_array)]

	def read_estimation(self):
		[_x, _y] = self.cartesian_readout()
		return [self.theta_est.phase, _x, _y]



########



class LieGroupSpatialState:
	def __init__(self, _mean=np.matrix([[0.0], [0.0], [0.0]]), _cov=np.matrix([[0.01,0,0], [0,0.01,0], [0,0,0.01]])):
		self.mean = LieGroup(_mean.item(0,0), _mean.item(1,0), _mean.item(2,0))
		self.cov = _cov

	def time_update(self, input_w, input_w_std, input_v, input_v_std, dt):
		u_t = np.matrix([[input_w*dt], [input_v*dt], [0]])
		self.mean = self.mean * exp_SE2(u_t)

		F_t = Adjoint_rep(exp_SE2((-1)*u_t)).lg_matrix
		Phi_G = Phi(u_t)
		Q = np.matrix([[(input_w_std*dt)**2, 0, 0],
			[0, (input_v_std*dt)**2, 0],
			[0, 0, 0]])
		self.cov = F_t * self.cov * F_t.getT() + Phi_G * Q * Phi_G.getT()

	def bd_observation_update(self, bearing, bearing_std, distance, distance_std):

		[_theta, _x, _y] = self.mean.toEuclidean()

		dx = landmark[0]-_x
		dy = landmark[1]-_y

		distance_cov = distance_std ** 2
		bearing_cov = bearing_std ** 2

		est_bearing = (math.atan2(dy, dx) - _theta) % (2*math.pi)
		est_distance = math.sqrt(math.pow(dx,2) + math.pow(dy,2))

		observ = np.matrix([[bearing], [distance]])
		est_observ = np.matrix([[est_bearing], [est_distance]])

		H = np.matrix([[-1, dy/(math.pow(dx,2) + math.pow(dy,2)), -dx/(math.pow(dx,2) + math.pow(dy,2))], 
			[0, -dx/est_distance, -dy/est_distance]])

		S = np.matrix([[bearing_cov, 0],[0, distance_cov]]) + H * self.cov * H.getT()

		innovation = observ - est_observ
		innovation[0] = ((innovation[0] + math.pi) % (2*math.pi)) - math.pi

		m_t = self.cov * H.getT() * S.getI() * innovation
		self.mean = self.mean * exp_SE2(m_t)

		self.cov = self.cov - self.cov * H.getT() * S.getI() * H * self.cov

	def read_estimation(self):
		return self.mean.toEuclidean()
	



########

def func_A(x):
	return np.divide(special.iv(1, x), special.iv(0, x))

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



'''
def func_A_approx(x):

	I_1 = 1 - 3.0/(8*x) - 15.0/(128*x*x)
	I_0 = 1 + 1.0/(8*x) + 9.0/(128*x*x)

	return I_1 / I_0
'''



'''
def inv_func_A(y):

	_accuracy = 0.00000000000001

	_upper_limit = 700
	_middle_value = 1.1593199207635498   # func_A( ) = 0.50000000000427447
	_lower_limit =  0.0000000000001


	while _upper_limit - _lower_limit > _accuracy:

		# if y >= func_A(_middle_value):
		#	_lower_limit = _middle_value

		# else:
		#	_upper_limit = _middle_value


		if y < func_A(_middle_value):
			_upper_limit = _middle_value

		else:
			_lower_limit = _middle_value

		_middle_value = 0.5 * (_upper_limit+_lower_limit)

	return _middle_value


def inv_func_A_nr(y):

	x = 1.16
	diff = 10.0

	if y == 0:
		return .00000000000001
	elif y > .5:

		while (x < 700) and (diff > 0.1):
			func_value = func_A(x)
			x_next = x - (func_value-y)/(1 - func_value*(func_value + 1.0/y))
			diff = abs(x - x_next)
			x = x_next

		return min(x, 700)

	else:

		while diff > 0.00000000000001:
			func_value = func_A(x)
			x_next = x - (func_value-y)/(1 - func_value*(func_value + 1.0/y))
			diff = abs(x - x_next)
			x = x_next

		return x
'''

