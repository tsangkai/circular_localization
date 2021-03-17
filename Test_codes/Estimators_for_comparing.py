
import numpy as np
import yaml

from scipy import special
import math
import cmath


### import parameters

with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)

landmark = config['landmark_position']



class True_trajectrory:
	def __init__(self, _theta = 0, _x = 0, _y = 0, dt= 0.01):
		# state initialization
		self.theta = _theta
		self.x = _x
		self.y = _y
		self.dt = dt


	def true_time_update(self, real_v, real_w):
		#True state update
		self.x = self.x + real_v * self.dt * math.cos(self.theta)
		self.y = self.y + real_v * self.dt * math.sin(self.theta)
		self.theta = (self.theta + real_w * self.dt) % (2*math.pi)

		return self.theta, self.x, self.y






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

		self.num_of_module = 4    #m = 4
		for i in range(self.num_of_module):   
			self.module_x_array.append(GridModule(0.5*math.pow(1.5,i), 0, 650))
			self.module_y_array.append(GridModule(0.5*math.pow(1.5,i), 0, 650))


		# self._x = 0
		# self._y = 0




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
	


	def bd_observation_update(self, bearing, bearing_cct, distance, distance_std):
		distance_cov = distance_std**2

		_position = self.cartesian_readout()
		print(_position)

		# orientation update
		position_cct = self.module_x_array[-1].estimate.concentration * math.pow(2*math.pi/self.module_x_array[-1].scale ,2)
		dx = landmark[0]-_position[0]
		dy = landmark[1]-_position[1]
		est_distance = math.sqrt(dx**2 + dy**2)
		landmark_orientation = math.atan2(dy, dx)
		est_phase = landmark_orientation - bearing
		_cct = mean_of_cct(est_distance*distance*position_cct*0.5, bearing_cct)
		

		self.theta_est.force_observation_update(est_phase, _cct)


		# position update
		_equ_x = landmark[0] - distance * func_A(bearing_cct) * func_A(self.theta_est.concentration) * math.cos(self.theta_est.phase + bearing)
		_equ_y = landmark[1] - distance * func_A(bearing_cct) * func_A(self.theta_est.concentration) * math.sin(self.theta_est.phase + bearing)
		_equ_std = math.sqrt(distance_cov+ distance ** 2)


		for i in range(self.num_of_module):   # 5 modules, the scale grow from 25 cm with spatial ratio 1.5
			self.module_x_array[i].observation_update(_equ_x, _equ_std)
			self.module_y_array[i].observation_update(_equ_y, _equ_std)




	def _grid_to_catesian(self, module_array, max_value=2, min_value=-1): #Changes from -1.2-1.2 to -10-10
		_num = 801

		_space = np.linspace(min_value, max_value, num=_num)
		_likelihood = np.zeros_like(_space) 

		for i in range(_num):
			_point = _space[i]
			for module in module_array:
				_likelihood[i] += module.estimate.concentration * (math.cos(2.0*math.pi*_point/module.scale - module.estimate.phase))

		return _space[np.argmax(_likelihood)]

	def cartesian_readout(self):
		_x = self._grid_to_catesian(self.module_x_array)
		_y = self._grid_to_catesian(self.module_y_array)
		return [_x, _y]

	def read_estimation(self):
		[_x, _y] = self.cartesian_readout()
		return [self.theta_est.phase, _x, _y]



##########################################


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

