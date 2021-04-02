

import math

from util.A_func import func_A, mean_of_cct
from algorithms.basic_estimate import GaussianEstimate, CircularEstimate


class hybrid_loc_algo:
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


	def bd_observation_update(self, landmark, bearing, bearing_cct, distance, distance_std):
		distance_cov = distance_std ** 2

		dx = landmark[0]-self.x_est.mean
		dy = landmark[1]-self.y_est.mean
		est_distance = math.sqrt(dx**2 + dy**2)
		landmark_orientation = math.atan2(dy, dx)


		# intermediate terms
		_equ_phase = landmark_orientation - bearing
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

