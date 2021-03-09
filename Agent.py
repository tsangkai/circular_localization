from scipy import special
import numpy as np
import math
import cmath

import Estimators
# import Parameter

import yaml

with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)



class Agent:
	def __init__(self, _theta=0.0, _position=[0,0], _init_theta_given=True, _init_theta_cct=500.0):

		# parameter
		self.time = 0.0

		# state initialization
		self.theta = _theta 
		self.position = _position


		if _init_theta_given:
			initial_state = np.matrix([_theta, _position[0], _position[1]]).getT()
			initial_cov = _cov=np.matrix([[0.01,0,0], [0,0.01,0], [0,0,0.01]])

			self.EKF_estimate = Estimators.GaussianSpatialState(_mean=initial_state, _cov=initial_cov)
			self.lie_estimate = Estimators.LieGroupSpatialState(_mean=initial_state, _cov=initial_cov) 
			self.hybrid_estimate = Estimators.HybridSpatialState(_phase=_theta, _concentration=1.0/0.01, _x=_position[0], _x_std=0.01, _y=_position[1], _y_std=0.01)
			self.circular_estimate = Estimators.CircularSpatialState(_phase=_theta, _concentration=1.0/0.01)
			     

		else:                                    # unkown initial case (dynamic sim)
			theta_cov = 1.0 / _init_theta_cct 
			initial_state = np.matrix([0, 0, 0]).getT()
			initial_cov = _cov=np.matrix([[theta_cov,0,0], [0,0.01,0], [0,0,0.01]])

			self.EKF_estimate = Estimators.GaussianSpatialState(initial_state, initial_cov)
			self.hybrid_estimate = Estimators.HybridSpatialState(_phase=0, _concentration=_init_theta_cct, _x=0, _x_std=0.01, _y=0, _y_std=0.01)
			self.circular_estimate = Estimators.CircularSpatialState(_phase=0, _concentration=_init_theta_cct)
			self.lie_estimate = Estimators.LieGroupSpatialState(initial_state, initial_cov)     




	def time_update(self):

		dt = config['dt'] # s
		self.time += dt

		# angular velocity
		w = config['w'] 
		# w = 0.8 * (1 - math.sqrt(self.position[0]**2 + self.position[1]**2)/1.3)
		w_std = config['w_std'] 

		# translational velocity
		v = config['v'] 
		v_std = config['v_std'] 
        
		real_v = v + np.random.normal(0, v_std)
		real_w = w + np.random.normal(0, w_std)

		# state update
		self.position[0] = self.position[0] + real_v * dt * math.cos(self.theta)
		self.position[1] = self.position[1] + real_v * dt * math.sin(self.theta)
		self.theta = (self.theta + real_w * dt) % (2*math.pi)

		# estimate update
		self.EKF_estimate.time_update(w, w_std, v, v_std, dt)
		self.hybrid_estimate.time_update(w, w_std, v, v_std, dt)
		self.circular_estimate.time_update(w, w_std, v, v_std, dt)
		self.lie_estimate.time_update(w, w_std, v, v_std, dt)



	def rel_observation_update(self):

		# parameters
		landmark = config['landmark_position']

		d_std = config['d_std'] 

		# observation construction
		observation = np.matrix(landmark).getT() - np.matrix(self.position).getT() + np.matrix([[np.random.normal(0, d_std)],[np.random.normal(0, d_std)]])

		# estimate update
		self.EKF_estimate.rel_observation_update(observation, d_std)


	def direct_observation_update(self):

		obs_theta_cct = config['phi_cct'] # Parameter.phi_cct
		obs_theta = self.theta + np.random.vonmises(0, obs_theta_cct)

		obs_std = config['d_std'] # Parameter.d_std
		obs_x = self.position[0] + np.random.normal(0, obs_std)
		obs_y = self.position[1] + np.random.normal(0, obs_std)
		self.circular_estimate.direct_observation_update(obs_theta, obs_theta_cct, obs_x, obs_y, obs_std)


	def bd_observation_update(self):

		# parameters
		landmark = config['landmark_position']

		phi_cct = config['phi_cct'] 
		phi_std = 1.0 / math.sqrt(phi_cct) #0.01
		d_std = config['d_std']
		
		
		dx = landmark[0] - self.position[0]
		dy = landmark[1] - self.position[1]

		observ_bearing = (math.atan2(dy, dx) - self.theta + np.random.vonmises(0, phi_cct)) % (2*math.pi)
		observ_distance = math.sqrt(dx**2 + dy**2) + np.random.normal(0, d_std)

		# notice that some input is phi_std, and some is phi_cct
		self.EKF_estimate.bd_observation_update(observ_bearing, phi_std, observ_distance, d_std)
		self.hybrid_estimate.bd_observation_update(observ_bearing, phi_cct, observ_distance, d_std)
		self.circular_estimate.bd_observation_update(observ_bearing, phi_cct, observ_distance, d_std)
		self.lie_estimate.bd_observation_update(observ_bearing, phi_std, observ_distance, d_std)


	def estimation_error(self, est_theta, est_x, est_y):

		or_error = 1 - math.cos(self.theta - est_theta)
		loc_error = math.sqrt((self.position[0] - est_x)**2 + (self.position[1] - est_y)**2)

		return [or_error, loc_error]

