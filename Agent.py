import numpy as np
import math
import yaml

from algorithms.EKF import EKF
from algorithms.LGEKF import LGEKF
from algorithms.hybrid import hybrid_loc_algo
from algorithms.circular import circular_loc_algo


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
			initial_cov =np.array([[0.01,0,0], [0,0.01,0], [0,0,0.01]], dtype=float)

			# self.EKF_estimate = Estimators.GaussianSpatialState(_mean=initial_state, _cov=initial_cov)
			self.EKF_estimate = EKF(_mean=initial_state, _cov=initial_cov)
			self.lie_estimate = LGEKF(_mean=initial_state, _cov=initial_cov) 
			self.hybrid_estimate = hybrid_loc_algo(_phase=_theta, _concentration=1.0/0.01, _x=_position[0], _x_std=0.01, _y=_position[1], _y_std=0.01)
			self.circular_estimate = circular_loc_algo(_phase=_theta, _concentration=1.0/0.01)
			     

		else:                                    # unkown initial case (dynamic sim)
			theta_cov = 1.0 / _init_theta_cct 
			initial_state = np.matrix([0.0, 0.0, 0.0]).getT()
			initial_cov =np.array([[theta_cov,0,0], [0,0.01,0], [0,0,0.01]], dtype=float)

			# self.EKF_estimate = Estimators.GaussianSpatialState(_mean=initial_state, _cov=initial_cov)
			self.EKF_estimate = EKF(_mean=initial_state, _cov=initial_cov)
			self.lie_estimate = LGEKF(initial_state, initial_cov)     
			self.hybrid_estimate = hybrid_loc_algo(_phase=0.0, _concentration=_init_theta_cct, _x=0.0, _x_std=0.01, _y=0.0, _y_std=0.01)
			self.circular_estimate = circular_loc_algo(_phase=0.0, _concentration=_init_theta_cct)




	def time_update(self):

		dt = config['dt'] # s
		self.time += dt

		# angular velocity
		w = config['w'] 
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
		self.EKF_estimate.bd_observation_update(landmark, observ_bearing, phi_std, observ_distance, d_std)
		self.hybrid_estimate.bd_observation_update(landmark, observ_bearing, phi_cct, observ_distance, d_std)
		self.circular_estimate.bd_observation_update(landmark, observ_bearing, phi_cct, observ_distance, d_std)
		self.lie_estimate.bd_observation_update(landmark, observ_bearing, phi_std, observ_distance, d_std)


	def estimation_error(self, est_theta, est_x, est_y):

		or_error = 1 - math.cos(self.theta - est_theta)
		loc_error = math.sqrt((self.position[0] - est_x)**2 + (self.position[1] - est_y)**2)

		return [or_error, loc_error]


