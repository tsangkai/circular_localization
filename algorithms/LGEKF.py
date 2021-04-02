import numpy as np
import math

from util.LieGroup import *


class LGEKF:
	def __init__(self, _mean=np.zeros((3,1)), _cov=0.01*np.identity(3)):
		self.mean = LieGroup(_mean[0,0], _mean[1,0], _mean[2,0])
		self.cov = _cov

	def time_update(self, input_w, input_w_std, input_v, input_v_std, dt):
		u_t = np.array([[input_w*dt], [input_v*dt], [0]])
		self.mean = self.mean * exp_SE2(u_t)

		F_t = Adjoint_rep(exp_SE2((-1)*u_t)).lg_matrix
		Phi_G = Phi(u_t)
		Q = np.array([[(input_w_std*dt)**2, 0, 0],
			[0, (input_v_std*dt)**2, 0],
			[0, 0, 0]])
		self.cov = F_t @ self.cov @ F_t.T + Phi_G @ Q @ Phi_G.T


	def bd_observation_update(self, landmark, bearing, bearing_std, distance, distance_std):

		[_theta, _x, _y] = self.mean.toEuclidean()

		dx = landmark[0]-_x
		dy = landmark[1]-_y

		distance_cov = distance_std ** 2
		bearing_cov = bearing_std ** 2

		est_bearing = (math.atan2(dy, dx) - _theta) % (2*math.pi)
		est_distance = math.sqrt(dx**2 + dy**2)

		H = np.array([[-1, dy/(math.pow(dx,2) + math.pow(dy,2)), -dx/(math.pow(dx,2) + math.pow(dy,2))], 
			[0, -dx/est_distance, -dy/est_distance]])

		S = np.array([[bearing_cov, 0],[0, distance_cov]]) + H @ self.cov @ H.T

		observ = np.array([[bearing], [distance]])
		est_observ = np.array([[est_bearing], [est_distance]])

		innovation = observ - est_observ
		innovation[0] = ((innovation[0] + math.pi) % (2*math.pi)) - math.pi

		m_t = self.cov @ H.T @ np.linalg.inv(S) @ innovation
		self.mean = self.mean * exp_SE2(m_t)

		Phi_G = Phi(m_t)

		self.cov = Phi_G @ (self.cov - self.cov @ H.T @ np.linalg.inv(S) @ H @ self.cov) @ Phi_G.T

	def read_estimation(self):
		return self.mean.toEuclidean()
	
