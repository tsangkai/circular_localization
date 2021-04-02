
import numpy as np
import math

class EKF:
	def __init__(self, _mean=np.zeros((3,1)), _cov=0.01*np.identity(3)):
		self.mean = _mean
		self.cov = _cov


	def time_update(self, input_w, input_w_std, input_v, input_v_std, dt):
		input_w_cov = math.pow(input_w_std, 2)
		input_v_cov = math.pow(input_v_std, 2)

		self.mean[0,0] += input_w * dt
		self.mean[1,0] += input_v * dt * math.cos(self.mean[0,0])
		self.mean[2,0] += input_v * dt * math.sin(self.mean[0,0])

		F_state_2_state = np.array([[1, 0, 0],
			                        [-input_v*dt*math.sin(self.mean[0,0]), 1, 0],
			                        [input_v*dt*math.cos(self.mean[0,0]),  0, 1]])

		F_input_2_state = dt * np.array([[1, 0],
			                        [0, math.cos(self.mean[0,0])],
			                        [0,  math.sin(self.mean[0,0])]])

		Q = np.array([[input_w_cov, 0],[0, input_v_cov]])

		self.cov = F_state_2_state @ self.cov @ F_state_2_state.T + F_input_2_state @ Q @ F_input_2_state.T


	def bd_observation_update(self, landmark, bearing, bearing_std, distance, distance_std):

		distance_cov = distance_std ** 2
		bearing_cov = bearing_std ** 2
		
		dx = landmark[0]-self.mean[1,0]
		dy = landmark[1]-self.mean[2,0]

		est_bearing = (math.atan2(dy, dx) - self.mean[0,0]) % (2*math.pi)
		est_distance = math.sqrt(math.pow(dx,2) + math.pow(dy,2))

		observ = np.array([[bearing], [distance]])
		est_observ = np.array([[est_bearing], [est_distance]])

		H = np.array([[-1, dy/(math.pow(dx,2) + math.pow(dy,2)), -dx/(math.pow(dx,2) + math.pow(dy,2))], 
			[0, -dx/est_distance, -dy/est_distance]])

		S = np.array([[bearing_cov, 0],[0, distance_cov]]) + H @ self.cov @ H.T

		innovation = observ - est_observ
		innovation[0] = ((innovation[0] + math.pi) % (2*math.pi)) - math.pi

		self.mean = self.mean + self.cov @ H.T @ np.linalg.inv(S) @ innovation
		self.cov = self.cov - self.cov @ H.T @ np.linalg.inv(S) @ H @ self.cov

	def read_estimation(self):
		return [self.mean[0,0], self.mean[1,0], self.mean[2,0]]