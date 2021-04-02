
import numpy as np

import math
import cmath

from util.A_func import mean_of_cct


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
	def __init__(self, _phase=0, _concentration=100):
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
