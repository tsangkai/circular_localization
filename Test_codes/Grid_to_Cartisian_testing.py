import numpy as np
import matplotlib.pyplot as plt
import yaml

from scipy import special
import math
import cmath


import Estimators

w = 0.8        # rad/s
w_std = 0.00004
# translational velocity
v = 0.3        # m/s
v_std = 0.00003
##### Observation update noise parameter
phi_cct = 1000.0
d_std = 0.01 
dt = 0.1

t_end = 50
_theta = 0



time = 0.0
# state initialization
theta = _theta 
init_position = [1,2]








# true_trajectory = Estimators.True_trajectrory(_theta = theta, _x = init_position[0], _y = init_position[1], dt= dt)
true_trajectory = Estimators.True_trajectrory(_theta = theta, _x = init_position[0], _y = init_position[1], dt= dt)
circular_estimate = Estimators.CircularSpatialState(_phase=_theta, _concentration=1.0/0.01)


true_states = np.zeros((t_end,3))
circular_setimate_states = np.zeros((t_end,3))


for i in range(0, t_end, 1):
	
	real_v = v + np.random.normal(0, v_std)
	real_w = w + np.random.normal(0, w_std)

	true_states[i] = true_trajectory.true_time_update(real_v, real_w)
	
	circular_estimate.time_update(w, w_std, v, v_std, dt)
	circular_setimate_states[i] = circular_estimate.read_estimation()
    
plt.scatter(circular_setimate_states[:,1], circular_setimate_states[:,2])
plt.plot(true_states[:,1], true_states[:,2])

plt.show()