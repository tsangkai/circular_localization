# Localization trajectory

import numpy as np
import math

import yaml
import matplotlib.pyplot as plt

import Agent


### import parameters

with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)

N = config['num_of_trial'] 

num_T = config['num_T'] 
num_t = config['num_t'] 

total_sample_number = (num_t+1) * num_T


### output data

time_arr = np.zeros([total_sample_number, 1])    

error_ekf = np.zeros([total_sample_number, 2])
error_hybrid = np.zeros([total_sample_number, 2])
error_circular = np.zeros([total_sample_number, 2])
error_lie = np.zeros([total_sample_number, 2])



# simulation


for n in range(N):
	print(n)

	agent_1 = Agent.Agent(_theta = np.random.uniform(-math.pi, math.pi), _position = [0,0], 
		_init_theta_given=False, _init_theta_cct=0.1)

	i = 0
	for T in range(num_T):
		for t in range(num_t):

			agent_1.time_update()
			time_arr[i] = agent_1.time 

			# EKF
			[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(ekf_theta, ekf_x, ekf_y)
			error_ekf[i,0] += or_error / total_sample_number
			error_ekf[i,1] += loc_error / total_sample_number
			
			# hybrid
			[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(hybrid_theta, hybrid_x, hybrid_y)
			error_hybrid[i,0] += or_error / total_sample_number
			error_hybrid[i,1] += loc_error / total_sample_number	

			# lie
			[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(lie_theta, lie_x, lie_y)
			error_lie[i,0] += or_error / total_sample_number
			error_lie[i,1] += loc_error / total_sample_number

			i = i+1


		agent_1.bd_observation_update()
		time_arr[i] = agent_1.time 

		# EKF
		[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
		[or_error, loc_error] = agent_1.estimation_error(ekf_theta, ekf_x, ekf_y)
		error_ekf[i,0] += or_error / total_sample_number
		error_ekf[i,1] += loc_error / total_sample_number

		# hybrid
		[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
		[or_error, loc_error] = agent_1.estimation_error(hybrid_theta, hybrid_x, hybrid_y)
		error_hybrid[i,0] += or_error / total_sample_number
		error_hybrid[i,1] += loc_error / total_sample_number

		# lie
		[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
		[or_error, loc_error] = agent_1.estimation_error(lie_theta, lie_x, lie_y)
		error_lie[i,0] += or_error / total_sample_number
		error_lie[i,1] += loc_error / total_sample_number
			
		i = i+1

	del agent_1

### visualization

plot_color = {
	'EKF': config['color']['grenadine'],
	'LG-EKF': config['color']['mustard'],
	'hybrid': config['color']['navy'],
	'circular': config['color']['spruce']
}

plt.figure(1)

plt.subplot(211)

plt.plot(time_arr, error_ekf[:,0], color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
plt.plot(time_arr, error_hybrid[:,0], color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
plt.plot(time_arr, error_lie[:,0], color = plot_color['LG-EKF'], linewidth=1.6, label = 'Lie-EKF')

plt.ylabel('orientation err')
plt.ylim([0, 0.012])

plt.legend()

plt.subplot(212)
plt.plot(time_arr, error_ekf[:,1], color = plot_color['EKF'], linewidth=1.6)
plt.plot(time_arr, error_hybrid[:,1], color = plot_color['hybrid'], linewidth=1.6)
plt.plot(time_arr, error_lie[:,1], color = plot_color['LG-EKF'], linewidth=1.6)

plt.ylabel('position err')

# plt.ylim([0, 0.3])
plt.show()



