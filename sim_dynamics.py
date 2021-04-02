# Localization trajectory

import numpy as np
import math

import yaml
import matplotlib.pyplot as plt

import Agent


### import parameters

with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)

# output_init_file = open("result/dynamics.csv", "w")
N = config['num_of_trial'] 

num_T = config['num_T'] 
num_t = config['num_t'] 

total_sample_number = (num_t+1) * num_T
# np.random.seed(1)

### output data
time_arr = np.zeros([total_sample_number, 1])    

error_ekf = np.zeros([total_sample_number, 2])
error_hybrid = np.zeros([total_sample_number, 2])
error_circular = np.zeros([total_sample_number, 2])
error_lie = np.zeros([total_sample_number, 2])


# simulation
for n in range(N):
	print(n)
	theta_ccr = 200.0
	init_theta = np.random.vonmises(0.0, theta_ccr)
	agent_1 = Agent.Agent(_theta = init_theta, _position = [0.0, 0.0], 
	   	_init_theta_given=False, _init_theta_cct=theta_ccr)

	# agent_1 = Agent.Agent(_theta = np.random.uniform(-math.pi, math.pi), _position = [0.0, 0.0], 
	#  	_init_theta_given=False, _init_theta_cct=0.01)

	i = 0
	for T in range(num_T):
		for t in range(num_t):

			agent_1.time_update()
			time_arr[i] = agent_1.time
		

			# EKF
			[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(ekf_theta, ekf_x, ekf_y)
			error_ekf[i,0] += or_error / N
			error_ekf[i,1] += loc_error / N
			
			# hybrid
			[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(hybrid_theta, hybrid_x, hybrid_y)
			error_hybrid[i,0] += or_error / N
			error_hybrid[i,1] += loc_error / N	


			#lie
			[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(lie_theta, lie_x, lie_y)
			error_lie[i,0] += or_error / N
			error_lie[i,1] += loc_error / N
			
			#circular
			[circular_theta, circular_x, circular_y] = agent_1.circular_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(circular_theta, circular_x, circular_y)
			error_circular[i,0] += or_error / N
			error_circular[i,1] += loc_error / N
			
			#Write to the file
			# output_str = '{}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}\n'.format(n, time_arr[i].item(), error_ekf[i,0], error_ekf[i,1],\
			# 																						error_hybrid[i,0], error_hybrid[i,1],\
			# 																						error_lie[i,0], error_lie[i,1],\
			# 																						error_circular[i,0], error_circular[i,1])	
			# output_init_file.write(output_str)

			i = i+1


		agent_1.bd_observation_update()
		time_arr[i] = agent_1.time 

		# EKF
		[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
		[or_error, loc_error] = agent_1.estimation_error(ekf_theta, ekf_x, ekf_y)
		error_ekf[i,0] += or_error / N
		error_ekf[i,1] += loc_error / N

		# hybrid
		[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
		[or_error, loc_error] = agent_1.estimation_error(hybrid_theta, hybrid_x, hybrid_y)
		error_hybrid[i,0] += or_error / N
		error_hybrid[i,1] += loc_error / N

		# lie
		[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
		[or_error, loc_error] = agent_1.estimation_error(lie_theta, lie_x, lie_y)
		error_lie[i,0] += or_error / N
		error_lie[i,1] += loc_error / N

		#circular
		[circular_theta, circular_x, circular_y] = agent_1.circular_estimate.read_estimation()
		[or_error, loc_error] = agent_1.estimation_error(circular_theta, circular_x, circular_y)
		error_circular[i,0] += or_error / N
		error_circular[i,1] += loc_error / N
		
		#Write to the file
		# output_str = '{}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}\n'.format(n, time_arr[i].item(), error_ekf[i,0], error_ekf[i,1], \
		# 																							error_hybrid[i,0], error_hybrid[i,1], \
		# 																							error_lie[i,0], error_lie[i,1],  \
		# 																							error_circular[i,0], error_circular[i,1])	
		# output_init_file.write(output_str)
		


		i = i+1

	del agent_1



dynamc_error = np.hstack((time_arr,error_ekf, error_hybrid, error_lie, error_circular))
np.savetxt('result/dynamics.txt', dynamc_error)
print(dynamc_error.shape)
# output_init_file.close()
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
plt.plot(time_arr, error_circular[:,0], color = plot_color['circular'], linewidth=1.6, label = 'circular')
plt.plot(time_arr, error_lie[:,0], color = plot_color['LG-EKF'], linewidth=1.6, label = 'Lie-EKF')

plt.ylabel('orientation err')
plt.legend()
plt.subplot(212)
plt.plot(time_arr, error_ekf[:,1], color = plot_color['EKF'], linewidth=1.6)
plt.plot(time_arr, error_hybrid[:,1], color = plot_color['hybrid'], linewidth=1.6)
plt.plot(time_arr, error_circular[:,1], color = plot_color['circular'], linewidth=1.6)
plt.plot(time_arr, error_lie[:,1], color = plot_color['LG-EKF'], linewidth=1.6)
plt.ylabel('position err')

# plt.ylim([0, 0.3])
plt.show()



