# estimation error vs the number of trials

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

theta_ccr_arr = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]


### output data

orientation_err = np.zeros((3, len(theta_ccr_arr)))
position_err = np.zeros((3, len(theta_ccr_arr)))

### simulation

output_init_file = open("result/initial.txt", "w")

theta_ccr_idx = 0
    
for theta_ccr in theta_ccr_arr:

	print("initial concentration parameter = " + str(theta_ccr))

	error_ekf = np.zeros([1, 2])
	error_ekf_2 = np.zeros([1, 2])

	error_hybrid = np.zeros([1, 2])
	error_hybrid_2 = np.zeros([1, 2])

	error_lie = np.zeros([1, 2])
	error_lie_2 = np.zeros([1, 2])

	for n in range(N):

		init_theta = np.random.vonmises(0, theta_ccr)
		agent_1 = Agent.Agent(_theta=init_theta, _position=[0,0], _init_theta_given=False, _init_theta_cct=theta_ccr)

		for T in range(num_T):

			for t in range(num_t):

				agent_1.time_update()
				
				# EKF
				[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
				[or_error, loc_error] = agent_1.estimation_error(ekf_theta, ekf_x, ekf_y)

				error_ekf[0,0] += or_error
				error_ekf[0,1] += loc_error 
				error_ekf_2[0,0] += or_error ** 2
				error_ekf_2[0,1] += loc_error ** 2	

				# hybrid
				[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
				[or_error, loc_error] = agent_1.estimation_error(hybrid_theta, hybrid_x, hybrid_y)

				error_hybrid[0,0] += or_error 
				error_hybrid[0,1] += loc_error
				error_hybrid_2[0,0] += or_error ** 2
				error_hybrid_2[0,1] += loc_error ** 2

				# lie
				[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
				[or_error, loc_error] = agent_1.estimation_error(lie_theta, lie_x, lie_y)

				error_lie[0,0] += or_error 
				error_lie[0,1] += loc_error
				error_lie_2[0,0] += or_error ** 2
				error_lie_2[0,1] += loc_error ** 2


			agent_1.bd_observation_update()

			# EKF
			[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(ekf_theta, ekf_x, ekf_y)

			error_ekf[0,0] += or_error
			error_ekf[0,1] += loc_error 
			error_ekf_2[0,0] += or_error ** 2
			error_ekf_2[0,1] += loc_error ** 2	
			

			# hybrid
			[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(hybrid_theta, hybrid_x, hybrid_y)

			error_hybrid[0,0] += or_error 
			error_hybrid[0,1] += loc_error
			error_hybrid_2[0,0] += or_error ** 2
			error_hybrid_2[0,1] += loc_error ** 2


			# lie
			[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(lie_theta, lie_x, lie_y)

			error_lie[0,0] += or_error 
			error_lie[0,1] += loc_error
			error_lie_2[0,0] += or_error ** 2
			error_lie_2[0,1] += loc_error ** 2


	orientation_err[0, theta_ccr_idx] = error_ekf[0,1]/(total_sample_number*N)
	position_err[0, theta_ccr_idx] = math.sqrt(total_sample_number*N*error_ekf_2[0,1] - error_ekf[0,1]**2)/(total_sample_number*N)

	orientation_err[1, theta_ccr_idx] = error_hybrid[0,1]/(total_sample_number*N)
	position_err[1, theta_ccr_idx] = math.sqrt(total_sample_number*N*error_hybrid_2[0,1] - error_hybrid[0,1]**2)/(total_sample_number*N)

	orientation_err[2, theta_ccr_idx] = error_lie[0,1]/(total_sample_number*N)
	position_err[2, theta_ccr_idx] = math.sqrt(total_sample_number*N*error_lie_2[0,1] - error_lie[0,1]**2)/(total_sample_number*N)

	output_str = '{:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4}\n'.format(theta_ccr, error_ekf[0,0] / (total_sample_number*N), error_ekf[0,1] / (total_sample_number*N), error_hybrid[0,0] / (total_sample_number*N), error_hybrid[0,1] / (total_sample_number*N), error_lie[0,0] / (total_sample_number*N), error_lie[0,1] / (total_sample_number*N))
	output_init_file.write(output_str)


	theta_ccr_idx += 1

output_init_file.close()



### visualization

plot_color = {
	'EKF': config['color']['grenadine'],
	'LG-EKF': config['color']['mustard'],
	'hybrid': config['color']['navy'],
	'circular': config['color']['spruce']
}

plt.figure(1)

plt.subplot(211)

plt.plot(theta_ccr_arr, orientation_err[0,:], color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
plt.plot(theta_ccr_arr, orientation_err[1,:], color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
plt.plot(theta_ccr_arr, orientation_err[2,:], color = plot_color['LG-EKF'], linewidth=1.6, label = 'Lie-EKF')

plt.ylabel('orientation err')
plt.xscale('log') 

plt.legend()

plt.subplot(212)
plt.plot(theta_ccr_arr, position_err[0,:], color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
plt.plot(theta_ccr_arr, position_err[1,:], color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
plt.plot(theta_ccr_arr, position_err[2,:], color = plot_color['LG-EKF'], linewidth=1.6, label = 'Lie-EKF')

plt.ylabel('position err')
plt.xlabel('initial concentration parameter $\kappa_0$')
plt.xscale('log') 

plt.show()

