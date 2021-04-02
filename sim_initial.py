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
np.random.seed(1)

theta_ccr_arr = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]


### output data

orientation_err = np.zeros((4, len(theta_ccr_arr)))
position_err = np.zeros((4, len(theta_ccr_arr)))

### simulation

output_init_file = open("result/initial.csv", "w")

theta_ccr_idx = 0

err_array = np.zeros((len(theta_ccr_arr), 8, N, total_sample_number))    
for theta_ccr in theta_ccr_arr:

	print("initial concentration parameter = " + str(theta_ccr))


	error_ekf = np.zeros([1, 2])
	error_ekf_2 = np.zeros([1, 2])


	error_hybrid = np.zeros([1, 2])
	error_hybrid_2 = np.zeros([1, 2])

	error_lie = np.zeros([1, 2])
	error_lie_2 = np.zeros([1, 2])

	error_circular = np.zeros([1, 2])
	error_circular_2 = np.zeros([1, 2])

	i = 0

	for n in range(N):

		sample_idx = 0
		init_theta = np.random.vonmises(0, theta_ccr)
		agent_1 = Agent.Agent(_theta=init_theta, _position=[0,0], _init_theta_given=False, _init_theta_cct=theta_ccr)

		for T in range(num_T):

			for t in range(num_t):

				agent_1.time_update()
				
				# EKF
				[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
				[or_error, loc_error] = agent_1.estimation_error(ekf_theta, ekf_x, ekf_y)

				err_array[theta_ccr_idx, 0, n, sample_idx] = or_error
				err_array[theta_ccr_idx, 1, n, sample_idx]= loc_error

				error_ekf[0,0] += or_error
				error_ekf[0,1] += loc_error 
				error_ekf_2[0,0] += or_error ** 2
				error_ekf_2[0,1] += loc_error ** 2


				# hybrid
				[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
				[or_error, loc_error] = agent_1.estimation_error(hybrid_theta, hybrid_x, hybrid_y)

				err_array[theta_ccr_idx, 2, n, sample_idx] = or_error
				err_array[theta_ccr_idx, 3, n,  sample_idx]= loc_error

				error_hybrid[0,0] += or_error 
				error_hybrid[0,1] += loc_error
				error_hybrid_2[0,0] += or_error ** 2
				error_hybrid_2[0,1] += loc_error ** 2

				# lie
				[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
				[or_error, loc_error] = agent_1.estimation_error(lie_theta, lie_x, lie_y)

				err_array[theta_ccr_idx, 4, n, sample_idx] = or_error
				err_array[theta_ccr_idx, 5, n, sample_idx]= loc_error

				error_lie[0,0] += or_error 
				error_lie[0,1] += loc_error
				error_lie_2[0,0] += or_error ** 2
				error_lie_2[0,1] += loc_error ** 2 

				#circular
				[circular_theta, circular_x, circular_y] = agent_1.circular_estimate.read_estimation()
				[or_error, loc_error] = agent_1.estimation_error(circular_theta, circular_x, circular_y)

				err_array[theta_ccr_idx, 6, n, sample_idx] = or_error
				err_array[theta_ccr_idx, 7, n, sample_idx]= loc_error

				error_circular[0,0] += or_error 
				error_circular[0,1] += loc_error
				error_circular_2[0,0] += or_error ** 2
				error_circular_2[0,1] += loc_error ** 2

				i = i+1
				sample_idx = sample_idx + 1


			agent_1.bd_observation_update()
			# agent_1.direct_observation_update()

			# EKF
			[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(ekf_theta, ekf_x, ekf_y)

			err_array[theta_ccr_idx, 0, n, sample_idx] = or_error
			err_array[theta_ccr_idx, 1, n, sample_idx]= loc_error

			error_ekf[0,0] += or_error
			error_ekf[0,1] += loc_error 
			error_ekf_2[0,0] += or_error ** 2
			error_ekf_2[0,1] += loc_error ** 2	
			

			# hybrid
			[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(hybrid_theta, hybrid_x, hybrid_y)

			err_array[theta_ccr_idx, 2, n, sample_idx] = or_error
			err_array[theta_ccr_idx, 3, n, sample_idx]= loc_error

			error_hybrid[0,0] += or_error 
			error_hybrid[0,1] += loc_error
			error_hybrid_2[0,0] += or_error ** 2
			error_hybrid_2[0,1] += loc_error ** 2


			# lie
			[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(lie_theta, lie_x, lie_y)

			err_array[theta_ccr_idx, 4, n, sample_idx] = or_error
			err_array[theta_ccr_idx, 5, n, sample_idx]= loc_error

			error_lie[0,0] += or_error 
			error_lie[0,1] += loc_error
			error_lie_2[0,0] += or_error ** 2
			error_lie_2[0,1] += loc_error ** 2


			#circular
			[circular_theta, circular_x, circular_y] = agent_1.circular_estimate.read_estimation()
			[or_error, loc_error] = agent_1.estimation_error(circular_theta, circular_x, circular_y)

			err_array[theta_ccr_idx, 6, n, sample_idx] = or_error
			err_array[theta_ccr_idx, 7, n, sample_idx]= loc_error
			
			error_circular[0,0] += or_error 
			error_circular[0,1] += loc_error
			error_circular_2[0,0] += or_error ** 2
			error_circular_2[0,1] += loc_error ** 2

			i = i+1
			sample_idx = sample_idx + 1

		del agent_1

	

	orientation_err[0, theta_ccr_idx] = error_ekf[0,0]/(total_sample_number*N)
	position_err[0, theta_ccr_idx] = math.sqrt(total_sample_number*N*error_ekf_2[0,1] - error_ekf[0,1]**2)/(total_sample_number*N)

	orientation_err[1, theta_ccr_idx] = error_hybrid[0,0]/(total_sample_number*N)
	position_err[1, theta_ccr_idx] = math.sqrt(total_sample_number*N*error_hybrid_2[0,1] - error_hybrid[0,1]**2)/(total_sample_number*N)

	orientation_err[2, theta_ccr_idx] = error_lie[0,0]/(total_sample_number*N)
	position_err[2, theta_ccr_idx] = math.sqrt(total_sample_number*N*error_lie_2[0,1] - error_lie[0,1]**2)/(total_sample_number*N)

	orientation_err[3, theta_ccr_idx] = error_circular[0,0]/(total_sample_number*N)
	position_err[3, theta_ccr_idx] = math.sqrt(total_sample_number*N*error_circular_2[0,1] - error_circular[0,1]**2)/(total_sample_number*N)

	output_str = '{:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4}, {:2.4} \n'.format(theta_ccr, error_ekf[0,0] / (total_sample_number*N), error_ekf[0,1] / (total_sample_number*N), \
																									error_hybrid[0,0] / (total_sample_number*N), \
																									error_hybrid[0,1] / (total_sample_number*N), error_lie[0,0] / (total_sample_number*N),\
		 																							error_lie[0,1] / (total_sample_number*N),error_circular[0,0] / (total_sample_number*N), \
		 																							error_circular[0,1] / (total_sample_number*N))
	output_init_file.write(output_str)


	theta_ccr_idx += 1

output_init_file.close()

reshaped_error_data = err_array.reshape(len(theta_ccr_arr), 8 * total_sample_number*N) 
np.savetxt('result/test_error_data.txt', reshaped_error_data)
print("data shape is:", err_array.shape)
print("Reshaped data is:", reshaped_error_data.shape)



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
plt.plot(theta_ccr_arr, orientation_err[3,:], color = plot_color['circular'], linewidth=1.6, label = 'circular')


plt.ylabel('orientation err')
plt.xscale('log') 

plt.legend()

plt.subplot(212)
plt.plot(theta_ccr_arr, position_err[0,:], color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
plt.plot(theta_ccr_arr, position_err[1,:], color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
plt.plot(theta_ccr_arr, position_err[2,:], color = plot_color['LG-EKF'], linewidth=1.6, label = 'Lie-EKF')
plt.plot(theta_ccr_arr, position_err[3,:], color = plot_color['circular'], linewidth=1.6, label = 'circular')

plt.ylabel('position err')
plt.xlabel('initial concentration parameter $\kappa_0$')
plt.xscale('log') 

plt.show()

