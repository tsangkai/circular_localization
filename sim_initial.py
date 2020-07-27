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


error_ekf = np.zeros([1, 2])
error_ekf_2 = np.zeros([1, 2])

error_hybrid = np.zeros([1, 2])
error_hybrid_2 = np.zeros([1, 2])

error_lie = np.zeros([1, 2])
error_lie_2 = np.zeros([1, 2])


    
for n in range(N):

	init_theta = np.random.vonmises(0, config['unkonwn_theta_cct'])
	agent_1 = Agent.Agent(_theta=init_theta, _position=[0,0], _init_theta_given=False)

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




print('\ninitial concentration parameter')
print(config['unkonwn_theta_cct'])
print('EKF error')
print(error_ekf[0,1]/(total_sample_number*N))
print(math.sqrt(total_sample_number*N*error_ekf_2[0,1] - error_ekf[0,1]**2)/(total_sample_number*N))
print('hybrid error')
print(error_hybrid[0,1]/(total_sample_number*N))
print(math.sqrt(total_sample_number*N*error_hybrid_2[0,1] - error_hybrid[0,1]**2)/(total_sample_number*N))
print('Lie-EKF error')
print(error_lie[0,1]/(total_sample_number*N))
print(math.sqrt(total_sample_number*N*error_lie_2[0,1] - error_lie[0,1]**2)/(total_sample_number*N))


output_init_file = open("result/initial.txt", "a")

output_str = '{:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4}\n'.format(config['unkonwn_theta_cct'], error_ekf[0,0] / (total_sample_number*N), error_ekf[0,1] / (total_sample_number*N), error_hybrid[0,0] / (total_sample_number*N), error_hybrid[0,1] / (total_sample_number*N), error_lie[0,0] / (total_sample_number*N), error_lie[0,1] / (total_sample_number*N))

output_init_file.write(output_str)
output_init_file.close()



