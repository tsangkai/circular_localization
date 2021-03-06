# Localization trajectory

import numpy as np

import yaml
import matplotlib.pyplot as plt

import Agent


### import parameters

with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)


num_T = config['num_T'] 
num_t = config['num_t'] 

total_sample_number = (num_t+1) * num_T


### output data

groundtruth = np.zeros([total_sample_number, 2])

data_ekf = np.zeros([total_sample_number, 2])
data_hybrid = np.zeros([total_sample_number, 2])
data_circular = np.zeros([total_sample_number, 2])
data_lie = np.zeros([total_sample_number, 2])

output_traj_file = open("result/trajectory.txt", "w")


### simulation

agent_1 = Agent.Agent()

i = 0
for T in range(num_T):
	for t in range(num_t):

		agent_1.time_update()

		groundtruth[i,0] = agent_1.position[0]
		groundtruth[i,1] = agent_1.position[1]

		# EKF
		[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
		data_ekf[i,0] = ekf_x
		data_ekf[i,1] = ekf_y

		# hybrid
		[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
		data_hybrid[i,0] = hybrid_x
		data_hybrid[i,1] = hybrid_y

		# circular
		[circular_theta, circular_x, circular_y] = agent_1.circular_estimate.read_estimation()
		data_circular[i,0] = circular_x
		data_circular[i,1] = circular_y

		# lie
		[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
		data_lie[i,0] = lie_x
		data_lie[i,1] = lie_y

		result_str = '{:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4}\n'.format(agent_1.time, agent_1.position[0], agent_1.position[1], ekf_x, ekf_y, hybrid_x, hybrid_y, circular_x, circular_y, lie_x, lie_y)
		output_traj_file.write(result_str)

		i = i+1


	agent_1.bd_observation_update()
	agent_1.direct_observation_update() # for circular representation

	groundtruth[i,0] = agent_1.position[0]
	groundtruth[i,1] = agent_1.position[1]

	# EKF
	[ekf_theta, ekf_x, ekf_y] = agent_1.EKF_estimate.read_estimation()
	data_ekf[i,0] = ekf_x
	data_ekf[i,1] = ekf_y

	# hybrid
	[hybrid_theta, hybrid_x, hybrid_y] = agent_1.hybrid_estimate.read_estimation()
	data_hybrid[i,0] = hybrid_x
	data_hybrid[i,1] = hybrid_y

	# circular
	[circular_theta, circular_x, circular_y] = agent_1.circular_estimate.read_estimation()
	data_circular[i,0] = circular_x
	data_circular[i,1] = circular_y

	# lie
	[lie_theta, lie_x, lie_y] = agent_1.lie_estimate.read_estimation()
	data_lie[i,0] = lie_x
	data_lie[i,1] = lie_y

	result_str = '{:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4} {:2.4}\n'.format(agent_1.time, agent_1.position[0], agent_1.position[1], ekf_x, ekf_y, hybrid_x, hybrid_y, circular_x, circular_y, lie_x, lie_y)
	output_traj_file.write(result_str)

	i = i+1


del agent_1

output_traj_file.close()


### visualization

plot_color = {
	'EKF': config['color']['grenadine'],
	'LG-EKF': config['color']['mustard'],
	'hybrid': config['color']['navy'],
	'circular': config['color']['spruce']
}

plt.figure(1)

plt.plot(groundtruth[:,0], groundtruth[:,1], 'k', linewidth=1.6, label = 'groundtruth')

plt.plot(data_ekf[:,0], data_ekf[:,1], '--', color = plot_color['EKF'], linewidth=1.6, label = 'EKF')
plt.plot(data_hybrid[:,0], data_hybrid[:,1],'--',  color = plot_color['hybrid'], linewidth=1.6, label = 'hybrid')
plt.plot(data_circular[:,0], data_circular[:,1],'--',  color = plot_color['circular'], linewidth=1.6, label = 'circular')
plt.plot(data_lie[:,0], data_lie[:,1],'--',  color = plot_color['LG-EKF'], linewidth=1.6, label = 'LG-EKF')

plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.legend()
plt.savefig('result/trajectory.png')
plt.show()



