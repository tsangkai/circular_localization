import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import yaml

with open('config.yaml') as config_file:
	config = yaml.load(config_file, Loader=yaml.FullLoader)

N = config['num_of_trial'] 

num_T = config['num_T'] 
num_t = config['num_t'] 

total_sample_number = (num_t+1) * num_T

theta_ccr_arr = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

# plot_color = {
# 	'gt': [0, 0, 0],
# 	'ekf': [0.0000, 0.5490, 0.3765],
# 	'lie': [0.8627, 0.2980, 0.2745],
# 	'hybrid': [0.8471, 0.6824, 0.2784],
# 	'circular': [0.2633, 0.4475, 0.7086],
# }

plot_color = {
	'EKF':    [0.2633, 0.4475, 0.7086],
	'LG-EKF': [0.8471, 0.6824, 0.2784],  #['mustard']
	'hybrid': [0.8627, 0.2980, 0.2745], #config['color']['navy'],
	'circular':[0.0000, 0.5490, 0.3765] # config['color']['spruce']
}

fig_width = 5.84
fig_height = 4.38
line_width = 1.2
alpha_value = 0.4



# error plot
trajectory_column_names = ["time", "p_x", "p_y", "ekf_x", "ekf_y", "hybrid_x", "hybrid_y", "circular_x", "circular_y", "lie_x", "lie_y"]
trajectory = pd.read_csv("result/trajectory.csv", names=trajectory_column_names)

initialization_column_names = ["theta_ccr", "or_error_ekf", "pos_error_ekf", "or_error_hybrid", "pos_error_hybrid", "or_error_lie", "pos_error_lie", "or_error_circular", "pos_error_circular"]
initial_err_data =  pd.read_csv("result/initial.csv", names=initialization_column_names)

theta_idx_size = 15
error_data = np.loadtxt('result/test_error_data.txt').reshape(theta_idx_size, 8, total_sample_number*N) 
# error_data = np.transpose(error_data,(1,0,2))
# print(error_data.shape)]

print(error_data.shape)



#Trajectory plot
def plot_trajectory():
	fig, ax = plt.subplots(1)
	ax.plot(trajectory['p_x'], trajectory['p_y'], 'k', linewidth=line_width, label = 'groundtruth')
	ax.plot(trajectory['ekf_x'], trajectory['ekf_y'], '--', color = plot_color['EKF'], linewidth=line_width, label = 'EKF')
	ax.plot(trajectory['lie_x'], trajectory['lie_y'], '--', color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF')
	ax.plot(trajectory['hybrid_x'], trajectory['hybrid_y'], '--', color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid')
	plt.plot(trajectory['circular_x'], trajectory['circular_y'], '--', color = plot_color['circular'], linewidth=line_width, label = 'circular')

	ax.set(xlabel='x [m]')
	ax.set(ylabel='y [m]')
	ax.legend(loc = 'upper right', ncol=2)
	ax.set_xlim(-0.5, 0.8)
	ax.set_ylim(-0.1, 1.2)

	fig.savefig('result/trajectory.pdf')
	plt.show()


def plot_dynamics():
	dynamics = np.loadtxt('result/dynamics.txt')
	fig, (ax1, ax2) = plt.subplots(2)
	fig.set_size_inches(fig_width, fig_height)
    
	ax1.plot(dynamics[:,0], dynamics[:,1], color = plot_color['EKF'], linewidth=line_width, label = 'EKF')
	ax1.plot(dynamics[:,0], dynamics[:,5], color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF')
	ax1.plot(dynamics[:,0], dynamics[:,3], color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid')
	ax1.plot(dynamics[:,0], dynamics[:,7], color = plot_color['circular'], linewidth=line_width, label = 'circular')
	ax1.set(ylabel='orientation error')
	ax1.set_ylim(-0.01, 0.13)
	ax1.legend(loc ='upper right')

	ax2.plot(dynamics[:,0], dynamics[:,2], color = plot_color['EKF'], linewidth=line_width, label = 'EKF')
	ax2.plot(dynamics[:,0], dynamics[:,6], color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF')
	ax2.plot(dynamics[:,0], dynamics[:,4], color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid')
	ax2.plot(dynamics[:,0], dynamics[:,8], color = plot_color['circular'], linewidth=line_width, label = 'circular')
	ax2.set_ylim(top=1.7)
	ax2.set(ylabel='position error [m]')
	ax2.set(xlabel='time [s]')
	fig.savefig('result/dynamics.pdf')
	plt.show()




def plot_initial(with_std=False):
	mean = np.zeros((theta_idx_size, 8))
	standard_div = np.zeros((theta_idx_size, 8))

	#Taking meand and standard diviation
	for theta_idx in range(theta_idx_size):
		for k in range(error_data.shape[1]):
			mean[theta_idx,k] = np.mean(error_data[theta_idx, k, :])
			standard_div[theta_idx,k] = np.std(error_data[theta_idx, k, :])

	
	#initaization error plots
	fig, (ax1, ax2) = plt.subplots(2)
	fig.set_size_inches(fig_width, fig_height)

	if with_std:
		ax1.fill_between(theta_ccr_arr, mean[:,0]-0.5*standard_div[:,0], mean[:,0]+0.5*standard_div[:,0], color = plot_color['EKF'], alpha = alpha_value,
				  linewidth=0.0)
		ax1.fill_between(theta_ccr_arr, mean[:,4]-0.5*standard_div[:,4], mean[:,4]+0.5*standard_div[:,4], color = plot_color['LG-EKF'], alpha = alpha_value,
				  linewidth=0.0)
		ax1.fill_between(theta_ccr_arr, mean[:,2]-0.5*standard_div[:,2], mean[:,2]+0.5*standard_div[:,2], color = plot_color['hybrid'], alpha = alpha_value,
				  linewidth=0.0)
		ax1.fill_between(theta_ccr_arr, mean[:,6]-0.5*standard_div[:,6], mean[:,6]+0.5*standard_div[:,6], color = plot_color['circular'], alpha = alpha_value,
				  linewidth=0.0)

	ax1.plot(theta_ccr_arr, mean[:,0], '-x', color = plot_color['EKF'], linewidth=line_width, label = 'EKF')
	ax1.plot(theta_ccr_arr, mean[:,4], '-x', color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF')
	ax1.plot(theta_ccr_arr, mean[:,2], '-x', color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid')
	ax1.plot(theta_ccr_arr, mean[:,6], '-x', color = plot_color['circular'], linewidth=line_width, label = 'circular')
    
    #labeling
	ax1.set_xscale('log') 
	ax1.set(ylabel='orientation error')
	ax1.legend(loc = 'upper right')

	if with_std:
		ax2.fill_between(theta_ccr_arr, mean[:,1]-0.5*standard_div[:,1], mean[:,1]+0.5*standard_div[:,1], color = plot_color['EKF'], alpha = alpha_value,
				linewidth=0.0)
		ax2.fill_between(theta_ccr_arr, mean[:,5]-0.5*standard_div[:,5], mean[:,5]+0.5*standard_div[:,5], color = plot_color['LG-EKF'], alpha = alpha_value,
				linewidth=0.0)
		ax2.fill_between(theta_ccr_arr, mean[:,3]-0.5*standard_div[:,3], mean[:,3]+0.5*standard_div[:,1], color = plot_color['hybrid'], alpha = alpha_value,
				linewidth=0.0)
		ax2.fill_between(theta_ccr_arr, mean[:,7]-0.5*standard_div[:,7], mean[:,7]+0.5*standard_div[:,7], color = plot_color['circular'], alpha = alpha_value,
				linewidth=0.0)

	ax2.plot(theta_ccr_arr, mean[:,1], '-x', color = plot_color['EKF'], linewidth=line_width, label = 'EKF') #ekf_pos_error
	ax2.plot(theta_ccr_arr, mean[:,5], '-x', color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF') #ekf_pos_error
	ax2.plot(theta_ccr_arr, mean[:,3], '-x', color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid') #ekf_pos_error
	ax2.plot(theta_ccr_arr, mean[:,7], '-x', color = plot_color['circular'], linewidth=line_width, label = 'circular') #ekf_pos_error
	
	#labeling
	ax2.set_xscale('log')
	ax2.set(ylabel='position error [m]')
	ax2.set(xlabel='initial concentration parameter $\kappa_0$')

	if with_std:
		fig.savefig('result/initial_std.pdf')
	else:
		fig.savefig('result/initial.pdf')

	plt.show()

def plot_last_of_initals():
	#initaization error plots
	fig, (ax1, ax2) = plt.subplots(2)
	fig.set_size_inches(fig_width, fig_height)

	ax1.plot(theta_ccr_arr, error_data[:,0,-1], '-x', color = plot_color['EKF'], linewidth=line_width, label = 'EKF')
	ax1.plot(theta_ccr_arr, error_data[:,4,-1], '-x', color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF')
	ax1.plot(theta_ccr_arr, error_data[:,2,-1], '-x', color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid')
	ax1.plot(theta_ccr_arr, error_data[:,6,-1], '-x', color = plot_color['circular'], linewidth=line_width, label = 'circular')
    
    #labeling
	ax1.set_xscale('log') 
	ax1.set(ylabel='orientation error')
	ax1.legend(loc = 'upper right')

	
	ax2.plot(theta_ccr_arr, error_data[:,1,-1], '-x', color = plot_color['EKF'], linewidth=line_width, label = 'EKF') #ekf_pos_error
	ax2.plot(theta_ccr_arr, error_data[:,5,-1], '-x', color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF') #ekf_pos_error
	ax2.plot(theta_ccr_arr, error_data[:,3,-1], '-x', color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid') #ekf_pos_error
	ax2.plot(theta_ccr_arr, error_data[:,7,-1], '-x', color = plot_color['circular'], linewidth=line_width, label = 'circular') #ekf_pos_error
	
	#labeling
	ax2.set_xscale('log')
	ax2.set(ylabel='position error [m]')
	ax2.set(xlabel='initial concentration parameter $\kappa_0$')

	fig.savefig('result/last_pos_of_initals.pdf')

	plt.show()


# plot_trajectory()
# plot_dynamics()
# plot_initial(False)
plot_last_of_initals()




