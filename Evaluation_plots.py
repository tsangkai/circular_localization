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
	'EKF':    [0.8627, 0.2980, 0.2745],  # ['grenadine']
	'LG-EKF': [0.8471, 0.6824, 0.2784],  #['mustard']
	'hybrid': [0.1333, 0.2275, 0.3686],  #config['color']['navy'],
	'circular':[0.0000, 0.3490, 0.3765], # config['color']['spruce']
}

fig_width = 7
fig_height = 7
line_width = 1.2


# error plot
trajectory_column_names = ["time", "p_x", "p_y", "ekf_x", "ekf_y", "hybrid_x", "hybrid_y", "circular_x", "circular_y", "lie_x", "lie_y"]
trajectory = pd.read_csv("result/trajectory.csv", names=trajectory_column_names)

initialization_column_names = ["theta_ccr", "or_error_ekf", "pos_error_ekf", "or_error_hybrid", "pos_error_hybrid", "or_error_lie", "pos_error_lie", "or_error_circular", "pos_error_circular"]
initial_err_data =  pd.read_csv("result/initial.csv", names=initialization_column_names)

theta_idx_size = 15
error_data = np.loadtxt('result/test_error_data.txt').reshape(theta_idx_size, 8, total_sample_number*N) 
# error_data = np.transpose(error_data,(1,0,2))
# print(error_data.shape)


def initial_plot2():
	mean = np.zeros((theta_idx_size, 8))
	standard_div = np.zeros((theta_idx_size, 8))

	#Taking meand and standard diviation
	for theta_idx in range(theta_idx_size):
		for k in range(error_data.shape[1]):
			mean[theta_idx,k] = np.mean(error_data[theta_idx, k, :])
			standard_div[theta_idx,k] = np.std(error_data[theta_idx, k, :])

	
	#initaization error plots
	fig3, (ax4, ax5) = plt.subplots(2)
	fig3.set_size_inches(fig_width, fig_height)

	ax4.plot(theta_ccr_arr, mean[:,0], '-x', color = plot_color['EKF'], linewidth=line_width, label = 'EKF') #ekf_pos_error
	ax4.fill_between(theta_ccr_arr, mean[:,0]-standard_div[:,0], mean[:,0]+standard_div[:,0], color = plot_color['EKF'], alpha = 0.5)
	ax4.plot(theta_ccr_arr, mean[:,2], '-x', color = plot_color['hybrid'], linewidth=line_width, label = 'Hybrid') #ekf_pos_error
	ax4.fill_between(theta_ccr_arr, mean[:,2]-standard_div[:,2], mean[:,2]+standard_div[:,2], color = plot_color['hybrid'], alpha = 0.5)
	ax4.plot(theta_ccr_arr, mean[:,4], '-x', color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF') #ekf_pos_error
	ax4.fill_between(theta_ccr_arr, mean[:,4]-standard_div[:,4], mean[:,4]+standard_div[:,4], color = plot_color['LG-EKF'], alpha = 0.5)
	ax4.plot(theta_ccr_arr, mean[:,6], '-x', color = plot_color['circular'], linewidth=line_width, label = 'Circular') #ekf_pos_error
	ax4.fill_between(theta_ccr_arr, mean[:,6]-standard_div[:,6], mean[:,6]+standard_div[:,6], color = plot_color['circular'], alpha = 0.5)
    
    #labeling
	ax4.set_xscale('log') 
	ax4.set(ylabel='orientation err')
	ax4.set(xlabel='initial concentration parameter $\kappa_0$')
	ax4.set_ylim(-0.5, 0.5)
	ax4.legend(loc = 1)
	
	ax5.plot(theta_ccr_arr, mean[:,1], '-x', color = plot_color['EKF'], linewidth=line_width, label = 'EKF') #ekf_pos_error
	ax5.fill_between(theta_ccr_arr, mean[:,1]-standard_div[:,1], mean[:,1]+standard_div[:,1], color = plot_color['EKF'], alpha = 0.5)
	ax5.plot(theta_ccr_arr, mean[:,3], '-x', color = plot_color['hybrid'], linewidth=line_width, label = 'Hybrid') #ekf_pos_error
	ax5.fill_between(theta_ccr_arr, mean[:,3]-standard_div[:,3], mean[:,3]+standard_div[:,1], color = plot_color['hybrid'], alpha = 0.5)
	ax5.plot(theta_ccr_arr, mean[:,5], '-x', color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF') #ekf_pos_error
	ax5.fill_between(theta_ccr_arr, mean[:,5]-standard_div[:,5], mean[:,5]+standard_div[:,5], color = plot_color['LG-EKF'], alpha = 0.5)
	ax5.plot(theta_ccr_arr, mean[:,7], '-x', color = plot_color['circular'], linewidth=line_width, label = 'Circular') #ekf_pos_error
	ax5.fill_between(theta_ccr_arr, mean[:,7]-standard_div[:,7], mean[:,7]+standard_div[:,7], color = plot_color['circular'], alpha = 0.5)
	
	#labeling
	ax5.set_xscale('log')
	ax5.set(ylabel='position err')
	ax5.set(xlabel='initial concentration parameter $\kappa_0$')
	ax5.set_ylim(0, 2)
	ax5.legend(loc = 1)
	fig3.savefig('result/initial2.png')
	


def plot_inital():
	#initaization error plots
	fig1, (ax1, ax2) = plt.subplots(2)
	fig1.set_size_inches(fig_width, fig_height)
	ax1.plot(initial_err_data['theta_ccr'], initial_err_data['or_error_ekf'], color = plot_color['EKF'], linewidth=line_width, label = 'EKF')
	ax1.plot(initial_err_data['theta_ccr'], initial_err_data['or_error_hybrid'], color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid')
	ax1.plot(initial_err_data['theta_ccr'], initial_err_data['or_error_lie'], color = plot_color['LG-EKF'], linewidth=line_width, label = 'Lie-EKF')
	ax1.plot(initial_err_data['theta_ccr'], initial_err_data['or_error_circular'], color = plot_color['circular'], linewidth=line_width, label = 'circular')
	ax1.set_xscale('log') 
	ax1.set(ylabel='orientation err')
	ax1.set(xlabel='initial concentration parameter $\kappa_0$')
	ax1.set_ylim(0, 0.1)
	ax1.legend(loc = 1)

	ax2.plot(initial_err_data['theta_ccr'], initial_err_data['pos_error_ekf'], color = plot_color['EKF'], linewidth=line_width, label = 'EKF')
	ax2.plot(initial_err_data['theta_ccr'], initial_err_data['pos_error_hybrid'], color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid')
	ax2.plot(initial_err_data['theta_ccr'], initial_err_data['pos_error_lie'], color = plot_color['LG-EKF'], linewidth=line_width, label = 'Lie-EKF')
	ax2.plot(initial_err_data['theta_ccr'], initial_err_data['pos_error_circular'], color = plot_color['circular'], linewidth=line_width, label = 'circular')
	ax2.set_xscale('log')
	ax2.set(ylabel='position err')
	ax2.set(xlabel='initial concentration parameter $\kappa_0$')
	ax2.set_ylim(0, 2.1)
	ax2.legend(loc = 1)
	fig1.savefig('result/initial.png')

def plot_trajectory():
	#Trajectory plot
	fig2, ax3 = plt.subplots(1)
	ax3.plot(trajectory['p_x'], trajectory['p_y'], 'k', linewidth=line_width, label = 'groundtruth')
	ax3.plot(trajectory['ekf_x'], trajectory['ekf_y'], '-x', color = plot_color['EKF'], linewidth=line_width, label = 'EKF')
	ax3.plot(trajectory['hybrid_x'], trajectory['hybrid_y'],'-x',  color = plot_color['hybrid'], linewidth=line_width, label = 'hybrid')
	ax3.plot(trajectory['lie_x'], trajectory['lie_y'],'-x',  color = plot_color['LG-EKF'], linewidth=line_width, label = 'LG-EKF')
	plt.plot(trajectory['circular_x'], trajectory['circular_y'],'-x',  color = plot_color['circular'], linewidth=1.6, label = 'circular')

	ax3.set(xlabel='x (m)')
	ax3.set(ylabel='y (m)')
	ax3.legend(loc = 1)
	fig2.savefig('result/trajectory.png')
	plt.show()


plot_inital()
initial_plot2()
plot_trajectory()




