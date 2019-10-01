#%%
import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
import glob
#%%
class Takens:

	'''
	constant
	'''
	tau_max = 30


	'''
	initializer
	'''
	def __init__(self, data,tau=None):
		self.data           = data
		if tau is None:
			self.tau, self.nmi  = self.__search_tau()
		else:
			self.tau = tau


	'''
	reconstruct data by using searched tau
	'''
	def reconstruct(self):
		_data1 = self.data[:-2]
		_data2 = np.roll(self.data, -1 * self.tau)[:-2]
		_data3 = np.roll(self.data, -2 * self.tau)[:-2]
		return np.array([_data1, _data2, _data3])


	'''
	find tau to use Takens' Embedding Theorem
	'''
	def __search_tau(self):

		# Create a discrete signal from the continunous dynamics
		hist, bin_edges = np.histogram(self.data, bins=200, density=True)
		bin_indices = np.digitize(self.data, bin_edges)
		data_discrete = self.data[bin_indices]

		# find usable time delay via mutual information
		before     = 1
		nmi        = []
		res        = None

		for tau in range(1, self.tau_max):
			unlagged = data_discrete[:-tau]
			lagged = np.roll(data_discrete, -tau)[:-tau]
			nmi.append(metrics.normalized_mutual_info_score(unlagged, lagged))

			if res is None and len(nmi) > 1 and nmi[-2] < nmi[-1]:
				res = tau - 1

		if res is None:
			res = 50

		return res, nmi


class Dataset:
	def __init__(self,dir_list):
		self.root_path_list = dir_list
		for dir in self.root_path_list:
			path_list = glob.glob(os.path.join(dir, "*.csv"), recursive=True)
			if len(path_list)==0:
				print(f"Cannot find any files inside {dir}")
				self.exp_path_list.extend(path_list)
        # self.exp_path_list = [p.replace('./', '') for p in self.exp_path_list if 'old' not in p]
		# print(self.exp_path_list)
		for path in self.exp_path_list:
			df = pd.read_csv(path)
			self.eigenworm_exp_list.append(np.array(df.loc[:,self.var_name]))
			[self.behavior_label_dict[k].append(df.loc[:,k].values) for k in self.behavior_label_name]
			self.stimulus_list.append(np.array(df.loc[:,'led']))
		
#%%
import itertools
import os 

var_name = ['a_1','a_2','a_3','a_4','a_5','VelocityTailToHead']

root_path_list = ['data/141_ASH_02/']
exp_path_list = []
for dir in root_path_list:
	path_list = glob.glob(os.path.join(dir, "*.csv"), recursive=True)
	if len(path_list)==0:
		print(f"Cannot find any files inside {dir}")
	exp_path_list.extend(path_list)

	dataset = {}
	for i, name in enumerate(var_name):
		eigenworm_exp_list = []
		d = []
		for path in exp_path_list:
			df = pd.read_csv(path)
			d.append(np.array(df.loc[:,name]))
		d_1D = list(itertools.chain.from_iterable(d))

		dataset[name] = d_1D

#%%
# eigenworm_exp_list =  np.array(eigenworm_exp_list)
#
fig, axs = plt.subplots(6,2, figsize=(15,30))
plt_sample_size = 30000
for i, name in enumerate(var_name):
	takens = Takens(dataset[name],tau=10)
	emd = takens.reconstruct()    
	tau = takens.tau
	
	axs[i,0].scatter(emd[0,:plt_sample_size],emd[1,:plt_sample_size],s=1,c=dataset["VelocityTailToHead"][:plt_sample_size])
	axs[i,1].plot(emd[0,:10000],emd[1,:10000],lw=0.2)
	axs[i,0].set_title(name+'\n'+'tau:'+str(tau))
	axs[i,1].set_title(name+'\n'+'tau:'+str(tau))

plt.show()

#%%
fig.savefig("tde.png",dpi=150)

#%%
position = np.array(df.loc[:,["posCentroidX","posCentroidY"]])
velocity = np.array(df.loc[:,["VelocityTailToHead"]])

#%%
# plt.plot(emd[0,:1000],emd[1,:1000],'gray',lw=0.1)
plt.scatter(emd[0,:60000],emd[2,:60000],s=1,c=dataset["VelocityTailToHead"][:60000])
plt.savefig('tde_velocity_all.png',dpi=150)


#%%
