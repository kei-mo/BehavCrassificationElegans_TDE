
#%%
import numpy as np
from sklearn import metrics

#%%
class Takens:

	'''
	constant
	'''
	tau_max = 100


	'''
	initializer
	'''
	def __init__(self, data):
		self.data           = data
		self.tau, self.nmi  = self.__search_tau()


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


#%%
def generate(data_length, odes, state, parameters):
    data = np.zeros([state.shape[0], data_length])

    for i in range(5000):
        state = rk4(odes, state, parameters)

    for i in range(data_length):
        state = rk4(odes, state, parameters)
        data[:, i] = state

    return data

#%%
def rk4(odes, state, parameters, dt=0.01):
    k1 = dt * odes(state, parameters)
    k2 = dt * odes(state + 0.5 * k1, parameters)
    k3 = dt * odes(state + 0.5 * k2, parameters)
    k4 = dt * odes(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

#%%
def lorenz_odes(xyz, sbr):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    sigma = sbr[0]
    beta = sbr[1]
    rho = sbr[2]
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def lorenz_generate(data_length):
    return generate(data_length, lorenz_odes, \
        np.array([-8.0, 8.0, 27.0]),  np.array([10.0, 8/3.0, 28.0]))


if __name__ == "__main__":
    data = lorenz_generate(2**10)
    takens = Takens(data[0])
    # res,nmi = takens.__search_tau()

    emd = takens.reconstruct()    
    print(emd,emd.shape)
    
    plt.plot(emd[0],emd[1])
    plt.show()
    

#%%
