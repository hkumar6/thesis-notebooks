"""Create animation for beamer
"""

#%% Imports
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from matplotlib.colors import Normalize
import numpy as np
import pylab
from scipy.integrate import solve_ivp
from net_sym import get_strongly_connected_digraph


#%% Get a graph
N = 6
g = get_strongly_connected_digraph(N)
node_list = np.arange(0,N)
A = nx.adjacency_matrix(g, nodelist=node_list)
D = np.diag(np.asarray(np.sum(A, axis=1)).reshape(-1))

#%% Integrate system for saddle-node
u = 0.7
s_sn = lambda x,u: x + np.square(x) + u
f_sn = lambda t,x: -D.dot(x) + A.dot(s_sn(x,u))
s_psup = lambda x,u: (u+1)*np.tanh(x)
f_p = lambda t,x: -D.dot(x) + A.dot(s_psup(x,u))
init = 1 - 2*np.random.rand(N)
M = 100
res_p = np.zeros((N,M))
res_p[:,0] = init
dt = 0.01
for i in np.arange(1,M):
    res_p[:,i] = res_p[:,i-1] + dt*f_p(0,res_p[:,i-1])
#res = solve_ivp(fun=f_sn, t_span=[0,10], y0=init)

#%% Another set
u = -0.5
res_n = np.zeros((N,M))
res_n[:,0] = init
for i in np.arange(1,M):
    res_n[:,i] = res_n[:,i-1] + dt*f_p(0,res_n[:,i-1])

#%% Plot all graphs
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'), norm=Normalize(vmin=-1, vmax=1))
sm._A = []
for i in np.arange(0,M):
    nx.draw_circular(g, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1, node_color = res_p[:,i])
    plt.colorbar(sm)
    plt.savefig("images/sc_"+str(N)+"/pos_"+str(i).zfill(3)+".png")
    plt.close()

#%%
for i in np.arange(0,M):
    nx.draw_circular(g, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1, node_color = res_n[:,i])
    plt.colorbar(sm)
    plt.savefig("images/sc_"+str(N)+"/neg_"+str(i).zfill(3)+".png")
    plt.close()
