
# coding: utf-8

# ## Bifurcations of consensus on digraphs
# 
# 
# Franci, A., & Nov, O. C. (n.d.). A Realization Theory for Bio-inspired Collective Decision-Making. Retrieved from https://arxiv.org/pdf/1503.08526v3.pdf

# In[1]: Initialise

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from matplotlib.colors import Normalize
from scipy.integrate import solve_ivp
import pylab
from network2tikz import plot
import matplotlib2tikz as mplt2tikz
import net_sym as ns



# #### Complete graph
# 
# With the network dynammics:
# 
# $\dot{\mathbf{x}} = -Dx + A\cdot \mathbf{S}(\mathbf{x};u)$
# 
# Bifurcation function on the consensus manifold:
# 
# $\phi(y,u) = y - S(y,u)$

# Examples
# N = 10
# g = nx.complete_graph(N, create_using=nx.DiGraph())
# to create a graph from the conjectured set X,
#   - create an in-tree (or reverse tree)
#     g = nx.balanced_tree(1,2, create_using=nx.DiGraph).reverse()
#   - add an edge from the root to any other node
#     g.add_edge(0,1)

# In[4]:
N = 6
g = ns.get_strongly_connected_digraph(N)
layout_dict = nx.circular_layout(g)
plot(g,"strongly_connected_"+str(N)+".tex", standalone=False, layout=layout_dict)
nx.draw_networkx(g, pos=layout_dict)
plt.close()


# In[5]:
A = nx.adjacency_matrix(g)
D = np.diag(np.asarray(np.sum(nx.adjacency_matrix(g), axis=1)).reshape(-1))
circle1=plt.Circle((0,0),.5,color='r')


# ### Saddle-node
# 
# $\dot{\mathbf{x}} = -Dx + A\cdot \mathbf{S}_{sn}(\mathbf{x};u)$
# 
# $S_{sn}(x,u) = x + x^2 + u$
# 
# $S_{sn}^i(x,u) = -0.5 \pm \sqrt{x-u+\frac{1}{4}}$

# In[6]:


s_sn = lambda x,u: x + np.square(x) + u
s_sn_inv_p = lambda x,u: -0.5 + np.sqrt(np.abs(x-u+0.25))
s_sn_inv_n = lambda x,u: -0.5 - np.sqrt(np.abs(x-u+0.25))

def saddle_node_plot(u, inv):
    x_range = np.arange(-1, 1, 0.01)
    x_range_i = np.arange(u-0.25, 1, 0.01)
    phi_sn = lambda x,u: x - s_sn(x,u)
    plt.figure()
    plt.plot(x_range, s_sn(x_range, u), linestyle='-', color="blue", label=r"$S_\mathrm{sn}$")
    if inv:
        plt.plot(x_range_i, s_sn_inv_p(x_range_i, u), linestyle='-', color="red")
        plt.plot(x_range_i, s_sn_inv_n(x_range_i, u), linestyle='-', color="red", label=r'$S_\mathrm{sn}^{-1}$')
    plt.plot(x_range, x_range, linestyle='-', color='green')
    plt.ylim(-1.5, 1.5)
    plt.axhline(y=0, color="black")
    plt.axvline(x=0, color="black")
    plt.axis('off')
    s = 'n' if u < 0 else 'p' if u > 0 else '0'
    plt.savefig("bif_fold_"+s+".svg")
    plt.close()



# In[8]:
saddle_node_plot(-0.3, True)
saddle_node_plot(0, True)
saddle_node_plot(0.3, True)


# In[11]:
u_range = np.linspace(start=-0.5, stop=-0.01, num=50)
x_s = np.zeros((np.size(u_range), N))
init = np.random.rand(N)/10
# stable
for i in np.arange(0, np.size(u_range)):
    u = u_range[i]
    f_sn = lambda t,x: -D.dot(x) + A.dot(s_sn(x,u))
    res = solve_ivp(fun=f_sn, t_span=[0,600], y0=init)
    x_s[i] = res.y[:,np.shape(res.y)[1]-1]


# In[12]:


x_us = np.zeros((np.size(u_range), N))
# unstable
for i in np.arange(0, np.size(u_range)):
    u = u_range[i]
    f_sn = lambda t,x: -D.dot(x) + A.dot(s_sn_inv_p(x,u))
    res = solve_ivp(fun=f_sn, t_span=[0,600], y0=init)
    x_us[i] = res.y[:,np.shape(res.y)[1]-1]


# In[13]:


plt.plot(u_range, x_s[:,0], color="black", label="stable")
plt.plot(u_range, x_us[:,0], color="black", linestyle='--', label="unstable")
plt.axhline(y=0)
plt.axvline(x=0)
plt.axis('off')
plt.savefig("bif_fold.svg")
plt.close()


# ### Transcritical
# 
# $\dot{\mathbf{x}} = -Dx + A\cdot \mathbf{S}_t(\mathbf{x};u)$
# 
# $S_t(x,u) = (1+u)x + x^2$
# 
# $S_t^i(x,u) = -\frac{1}{2}(1+u) \pm \sqrt{x + \frac{1}{4}(1+u)}$

# In[14]:


s_t = lambda x,u: (1+u)*x + np.square(x)
s_t_inv_p = lambda x,u: -0.5*(1+u) + np.sqrt(x + 0.25*np.square(1+u))
s_t_inv_n = lambda x,u: -0.5*(1+u) - np.sqrt(x + 0.25*np.square(1+u))

def transcritical_plot(u, inv):
    x_range = np.arange(-1, 1, 0.01)
    phi_t = lambda x,u: x - s_t(x,u)
    x_range_i = np.arange(-0.25*np.square(1+u), 1, 0.01)
    plt.plot(x_range, s_t(x_range, u), linestyle='-', color="blue", label=r'$S_\mathrm{t}$')
    if inv:
        plt.plot(x_range_i, s_t_inv_p(x_range_i, u), linestyle='-', color="red")
        plt.plot(x_range_i, s_t_inv_n(x_range_i, u), linestyle='-', color="red", label=r'$S_\mathrm{t}^{-1}$')
    plt.plot(x_range, x_range, linestyle='-', color='green')
    plt.axis('off')
    plt.axhline(y=0, color="black")
    plt.axvline(x=0, color="black")
    plt.ylim(-1.5,1.5)
    s = 'n' if u < 0 else 'p' if u > 0 else '0'
    plt.savefig("bif_trans_"+s+".svg")
    plt.close()



# In[16]:
transcritical_plot(-0.8, True)
transcritical_plot(0, True)
transcritical_plot(0.8, True)

# In[19]:
dt = 0.01
u_range = np.arange(-1, 1, dt)
n_iter = 1000
N = nx.number_of_nodes(g)
x_s = np.zeros((np.size(u_range), N))

# forward time to get to stable fixed points
for i in np.arange(0,np.size(u_range)):
    u = u_range[i]
    f_t = lambda t,x:-D.dot(x) + A.dot(s_t(x,u))
    res = solve_ivp(fun=f_t, t_span=[0,30], y0=-np.random.rand(N))
    x_s[i] = res.y[:,np.shape(res.y)[1]-1]


# In[20]:


x_us = np.zeros((np.size(u_range), N))
# invert stability to get to unstable fixed points
for i in np.arange(0,np.size(u_range)):
    u = u_range[i]
    f_t = lambda t,x:-D.dot(x) + A.dot(s_t_inv_p(x,u))
    res = solve_ivp(fun=f_t, t_span=[0,30], y0=np.random.rand(N)/100)
    x_us[i] = res.y[:,np.shape(res.y)[1]-1]


# In[21]:


plt.plot(u_range, x_s[:,0], label="stable", color="black")
plt.plot(u_range, x_us[:,0], label="unstable", color="black", linestyle='--')
plt.axis('off')
plt.axvline(x=0)
plt.savefig("bif_trans.svg")
plt.close()


# ### Pitchfork
# 
# #### Subcritical pitchfork
# 
# $\dot{\mathbf{x}} = -Dx + A\cdot \mathbf{S}_{psub}(\mathbf{x};u)$
# 
# $S_{psub}(x,u) = (1+u)tan(x)$
# 
# $S_{psub}^i(x,u) = arctan(\frac{x}{1+u})$

# In[22]:


u = -0.2
s_psub = lambda x,u: (1+u)*np.tan(x)
s_psub_inv = lambda x,u: np.arctan(x/(1+u))

def pitchfork_sub_plot(u, inv):
    x_range = np.arange(-1.56, 1.56, 0.01)
    phi_p = lambda x,u: x - s_p(x,u)
    plt.plot(x_range, s_psub(x_range, u), color="blue", label=r'$S_\mathrm{psub}$')
    if inv:
        plt.plot(x_range, s_psub_inv(x_range, u), color="red", label=r'$S_\mathrm{psub}^{-1}$')
    plt.plot(x_range, x_range, linestyle='-', color='green')
    plt.ylim(-2,2)
    plt.axis('off')
    plt.axvline(x=0, color="black")
    plt.axhline(y=0, color="black")
    s = 'n' if u < 0 else 'p' if u > 0 else '0'
    plt.savefig("bif_pitch_sub_"+s+".svg")
    plt.close()


# In[24]:
pitchfork_sub_plot(-0.5,True)
pitchfork_sub_plot(0,True)
pitchfork_sub_plot(0.5,True)


# In[27]:


dt = 0.01
u_range = np.arange(-1+dt, 1, dt)
n_iter = 1000
N = nx.number_of_nodes(g)
x_us_p = np.zeros((np.size(u_range), N))
x_us_n = np.zeros((np.size(u_range), N))

for i in np.arange(0,np.size(u_range)):
    u = u_range[i]
    f_p = lambda t,x: -D.dot(x) + A.dot(s_psub_inv(x,u))
    res = solve_ivp(fun=f_p, t_span=[0,100], y0=np.random.rand(N)/10)
    x_us_p[i] = res.y[:,-1]

for i in np.arange(0,np.size(u_range)):
    u = u_range[i]
    f_p = lambda t,x: -D.dot(x) + A.dot(s_psub_inv(x,u))
    res = solve_ivp(fun=f_p, t_span=[0,100], y0=-np.random.rand(N)/10)
    x_us_n[i] = res.y[:,-1]


# In[28]:


u_range_stable = np.arange(-1+dt, -dt, dt)
x_s = np.zeros((np.size(u_range_stable), N))
for i in np.arange(0,np.size(u_range_stable)):
    u = u_range_stable[i]
    f_p = lambda t,x: -D.dot(x) + A.dot(s_psub(x,u))
    res = solve_ivp(fun=f_p, t_span=[0,30], y0=np.random.rand(N)/10)
    x_s[i] = res.y[:,-1]


# In[29]:


plt.plot(u_range, x_us_p[:,0], color="black", linestyle='--')
plt.plot(u_range, x_us_n[:,0], label="unstable", color="black", linestyle='--')
plt.plot(u_range_stable, x_s[:,0], label="stable", linestyle='-', color="black")
plt.axis('off')
plt.axvline(x=0)
plt.savefig("bif_pitch_sub.svg")
plt.close()

# #### Supercritical pitchfork
# 
# $\dot{\mathbf{x}} = -Dx + A\cdot \mathbf{S}_\mathrm{psup}(\mathbf{x};u)$
# 
# $S_\mathrm{psup}(x,u) = (1+u)tanh(x)$
# 
# $S_\mathrm{psup}^i(x,u) = arctanh(\frac{x}{1+u})$

# In[30]:


s_psup = lambda x,u: (u+1)*np.tanh(x)
s_psup_inv = lambda x,u: np.arctanh(x/(u+1))
phi_psup = lambda x,u: x - s_psup(x,u)

def pitchfork_super_plot(u, inv):
    x_range = np.arange(-2, 2, 0.01)
    plt.plot(x_range, s_psup(x_range, u), color="blue", label=r'$S_\mathrm{psup}$')
    if inv and u >= -1:
        x_range_inv = np.arange(-(1+u)+dt, (1+u)-dt, 0.01)
        plt.plot(x_range_inv, s_psup_inv(x_range_inv, u), color="red", label=r'$S_\mathrm{psup}^{-1}$')
    plt.plot(x_range, x_range, linestyle='-', color='green')
    plt.ylim(-2.5, 2.5)
    plt.axis('off')
    plt.axhline(y=0, color="black")
    plt.axvline(x=0, color="black")
    s = 'n' if u < 0 else 'p' if u > 0 else '0'
    plt.savefig("bif_pitch_sup_"+s+".svg")
    plt.close()


# In[32]:
pitchfork_super_plot(0.8,True)
pitchfork_super_plot(0,True)
pitchfork_super_plot(-0.4,True)


# In[35]:
x_s_p = np.zeros((np.size(u_range), N))
x_s_n = np.zeros((np.size(u_range), N))

for i in np.arange(0,np.size(u_range)):
    u = u_range[i]
    f_p = lambda t,x: -D.dot(x) + A.dot(s_psup(x,u))
    res = solve_ivp(fun=f_p, t_span=[0,100], y0=np.random.rand(N)/10)
    x_s_p[i] = res.y[:,-1]

for i in np.arange(0,np.size(u_range)):
    u = u_range[i]
    f_p = lambda t,x: -D.dot(x) + A.dot(s_psup(x,u))
    res = solve_ivp(fun=f_p, t_span=[0,100], y0=-np.random.rand(N)/10)
    x_s_n[i] = res.y[:,-1]


# In[36]:
u_range_unstable = np.arange(dt, 1-dt, dt)
x_us = np.zeros((np.size(u_range_unstable), N))
for i in np.arange(0,np.size(u_range_unstable)):
    u = u_range_unstable[i]
    f_p = lambda t,x: -D.dot(x) + A.dot(s_psup_inv(x,u))
    res = solve_ivp(fun=f_p, t_span=[0,30], y0=np.random.rand(N)/10)
    x_us[i] = res.y[:,-1]


# In[37]:
plt.plot(u_range, x_s_p[:,0], color="black", linestyle='-')
plt.plot(u_range, x_s_n[:,0], label="stable", color="black", linestyle='-')
plt.plot(u_range_unstable, x_us[:,0], label="unstable", linestyle='--', color="black")
plt.axis('off')
plt.axvline(x=0)
plt.savefig("bif_pitch_sup.svg")
plt.close()



#%%
