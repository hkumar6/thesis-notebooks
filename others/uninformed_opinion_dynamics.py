
# coding: utf-8

# ## Uninformed opinion dynamics
# 
# 
# Franci, A., & Nov, O. C. (n.d.). A Realization Theory for Bio-inspired Collective Decision-Making. Retrieved from https://arxiv.org/pdf/1503.08526v3.pdf

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from matplotlib.colors import Normalize
import numpy as np
import pylab
from net_sym import get_strongly_connected_digraph

# Network dynamics simulated for each kind:
# 
# $\dot{x_i} = -d_{ii} x_i + u\sum_{j=1,j \neq i}^N a_{ij} tanh(x_j) \\
# d_{ii} = \sum_{j=1}^n a_{ij} \\
# u = 1$
# 
# In matrix representation:
# 
# $\dot{\mathbf{x}} = -D\mathbf{x} + u A S(\mathbf{x}); S(\mathbf{x}) = tanh(\mathbf{x})$

# ### Complete network
# 
# $d_{ii} = N-1$

# In[2]:


#g = nx.complete_graph(10, create_using=nx.DiGraph())
g = get_strongly_connected_digraph(5)

# In[3]:


nx.draw_circular(g)
plt.show()


# In[4]:


#color_map = []
for i, n in g.nodes(data = True):
    n['decision'] = np.random.rand(1)[0]


# In[5]:


g.nodes(data = True)


# In[6]:


[n['decision'] for i, n in g.nodes(data = True)]


# In[7]:


nx.draw_circular(g, cmap=plt.get_cmap('jet'), vmin=0, vmax=1, node_color = [n['decision'] for i, n in g.nodes(data = True)])
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'), norm=Normalize(vmin=0, vmax=1))
sm._A = []
plt.colorbar(sm)
plt.show()


# In[8]:


dt = 0.01
u = 1
N = nx.number_of_nodes(g)

plt.colorbar(sm)

for iter in np.arange(100):
    for i, n in g.nodes(data = True):
        t1 = -(N-1)*n['decision']
        t2 = u*(np.sum(np.tanh([nbr_n['decision'] for nbr_i, nbr_n in g.nodes(data = True)])) - np.tanh(n['decision']))
        n['updated_decision'] = (t1 + t2)*dt + n['decision']
    print([n['updated_decision'] for i,n in g.nodes(data = True)])
    nx.draw_circular(g, cmap=plt.get_cmap('jet'), vmin=0, vmax=1, node_color = [n['updated_decision'] for i, n in g.nodes(data = True)])
    plt.savefig("images/complete/iter"+str(iter).zfill(2)+".png")
    for i, n in g.nodes(data = True):
        n['decision'] = n['updated_decision']


# In[9]:


[n['updated_decision'] for i,n in g.nodes(data = True)]


# In[13]:


nx.draw_circular(g, cmap=plt.get_cmap('jet'), vmin=0, vmax=1, node_color = [n['updated_decision'] for i, n in g.nodes(data = True)])
plt.colorbar(sm)
plt.show()


# In[14]:


get_ipython().system('convert -delay 5 images/complete/* images/network_complete.gif')


# In[15]:


from IPython.display import Image
Image(url='./images/network_complete.gif')


# In[16]:


nx.adjacency_matrix(g)


# ### Scale-free network

# In[19]:


er = nx.erdos_renyi_graph(directed=True, n=10, p=np.random.rand(1)[0])

nx.draw(er)
plt.show()


# In[20]:


for i, n in er.nodes(data = True):
    n['decision'] = 2*np.random.rand(1)[0] - 1


# In[21]:


nx.draw_circular(er, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1, node_color = [n['decision'] for i, n in er.nodes(data = True)])
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('jet'), norm=Normalize(vmin=-1, vmax=1))
sm._A = []
plt.colorbar(sm)
plt.show()


# In[22]:


er.nodes(data=True)


# In[23]:


A = nx.adjacency_matrix(er)


# In[24]:


D = np.diag(np.asarray(np.sum(nx.adjacency_matrix(er), axis=1)).reshape(-1))


# In[25]:


N_er = nx.number_of_nodes(er)
u = 1
dt = 0.01

plt.colorbar(sm)
for iter in np.arange(100):
    x = [n['decision'] for i, n in er.nodes(data=True)]
    dxdt = -D.dot(x) + u*A.dot(np.tanh(x))
    x_new = x + dxdt*dt
    for i, n in er.nodes(data = True):
        n['decision'] = x_new[i]
    nx.draw_circular(er, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1, node_color = [n['decision'] for i, n in er.nodes(data = True)])    
    plt.savefig("images/scale_free/iter"+str(iter).zfill(3)+".png")
    print([n['decision'] for i,n in er.nodes(data = True)])


# In[27]:


nx.draw_circular(er, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1, node_color = [n['decision'] for i, n in er.nodes(data = True)])
plt.colorbar(sm)
plt.show()


# In[28]:


get_ipython().system('convert -delay 5 -loop 0 images/scale_free/* images/network_scale_free.gif')


# In[29]:


from IPython.display import Image
Image(url='./images/network_scale_free.gif')


# ### Connected network

# In[2]:


gc = nx.scale_free_graph(10)
nx.draw_circular(gc)
plt.show()


# In[3]:


while not nx.is_strongly_connected(gc):
    e1 = np.random.randint(10)
    e2 = np.random.randint(10)
    print(e1, e2)
    gc.add_edge(e1, e2)

nx.is_strongly_connected(gc)


# In[37]:


nx.draw_circular(gc)
plt.show()


# In[40]:


for i, n in gc.nodes(data = True):
    n['decision'] = 2*np.random.rand(1)[0] - 1

nx.draw_circular(gc, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1, node_color = [n['decision'] for i, n in gc.nodes(data = True)])
plt.colorbar(sm)
plt.show()


# In[5]:


A = nx.adjacency_matrix(gc)
D = np.diag(np.asarray(np.sum(nx.adjacency_matrix(gc), axis=1)).reshape(-1))


# In[6]:


-D+A


# In[42]:


N_gc = nx.number_of_nodes(gc)
u = 1
dt = 0.01

plt.colorbar(sm)
for iter in np.arange(100):
    x = [n['decision'] for i, n in gc.nodes(data=True)]
    dxdt = -D.dot(x) + u*A.dot(np.tanh(x))
    x_new = x + dxdt*dt
    for i, n in gc.nodes(data = True):
        n['decision'] = x_new[i]
    nx.draw_circular(gc, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1, node_color = [n['decision'] for i, n in gc.nodes(data = True)])    
    plt.savefig("images/strongly_connected/iter"+str(iter).zfill(3)+".png")
    print([n['decision'] for i,n in gc.nodes(data = True)])


# In[44]:


nx.draw_circular(gc, cmap=plt.get_cmap('jet'), vmin=-1, vmax=1, node_color = [n['decision'] for i, n in gc.nodes(data = True)])
plt.colorbar(sm)
plt.show()


# In[45]:


get_ipython().system('convert -delay 5 -loop 0 images/strongly_connected/* images/network_strongly_connected.gif')


# In[1]:


from IPython.display import Image
Image(url='../images/network_strongly_connected.gif')

