# coding: utf-8

# ## Canard orbits
# 
# #### The van der Pol equation
# 
# \begin{aligned}
# \dot { x } & = x - \frac { x ^ { 3 } } { 3 } +  u\\
# \dot { u } & = \varepsilon(\lambda - x )
# \end{aligned}
# 
# Canard explosion for $\varepsilon = 0.05$ and $\lambda \approx 0.993491$

# #### Folded singularity
# 
# The fold point $(x_0,y_0) = (0,0)$ at $\lambda_0 = 0$ satisfies
# 
# \begin{align}
# f ( 0,0,0,0 ) &=& 0\\
# f _ { x } ( 0,0,0,0 ) &=& 0\\
# f _ { x x } ( 0,0,0,0 ) &=& 2 \neq 0\\
# f _ { u } ( 0,0,0,0 ) &=& -1 \neq 0
# \end{align}
# 
# This singularity is *generic* because
# \begin{align}
# g_x(0,0,0,0) &=& 1 \neq 0\\
# g_\lambda (0,0,0,0) &=& -1 \neq 0
# \end{align}
# 
# 
# For a maximal canard, $\lambda_c (\sqrt{\varepsilon}) = -\frac{\varepsilon}{8} + O(\varepsilon^{\frac{3}{2}})$.


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
from matplotlib.colors import Normalize
import numpy as np
from scipy.integrate import solve_ivp
import pylab


#eps = 0.05
#l = 0.99374
#f_t_vdp = lambda t,x: np.array([x[0]-(x[0]**3)/3+x[1], eps*(l-x[0])])


#res_vdp = solve_ivp(fun=f_t_vdp, t_span=[0,100], y0=np.array([2.0,1.0]))


x_range = np.arange(-2.5,2.5,0.01)

N = 3
#g = nx.complete_graph(N, create_using=nx.DiGraph)
#g = nx.DiGraph()
#g.add_edges_from(((0,1),(0,2)))
g = nx.cycle_graph(N, create_using=nx.DiGraph)
A = nx.adjacency_matrix(g)
D = np.diag(np.asarray(np.sum(nx.adjacency_matrix(g), axis=1)).reshape(-1))



L = D - A


s_xu = lambda x,u: 2*x - np.power(x,3)/3 + u
phi_xu = lambda x,u: s_xu(x,u) - x


eps = 0.05
l = 0.992559
step = 1e-6
init = np.append(2.0 + np.random.rand(N)/1000, 1.0)

#np.savetxt("canard-tests"+str(N)+"/init.log", init, delimiter=',')
#while l < 0.999:
f_t = lambda t,x:np.append(-D.dot(x[:-1]) + A.dot(s_xu(x[:-1],x[-1])), eps*(l - np.mean(x[:-1])))
res2 = solve_ivp(fun=f_t, t_span=[0,100], y0=init, method='BDF')
plt.figure()
plt.plot(x_range, -phi_xu(x_range,0), color='black')
plt.plot(res2.y[0,:], res2.y[-1,:], '-', color='red')
plt.xlabel('x')
plt.ylabel('u')
plt.title(["{:.9f}".format(t) for t in init])
#    plt.savefig("canard-tests"+str(N)+"/"+str(l)+".png")
#    plt.close()
#    l = l + step
