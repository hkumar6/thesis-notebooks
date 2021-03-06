
# Bifurcations of consensus on digraphs


```python
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
```

## Network dynamics


We know from \cref{bif_fn}, that with the following network dynammics:

\begin{equation}
\dot{\mathbf{x}} = -D\mathbf{x} + A\cdot \mathbf{S}(\mathbf{x};u),
\end{equation}

we get the following bifurcation function on the consensus manifold:

\begin{equation}
\phi(y,u) = S(y,u) - y.
\end{equation}

## Choice of digraph

This notebook can be run for different types and sizes of graphs because they can be easily created using \texttt{networkx}. First, set the number of nodes.


```python
N = 6
```

We provide examples of three types of graphs to choose from.

- A complete digraph


```python
# g = nx.complete_graph(N, create_using=nx.DiGraph())
```

- A strongly connected digraph


```python
g = ns.get_strongly_connected_digraph(N)
```

- A digraph from the conjectured set $X$


```python
# g = nx.balanced_tree(1,2, create_using=nx.DiGraph).reverse()
# g.add_edge(0,1)
# N = nx.number_of_nodes(g)
```

## Plot the graph

We want to plot the graph in a circular layout but there are many different layouts available in \texttt{networkx}. The \texttt{plot} function from network2tikz exports a \texttt{networkx} graph object to \texttt{tikz} format, which is better for \LaTeX documents.


```python
layout_dict = nx.circular_layout(g)
plot(
    g,
    "strongly_connected_"+str(N)+".tex", 
    standalone=False, 
    layout=layout_dict)
nx.draw_networkx(g, pos=layout_dict)
```

## Graph matrices

The output of the \texttt{adjacency_matrix} function is not ordered by node indices but rather random. To obtain a desired row ordering of the adjacency matrix, the \texttt{nodelist} parameter is available. However, node labels are not important for this task, so we skip it.


```python
A = nx.adjacency_matrix(g)
D = np.diag(np.asarray(np.sum(A, axis=1)).reshape(-1))
```

## Transcritical bifurcation of consensus

To simulate transcritical bifurcation of consensus, we choose the appropriate normal form for the interaction function from \cref{normal-forms}.

\begin{equation}
\begin{aligned}
\dot{\mathbf{x}} &= -Dx + A\cdot \mathbf{S}_\mathrm{t}(\mathbf{x};u),\\
S_\mathrm{t}(x,u) &= (1+u)x + x^2 .
\end{aligned}
\end{equation}


```python
s_t = lambda x,u: (1+u)*x + np.square(x)
```

We want to compute the unstable points for the system using \cref{inv_stability}. So, we will use the inverse of the selected interaction function.

\begin{equation}
S_\mathrm{t}^{-1}(x,u) = -\frac{1}{2}(1+u) \pm \sqrt{x + \frac{1}{4}(1+u)}.
\end{equation}


```python
s_t_inv_p = lambda x,u: -0.5*(1+u) + np.sqrt(x + 0.25*np.square(1+u))
s_t_inv_n = lambda x,u: -0.5*(1+u) - np.sqrt(x + 0.25*np.square(1+u))
```

### Visualize the system


```python
def transcritical_plot(u, inv):
    x_range = np.arange(-1, 1, 0.01)
    phi_t = lambda x,u: x - s_t(x,u)
    x_range_i = np.arange(-0.25*np.square(1+u), 1, 0.01)
    plt.plot(
        x_range, s_t(x_range, u), 
        linestyle='-', color="blue", 
        label=r'$S_\mathrm{t}$')
    if inv:
        plt.plot(
            x_range_i, s_t_inv_p(x_range_i, u), 
            linestyle='-', color="green")
        plt.plot(
            x_range_i, s_t_inv_n(x_range_i, u), 
            linestyle='-', color="green", 
            label=r'$S_\mathrm{t}^{-1}$')
    plt.plot(x_range, x_range, linestyle='-', color='black')
    #plt.xlabel(r'$x$')
    #plt.title(r"$u="+str(u)+"$")
    plt.axis('off')
    plt.axhline(y=0)
    plt.axvline(x=0)
    #plt.legend()
    plt.ylim(-1.5,1.5)
```


```python
transcritical_plot(-0.8, True)
plt.savefig("bif_trans_n.svg")
```


```python
transcritical_plot(0.8, True)
plt.savefig("bif_trans_p.svg")
```


```python
transcritical_plot(0, True)
plt.savefig("bif_trans_0.svg")
```

### Simulation

We set up the necessary variables required to produce the bifurcation diagram.


```python
dt = 0.01
u_range = np.arange(-1, 1, dt)
n_iter = 1000
x_s = np.zeros((np.size(u_range), N))
x_us = np.zeros((np.size(u_range), N))
```

To find the stable fixed points for the range of parameters, we solve an initial value problem (IVP) using $S_\mathrm{t}$ as the interaction function.


```python
# get stable fixed points
for i in np.arange(0,np.size(u_range)):
    u = u_range[i]
    f_t = lambda t,x:-D.dot(x) + A.dot(s_t(x,u))
    res = solve_ivp(fun=f_t, t_span=[0,30], y0=-np.random.rand(N))
    x_s[i] = res.y[:,np.shape(res.y)[1]-1]
```

Now, to find the unstable fixed points, we use $S_\mathrm{t}^{-1}$ as the interaction function.


```python
# to get unstable fixed points
for i in np.arange(0,np.size(u_range)):
    u = u_range[i]
    f_t = lambda t,x:-D.dot(x) + A.dot(s_t_inv_p(x,u))
    res = solve_ivp(fun=f_t, t_span=[0,30], y0=np.random.rand(N)/100)
    x_us[i] = res.y[:,np.shape(res.y)[1]-1]
```

Plot all the points computed to get the bifurcation diagram.


```python
plt.plot(u_range, x_s[:,0], label="stable", color="black")
plt.plot(u_range, x_us[:,0], label="unstable", color="black", linestyle='--')
#plt.xlabel(r'$u$')
#plt.ylabel(r'$x^*$')
plt.axis('off')
plt.axvline(x=0)
#plt.legend()
plt.savefig("bif_transcritical.svg")
```

## Saddle-node

\begin{equation}
\begin{aligned}
    \dot{\mathbf{x}} &= -Dx + A\cdot \mathbf{S}_\text{sn}(\mathbf{x};u)\\
    S_\text{sn}(x,u) &= x + x^2 + u\\
    S_\text{sn}^{-1}(x,u) &= -0.5 \pm \sqrt{x-u+\frac{1}{4}}
\end{aligned}
\end{equation}


```python
s_sn = lambda x,u: x + np.square(x) + u
s_sn_inv_p = lambda x,u: -0.5 + np.sqrt(np.abs(x-u+0.25))
s_sn_inv_n = lambda x,u: -0.5 - np.sqrt(np.abs(x-u+0.25))
```


```python
# stable
for i in np.arange(0, np.size(u_range)):
    u = u_range[i]
    f_sn = lambda t,x: -D.dot(x) + A.dot(s_sn(x,u))
    res = solve_ivp(fun=f_sn, t_span=[0,600], y0=np.random.rand(N)/10)
    x_s[i] = res.y[:,np.shape(res.y)[1]-1]
```


```python
# unstable
for i in np.arange(0, np.size(u_range)):
    u = u_range[i]
    f_sn = lambda t,x: -D.dot(x) + A.dot(s_sn_inv_p(x,u))
    res = solve_ivp(fun=f_sn, t_span=[0,600], y0=np.random.rand(N)/10)
    x_us[i] = res.y[:,np.shape(res.y)[1]-1]
```

## Pitchfork

### Subcritical pitchfork

\begin{equation}
\begin{aligned}
\dot{\mathbf{x}} &= -Dx + A\cdot \mathbf{S}_\mathrm{psub}(\mathbf{x};u),\\
S_\mathrm{psub}(x,u) &= (1+u)\mathrm{tan}(x),\\
S_\mathrm{psub}^{-1}(x,u) &= \mathrm{arctan}(\frac{x}{1+u})
\end{aligned}
\end{equation}


```python
s_psub = lambda x,u: (1+u)*np.tan(x)
s_psub_inv = lambda x,u: np.arctan(x/(1+u))
```


```python
# get unstable points on both branches
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
```


```python
# get stable points
u_range_stable = np.arange(-1+dt, -dt, dt)
x_s = np.zeros((np.size(u_range_stable), N))
for i in np.arange(0,np.size(u_range_stable)):
    u = u_range_stable[i]
    f_p = lambda t,x: -D.dot(x) + A.dot(s_psub(x,u))
    res = solve_ivp(fun=f_p, t_span=[0,30], y0=np.random.rand(N)/10)
    x_s[i] = res.y[:,-1]
```

### Supercritical pitchfork

\begin{equation}
\begin{aligned}
\dot{\mathbf{x}} &= -Dx + A\cdot \mathbf{S}_\mathrm{psup}(\mathbf{x};u),\\
S_\mathrm{psup}(x,u) &= (1+u)\mathrm{tanh}(x),\\
S_\mathrm{psup}^{-1}(x,u) &= \mathrm{arctanh}(\frac{x}{1+u})
\end{aligned}
\end{equation}


```python
s_psup = lambda x,u: (u+1)*np.tanh(x)
s_psup_inv = lambda x,u: np.arctanh(x/(u+1))
```


```python
# get stable points on both branches
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
```


```python
# get unstable points
for i in np.arange(0,np.size(u_range_unstable)):
    u = u_range_unstable[i]
    f_p = lambda t,x: -D.dot(x) + A.dot(s_psup_inv(x,u))
    res = solve_ivp(fun=f_p, t_span=[0,30], y0=np.random.rand(N)/10)
    x_us[i] = res.y[:,-1]
```
