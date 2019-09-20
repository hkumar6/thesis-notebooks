
# Fast-slow consensus system

We consider the system:

\begin{equation}
\begin{aligned}
\dot{\mathbf{x}} &= -D\mathbf{x} + A\mathbf{S}(\mathbf{x}, u) \\
\dot{u} &= -\varepsilon
\end{aligned}
\end{equation}

where $[\mathbf{S}(\mathbf{x},u)]_i = S(x_i,u)$

and $S(x,u) = 2x - \frac{x^3}{3} + u$.

Then, $\phi(x,u) = S(x,u) - x$.

Take $\varepsilon = 10^{-4}$.


```python
N = 4
g = ns.get_strongly_connected_digraph(N)
```


```python
#g = nx.complete_graph(3, create_using=nx.DiGraph())
A = nx.adjacency_matrix(g)
D = np.diag(np.asarray(np.sum(nx.adjacency_matrix(g), axis=1)).reshape(-1))
```


```python
from scipy.linalg import orth
from scipy.linalg import null_space
```


```python
s_xu = lambda x,u: 2*x - np.power(x,3)/3 + u
phi_xu = lambda x,u: x - s_xu(x,u)

x_range = np.arange(-2, 2, 0.01)
```

Integrate


```python
dt = 0.01
u_range = np.arange(0.7, 1.5, dt)
n_iter = 1000
N = nx.number_of_nodes(g)
x_n = np.zeros((np.size(u_range), N))

u = 0.5

f_t = lambda t,x:np.append(
    -D.dot(x[:-1]) + A.dot(s_xu(x[:-1],x[-1])),
    -1e-4)
res = solve_ivp(
    fun=f_t, 
    t_span=[0,8000], 
    y0=np.append(1.2 + np.random.rand(N)/10, 0.1), 
    method="BDF")
```


```python
plt.plot(x_range, phi_xu(x_range,0), color='black')
plt.plot(res.y[0,:], res.y[N,:], color='red')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Trajectory on the nullclines')
plt.grid()
```

### Periodic orbits


We now make a modification to the previous system to get periodic orbits:

\begin{equation}
\begin{aligned}
\dot{\mathbf{x}} &= -D\mathbf{x} + A\mathbf{S}(\mathbf{x}, u) \\
\dot{u} &= -\frac{\varepsilon}{N} \mathbf{1}_N^T\mathbf{x}
\end{aligned}
\end{equation}


```python
f_t = lambda t,x:np.append(
    -D.dot(x[:-1]) + A.dot(s_xu(x[:-1],x[-1])),
    -np.mean(x[:-1])*0.05)
res2 = solve_ivp(
    fun=f_t,
    t_span=[0,100],
    y0=np.append(1.2 + np.random.rand(N)/10, 0.1),
    method='BDF')
```


```python
plt.plot(x_range, phi_xu(x_range,0), color='black')
plt.plot(res2.y[0,:], res2.y[N,:], color='red')
plt.xlabel(r'$x$')
plt.ylabel(r'$u$')
plt.title("Oscillations on the consensus plane")
plt.grid()
mplt2tikz.save("fast_slow_osc.tex")
```

### Reduced dynamics

The reduced system dynamics are:

\begin{equation}
\begin{aligned}
\dot{x} &= \phi(x,u) = x - \frac{x^3}{3} + u\\
\dot{u} &= -\varepsilon x
\end{aligned}
\end{equation}


```python
red_f = lambda t,x: np.array([
    -phi_xu(x[0],x[1]),
    -x[0]*1e-4])
red_res = solve_ivp(
    fun=red_f,
    t_span=[0,40000],
    y0=[1.23,0.5],
    method="BDF")
```


```python
plt.plot(red_res.y[0], red_res.y[1])
plt.plot(x_range, phi_xu(x_range,0))
plt.xlabel(r"$x$")
plt.ylabel(r"$u$")
plt.title("Reduced system")
plt.grid(True)
mplt2tikz.save("fast_slow_reduced.tex")
```
