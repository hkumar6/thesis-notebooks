import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from scipy.linalg import solve_lyapunov
from scipy.integrate import solve_ivp

np.set_printoptions(precision=4)

def orth_matrix(A):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[0:num,:].T.conj()
    return Q


def out_degree_laplacian(g):
    A = nx.adjacency_matrix(g)
    D = np.diag(np.asarray(np.sum(A, axis=1)).reshape(-1))
    L = D - A
    return L

def separate_graphs(eqL):
    N = eqL.shape[0]
    posG = nx.Graph()
    for i in np.arange(0,N):
        for j in np.arange(i+1,N):
            if(eqL[i,j] < -1e-6):
                posG.add_edge(i,j)
    negG = nx.Graph()
    for i in np.arange(0,N):
        for j in np.arange(i+1,N):
            if(eqL[i,j] > 1e-6):
                negG.add_edge(i,j)
    return posG, negG