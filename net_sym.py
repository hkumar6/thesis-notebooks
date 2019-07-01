import numpy as np
import networkx as nx
#from scipy.linalg import null_space
from scipy.linalg import solve_lyapunov
from scipy.integrate import solve_ivp

np.set_printoptions(precision=4)

def orth_matrix(A, clean=False):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[0:num, :].T.conj()
    return Q


def out_degree_laplacian(g, node_list=None):
    A = nx.adjacency_matrix(g, nodelist=node_list)
    D = np.diag(np.asarray(np.sum(A, axis=1)).reshape(-1))
    L = D - A
    return L

def symmetrised_laplacian(g, node_list=None):
    l = out_degree_laplacian(g, node_list)
    q = orth_matrix(l).T
    rL = np.matmul(q, np.matmul(l, np.transpose(q)))
    sigma = solve_lyapunov(rL, np.identity(np.linalg.matrix_rank(l)))
    x = 2*np.matmul(np.transpose(q), np.matmul(sigma, q))
    eqL = np.linalg.pinv(x)
    return {
        'laplacian':l,
        'q':q,
        'red_laplacian':rL,
        'sigma':sigma,
        'x':x,
        'eq_laplacian':eqL}

def effective_resistance(g, node_list=None):
    x = symmetrised_laplacian(g, node_list)['x']
    eff_res = np.zeros(x.shape)
    for i in range(eff_res.shape[0]):
        for j in range(eff_res.shape[1]):
            eff_res[i, j] = x[i, i] + x[j, j] - 2*x[i, j]
    eff_res[abs(eff_res) < 1e-6] = 0
    return eff_res

def separate_graphs(eqL):
    N = eqL.shape[0]
    posG = nx.Graph()
    for i in np.arange(0, N):
        for j in np.arange(i+1, N):
            if(eqL[i, j] < -1e-6):
                posG.add_edge(i, j)
    negG = nx.Graph()
    for i in np.arange(0, N):
        for j in np.arange(i+1, N):
            if(eqL[i, j] > 1e-6):
                negG.add_edge(i, j)
    return posG, negG

def symmetrised_graph(g, node_list=None):
    eqL = symmetrised_laplacian(g, node_list)['eq_laplacian']
    eqG = nx.Graph()
    for i in np.arange(0, np.shape(eqL)[0]):
        for j in np.arange(i+1, np.shape(eqL)[1]):
            if np.abs(eqL[i, j]) > 1e-6:
                eqG.add_edge(
                    i, j,
                    weight=eqL[i, j],
                    color='blue' if eqL[i, j] < 0 else 'red')
    return eqG

def hypothesis1(g, node_list=None):
    l = out_degree_laplacian(g, node_list)
    q = orth_matrix(l).T
    test_matrix = np.zeros(l.shape)
    for i in np.arange(0, test_matrix.shape[0]):
        for j in np.arange(0, test_matrix.shape[1]):
            test_matrix[i, j] = sum(np.asarray(q[:, i])*np.asarray(q[:, j]))
    return test_matrix

def hypothesis2(g, node_list=None):
    l = out_degree_laplacian(g, node_list)
    q = orth_matrix(l).T
    test_matrix = np.zeros(l.shape)
    for i in range(q.shape[0]):
        test_matrix = test_matrix + np.multiply(q[i, :].T, q[i, :])
    return test_matrix

def get_strongly_connected_digraph(n):
    #g = nx.scale_free_graph(n)
    g = nx.DiGraph()
    g.add_nodes_from(range(0, n))
    #for node in g.nodes():
    #    if g.has_edge(node,node):
    #        g.remove_edge(node,node)
    while not nx.is_strongly_connected(g):
        e1 = np.random.randint(n)
        e2 = np.random.randint(n)
        if e1 != e2 and (not g.has_edge(e1, e2)):
            g.add_edge(e1, e2)
    return g
