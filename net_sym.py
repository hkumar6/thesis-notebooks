"""
Network symmetrization toolbox.
"""

import numpy as np
import networkx as nx
from scipy.linalg import solve_lyapunov
from scipy.integrate import solve_ivp

np.set_printoptions(precision=4)


def orth_matrix(A):
    """Return the orthogonal basis of the kernel of A using SVD.

    Arguments:
        A {numpy.ndarray} -- the (N,N) input matrix

    Returns:
        numpy.ndarray -- (N-k,N) where k = dim(ker(A)).
            The rows form the basis vectors of the subspace: perp(ker(A)).
    """
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[0:num, :].T.conj()
    return Q


def out_degree_laplacian(g, node_list=None):
    """Return the out-degree Laplacian of a digraph.

    Arguments:
        g {networkx.DiGraph} -- A digraph D = (V,E,W)

    Keyword Arguments:
        node_list {numpy.ndarray} -- List of nodes in V prescribing the 
            order of rows in the Laplacian L. The default behavior is 
            to take the order as the sorted list of all nodes numbered 
            0 to N. (default: np.arange(0,N))

    Returns:
        numpy.matrix -- The out degree Laplacian L
    """
    if node_list is None:
        node_list = np.arange(0, nx.number_of_nodes(g))
    A = nx.adjacency_matrix(g, nodelist=node_list)
    D = np.diag(np.asarray(np.sum(A, axis=1)).reshape(-1))
    L = D - A
    return L


def get_strongly_connected_digraph(n):
    """Return a strongly connected digraph with the specified 
    number of nodes

    Arguments:
        n {integer} -- The number of nodes required in the digraph.

    Returns:
        networkx.DiGraph -- A strongly connected digraph containing n nodes. 
            Edges are added by uniform sampling of integer pairs from [0,n],
            until the digraph is strongly connected. Multiple edges between
            any pair of nodes, and self-loops are not allowed.
    """
    g = nx.DiGraph()
    g.add_nodes_from(range(0, n))
    while not nx.is_strongly_connected(g):
        e1 = np.random.randint(n)
        e2 = np.random.randint(n)
        if e1 != e2 and (not g.has_edge(e1, e2)):
            g.add_edge(e1, e2)
    return g


def symmetrised_laplacian(g):
    """Return the results of all steps of
    the symmetrization algorithm.

    Arguments:
        g {networkx.DiGraph} -- A digraph D = (V,E,A)

    Returns:
        dict -- A dictionary with all the values involved in the 
            symmetrization algorithm. The entries are 'laplacian', 'q', 
            'red_laplacian', 'sigma', 'x' and 'eq_laplacian' of class 
            numpy.matrix; another entry is 'graph' of class networkx.Graph.
    """
    l = out_degree_laplacian(g)
    q = orth_matrix(l).T
    rL = np.matmul(q, np.matmul(l, np.transpose(q)))
    sigma = solve_lyapunov(
        rL,
        np.identity(np.linalg.matrix_rank(l)))
    x = 2*np.matmul(np.transpose(q), np.matmul(sigma, q))
    eqL = np.linalg.pinv(x)
    return {
        'laplacian': l,
        'q': q,
        'red_laplacian': rL,
        'sigma': sigma,
        'x': x,
        'eq_laplacian': eqL,
        'graph': graph_from_laplacian(eqL)}


def effective_resistance(g):
    """Return effective resistance of a weakly connected digraph.
    
    Arguments:
        g {networkx.DiGraph} -- A digraph D = (V,E,A)
    
    Returns:
        numpy.matrix -- The effective resistance of the graph
    """
    x = symmetrised_laplacian(g)['x']
    eff_res = np.zeros(x.shape)
    for i in range(eff_res.shape[0]):
        for j in range(eff_res.shape[1]):
            eff_res[i, j] = x[i, i] + x[j, j] - 2*x[i, j]
    eff_res[abs(eff_res) < 1e-6] = 0
    return eff_res


def separate_graphs(eqL):
    """Return two graphs with positive and negative edges,
    respectively.
    
    Arguments:
        eqL {numpy.matrix} -- Symmetric Laplacian matrix of a graph with
            positively and negatively weighted edges.
    
    Returns:
        tuple -- The first element is a positively weighted graph and the 
            other is negatively weighted.
    """
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


def graph_from_laplacian(laplacian):
    """Create undirected graph from input Laplacian

    Arguments:
        laplacian {numpy.matrix} -- The Laplacian (N,N) of an undirected 
            graph with N nodes.

    Returns:
        networkx.Graph -- An undirected graph whose signed Laplacian is 
            the input.
    """
    g = nx.Graph()
    for i in np.arange(0, np.shape(laplacian)[0]):
        for j in np.arange(i+1, np.shape(laplacian)[1]):
            if np.abs(laplacian[i, j]) > 1e-6:
                g.add_edge(
                    i, j,
                    weight=laplacian[i, j],
                    color='blue' if laplacian[i, j] < 0 else 'red')
    return g


def hypothesis1(g):
    """This function returns a matrix that might be related
    to the effective resistance in some way.
    
    Arguments:
        g {networkx.DiGraph} -- The input digraph
    
    Returns:
        numpy.matrix -- Each entry (i,j) in the matrix is the dot product 
            of the i and j row vectors of Q. m_{i,j} = <v_i, v_j>, 
            where Q' = [.. v_i ..].
    """
    l = out_degree_laplacian(g)
    q = orth_matrix(l).T
    m = np.zeros(l.shape)
    for i in np.arange(0, m.shape[0]):
        for j in np.arange(0, m.shape[1]):
            m[i, j] = sum(
                np.asarray(q[:, i])*np.asarray(q[:, j]))
    return m


def hypothesis2(g):
    """This function returns another matrix that might be related
    to the effective resistance in some way.
    
    Arguments:
        g {networkx.DiGraph} -- The input digraph
    
    Returns:
        numpy.matrix -- The result matrix is the sum of projections over 
            each orthogonal row vector of Q. m = sum(v_i * v_i'), 
            where Q' = [.. v_i ..]. This turns out to be the same output 
            as hypothesis1.
    """
    l = out_degree_laplacian(g)
    q = orth_matrix(l).T
    m = np.zeros(l.shape)
    for i in range(q.shape[0]):
        m = m + np.multiply(q[i, :].T, q[i, :])
    return m
