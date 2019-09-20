#!/usr/bin/env python3
"""
Enumeration of digraphs

Generate all digraphs for a given number of nodes (n)
with the following properties:
- No bi-directional edges.
- The underlying undirected graph (L = AWA^T) is connected.
- No isomorphic duplicates. (none found after the above 
    conditions)

K_n has n(n-1)/2 edges, any more will give rise to self-loops 
and hence can be ignored in the loop.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import net_sym
import pickle

graph_list = {}
n = 6
f = open(
    "graph_data/graph"+str(n)+"cd.txt",
    "r")
for l in f:
    # initialise a graph for every line
    g = nx.DiGraph()
    accept = True
    # convert line of text to int array
    e = np.array([int(j) for j in str.split(l)])
    # add edges
    for k in np.arange(2, e.size, 2):
        if g.has_edge(e[k+1],e[k]):
            break
        else:
            g.add_edge(e[k],e[k+1])
    # key = dim(ker(L))
    r = n - np.linalg.matrix_rank(
        net_sym.out_degree_laplacian(g))
    # check if the corresponding key exists
    if not r in graph_list:
        graph_list[r] = []
    # add the graph to list
    graph_list[r].append({
        'digraph':g,
        'sym':net_sym.symmetrised_laplacian(g)
    })

# display results: (#edges, #graphs)
print([(i,len(graph_list[i])) for i in graph_list])

# save results
pickle.dump(
    graph_list,
    open(
        "graph_data/digraph_sym_"+str(n)+".pkl",
        'wb'))
