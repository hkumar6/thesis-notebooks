#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:10:36 2019

@author: kumarharsha
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Generate all digraphs for a given number of nodes (n)
# with the following properties:
# - No bi-directional edges.
# - The underlying undirected graph (L = AWA^T) is connected.
# - No isomorphic duplicates. (none found after the above conditions)

# K_n has n(n-1)/2 edges, any more will give rise to self-loops and hence
# can be ignored in the loop.


graph_list = {}
i = 0
f = open("/Users/kumarharsha/thesis/graph_data/graph4cd.txt", "r")
for l in f:
    g = nx.DiGraph()
    accept = True
    e = np.array([int(j) for j in str.split(l)])
    if not e[1] in graph_list:
        graph_list[e[1]] = []
    for k in np.arange(2, e.size, 2):
        if g.has_edge(e[k+1],e[k]):
            break
        else:
            g.add_edge(e[k],e[k+1])
    else:
        if not any(nx.is_isomorphic(g, x) for x in graph_list[e[1]]):
            graph_list[e[1]].append(g)
            print(e)
            i = i+1
        else:
            print("isomorphic graph found")
#        plt.figure()
#        nx.draw_networkx(graph_list[i])
print([(i,len(graph_list[i])) for i in graph_list])