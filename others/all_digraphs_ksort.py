#!/usr/bin/env python3

#%% initialise
import networkx as nx
import numpy as np
import net_sym
import matplotlib.pyplot as plt
import pickle

#%% read data
n = 3
old_list = pickle.load(
    open("/Users/kumarharsha/thesis/graph_data/digraph_sym_"+str(n)+".pkl", "rb"))
new_list = {}


#%% make new list
for e in old_list.keys():
    for g in old_list[e]:
        k = n - np.linalg.matrix_rank(
            net_sym.out_degree_laplacian(
                g['digraph']))
        if not k in new_list:
            new_list[k] = []
        new_list[k].append(g)
pickle.dump(new_list, open("/Users/kumarharsha/thesis/graph_data/digraph_ksort_"+str(n)+".pkl", 'wb'))