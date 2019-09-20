#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 00:12:38 2019

@author: kumarharsha
"""

import numpy as np
import networkx as nx
import net_sym
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pickle

node_list = [0,1,2]

graph_list3 = pickle.load(
    open("/Users/kumarharsha/thesis/graph_data/digraph_ksort_3.pkl", "rb"))

all_graphs = []
for k in graph_list3.keys():
    all_graphs = all_graphs + graph_list3[k]

p = nx.circular_layout(all_graphs[0]['digraph'])

fig = plt.figure(figsize=(30,15))
i = 1
for g in all_graphs:
    # directed graph
    ax = fig.add_subplot(5, 9, i)
    ax.axis('off')
    if(i == 1):
        ax.set_title("Directed graph")
    nx.draw_networkx(g['digraph'], pos=p, ax=ax)

    sym = net_sym.symmetrised_laplacian(g['digraph'], node_list)
    # orthonormal basis for range: Q
    q = sym['q']
    q[abs(q) < 1e-6] = 0
    ax = fig.add_subplot(5, 9, i+1)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$Q$")
    sns.heatmap(q, annot=True, cbar=False, ax=ax)

    # node plots
    origin = [0], [0] # origin point
    ax = fig.add_subplot(5, 9, i+2)
    if(i == 1):
        ax.set_title(r"Visualising $Q$")
    if(q.shape[0] == 2):
        ax.quiver(*origin,
                  np.asarray(q[0,:])[0],
                  np.asarray(q[1,:])[0],
                  scale=5)
    else:
        ax.quiver(*origin,
                  np.asarray(q[0,:])[0],
                  np.repeat(0,3),
                  scale=5)

    # symmetrised graph
    eqG = g['sym']
    ax = fig.add_subplot(5, 9, i+3)
    ax.axis('off')
    if(i == 1):
        ax.set_title("Symmetrised graph")
    nx.draw_networkx(
            eqG, pos=p,
            edge_color=[eqG[u][v]['color'] for u,v in eqG.edges()],
            ax=ax)
    # dot products of node vectors
    ax = fig.add_subplot(5, 9, i+4)
    ax.axis('off')
    if(i == 1):
        ax.set_title("Pairwise dot products")
    sns.heatmap(
            net_sym.hypothesis1(g['digraph'], node_list),
            annot=True, cbar=False, ax=ax)
    # reduced laplacian
    ax = fig.add_subplot(5, 9, i+5)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$\bar{L}$")
    rl = sym['red_laplacian']
    rl[abs(rl) < 1e-6] = 0
    sns.heatmap(rl,annot=True, cbar=False, ax=ax)
    # sigma
    ax = fig.add_subplot(5, 9, i+6)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$\Sigma$")
    s = sym['sigma']
    s[abs(s) < 1e-6] = 0
    sns.heatmap(s, annot=True, cbar=False, ax=ax)
    # X
    ax = fig.add_subplot(5, 9, i+7)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$X$")
    x = sym['x']
    x[abs(x) < 1e-6] = 0
    sns.heatmap(x, annot=True, cbar=False, ax=ax)
    # Equivalent Laplacian
    ax = fig.add_subplot(5, 9, i+8)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$\hat{L}_u$")
    eqL = sym['eq_laplacian']
    eqL[abs(eqL) < 1e-6] = 0
    sns.heatmap(eqL, annot=True, cbar=False, ax=ax)
    i = i+9

plt.savefig("3nodes.pdf")
