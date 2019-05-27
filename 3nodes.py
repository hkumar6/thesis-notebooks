#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 00:12:38 2019

@author: kumarharsha
"""

import numpy as np
from imports import symmetrised_graph
from imports import hypothesis1
from imports import symmetrised_laplacian
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

all_graphs = {}
node_list = [0,1,2]

all_graphs["cycle1"] = nx.DiGraph()
all_graphs["cycle1"].add_edges_from([
        (0,1),
        (1,2),
        (2,0)])

all_graphs["cycle2"] = nx.DiGraph()
all_graphs["cycle2"].add_edges_from([
        (0,1),
        (2,0),
        (2,1)])

all_graphs["chain1"] = nx.DiGraph()
all_graphs["chain1"].add_edges_from([
        (0,1),
        (1,2)])

all_graphs["chain2"] = nx.DiGraph()
all_graphs["chain2"].add_edges_from([
        (0,1),
        (2,1)])

all_graphs["chain3"] = nx.DiGraph()
all_graphs["chain3"].add_edges_from([
        (1,0),
        (1,2)])

p = graphviz_layout(all_graphs["cycle1"])

fig = plt.figure(figsize=(30,15))
i = 1
for text, g in all_graphs.items():
    # directed graph
    ax = fig.add_subplot(5, 9, i)
    ax.axis('off')
    if(i == 1):
        ax.set_title("Directed graph")
    nx.draw_networkx(g, pos=p, ax=ax)

    sym = symmetrised_laplacian(g, node_list)
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
    eqG = symmetrised_graph(g, node_list)
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
            hypothesis1(g, node_list),
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
