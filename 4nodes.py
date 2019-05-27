#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  23 15:54:50 2019

@author: kumarharsha
"""

import numpy as np
from imports import *
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

all_graphs = {}
node_list = [0,1,2,3]

all_graphs["chain1"] = nx.DiGraph()
all_graphs["chain1"].add_edges_from([
        (0,1),
        (1,2),
        (2,3)])
all_graphs["chain2"] = nx.DiGraph()
all_graphs["chain2"].add_edges_from([
        (0,1),
        (1,2),
        (3,2)])
all_graphs["chain3"] = nx.DiGraph()
all_graphs["chain3"].add_edges_from([
        (0,1),
        (2,1),
        (2,3)])
all_graphs["chain4"] = nx.DiGraph()
all_graphs["chain4"].add_edges_from([
        (1,0),
        (2,1),
        (2,3)])
all_graphs["star1"] = nx.DiGraph()
all_graphs["star1"].add_edges_from([
        (0,2),
        (1,2),
        (3,2)])
all_graphs["star2"] = nx.DiGraph()
all_graphs["star2"].add_edges_from([
        (2,0),
        (1,2),
        (3,2)])
all_graphs["star3"] = nx.DiGraph()
all_graphs["star3"].add_edges_from([
        (2,0),
        (2,1),
        (3,2)])
all_graphs["star4"] = nx.DiGraph()
all_graphs["star4"].add_edges_from([
        (2,0),
        (2,1),
        (2,3)])
all_graphs["cycle1"] = nx.DiGraph()
all_graphs["cycle1"].add_edges_from([
        (0,1),
        (1,2),
        (2,3),
        (3,0)])
all_graphs["cycle2"] = nx.DiGraph()
all_graphs["cycle2"].add_edges_from([
        (1,0),
        (1,2),
        (2,3),
        (3,0)])
all_graphs["cycle3"] = nx.DiGraph()
all_graphs["cycle3"].add_edges_from([
        (1,0),
        (0,3),
        (1,2),
        (2,3)])
all_graphs["cycle4"] = nx.DiGraph()
all_graphs["cycle4"].add_edges_from([
        (1,0),
        (1,2),
        (3,0),
        (3,2)])


p = graphviz_layout(all_graphs["chain1"])

#fig, axes = plt.subplots(12,9, num=1, figsize=(35,36))
fig = plt.figure(figsize=(30,36))
i = 1
for text, g in all_graphs.items():
    # directed graph
    ax = fig.add_subplot(12, 10, i)
    ax.axis('off')
    if(i == 1):
        ax.set_title("Directed graph")
    nx.draw_networkx(g, pos=p, ax=ax)
    
    sym = symmetrised_laplacian(g, node_list)
    # orthonormal basis for range: Q
    q = sym['q']
    q[abs(q) < 1e-6] = 0
    ax = fig.add_subplot(12, 10, i+1)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$Q$")
    sns.heatmap(q, annot=True, cbar=False, ax=ax)
    
    # scores
    l = out_degree_laplacian(g, node_list)
    u, s, v = np.linalg.svd(l)
    score = u.dot(np.diag(s))
    ax = fig.add_subplot(12, 10, i+2, projection='3d')
    all_vectors = np.array([
            [0,0,0,score[0,0],score[0,1],score[0,2]],
            [0,0,0,score[1,0],score[1,1],score[1,2]],
            [0,0,0,score[2,0],score[2,1],score[2,2]]])
    x,y,z,u,v,w = zip(*all_vectors)
    ax.quiver(x,y,z,u,v,w)
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    
    # node plots
    origin = [0], [0] # origin point
    if(q.shape[0] == 3):
        ax = fig.add_subplot(12, 10, i+3, projection='3d')
        all_vectors = np.array([
                [0,0,0,q[0,0],q[1,0],q[2,0]],
                [0,0,0,q[0,1],q[1,1],q[2,1]],
                [0,0,0,q[0,2],q[1,2],q[2,2]],
                [0,0,0,q[0,3],q[1,3],q[2,3]]])
        x,y,z,u,v,w = zip(*all_vectors)
        ax.quiver(x,y,z,u,v,w)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
    else:
        ax = fig.add_subplot(12, 10, i+3)
        if(q.shape[0] == 2):
            ax.quiver(*origin,
                      np.asarray(q[0,:])[0],
                      np.asarray(q[1,:])[0],
                      scale=5)
        else:
            ax.quiver(*origin,
                      np.asarray(q[0,:])[0],
                      np.repeat(0,4),
                      scale=5)
    if(i == 1):
        ax.set_title(r"Visualising $Q$")

    # symmetrised graph
    eqG = symmetrised_graph(g, node_list)
    ax = fig.add_subplot(12, 10, i+4)
    ax.axis('off')
    if(i == 1):
        ax.set_title("Symmetrised graph")
    nx.draw_networkx(
            eqG, pos=p,
            edge_color=[eqG[u][v]['color'] for u,v in eqG.edges()],
            ax=ax)
    # dot products of node vectors
    ax = fig.add_subplot(12, 10, i+5)
    ax.axis('off')
    if(i == 1):
        ax.set_title("Pairwise dot products")
    sns.heatmap(
            hypothesis1(g, node_list),
            annot=True, cbar=False, ax=ax)
    # reduced laplacian
    ax = fig.add_subplot(12, 10, i+6)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$\mathbf{\bar{L}} = QLQ^T$")
    rl = sym['red_laplacian']
    rl[abs(rl) < 1e-6] = 0
    sns.heatmap(rl,annot=True, cbar=False, ax=ax)
    # sigma
    ax = fig.add_subplot(12, 10, i+7)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$\bar{L}\mathbf{\Sigma}+\mathbf{\Sigma}\bar{L}^T = I$")
    s = sym['sigma']
    s[abs(s) < 1e-6] = 0
    sns.heatmap(s, annot=True, cbar=False, ax=ax)
    # X
    ax = fig.add_subplot(12, 10, i+8)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$\mathbf{X}=Q^T\Sigma Q$")
    x = sym['x']
    x[abs(x) < 1e-6] = 0
    sns.heatmap(x, annot=True, cbar=False, ax=ax)
    # Equivalent Laplacian
    ax = fig.add_subplot(12, 10, i+9)
    ax.axis('off')
    if(i == 1):
        ax.set_title(r"$\mathbf{\hat{L}_u}=X^\dagger$")
    eqL = sym['eq_laplacian']
    eqL[abs(eqL) < 1e-6] = 0
    sns.heatmap(eqL, annot=True, cbar=False, ax=ax)
    i = i+10

plt.savefig("4nodes.pdf")
