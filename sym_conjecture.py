#!/usr/bin/env python3
"""
Functions to draw figures for the symmetrization algorithm,
to test the conjecture that the symmetrization definitely 
results in negative edges for digraphs with dim(ker(L)) > 1.
"""

#%% imports
import networkx as nx
import numpy as np
import net_sym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pickle
import random
import seaborn as sns


#%% Draw symmetrised graph
def draw_sym_both(g, save_pdf=False):
    """Helper function to produce plots of the symmetrization algorithm
    
    Arguments:
        g {[type]} -- [description]
    
    Keyword Arguments:
        save_pdf {bool} -- [description] (default: {False})
    """
    n = nx.number_of_nodes(g['digraph'])
    node_colors = [
        'green' if g['digraph'].out_degree(n) > 0 else 'red' 
        for n in g['digraph'].nodes()]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12,6),
        constrained_layout=True,
        gridspec_kw={'wspace':0.05,'hspace':0.05})

    ax[0].axis('off')
    lay = nx.circular_layout(g['digraph'])
    nx.draw_networkx(
        g['digraph'],
        pos=lay,ax=ax[0],
        node_size=2000, node_color=node_colors,
        font_color='w', arrowsize=30, arrowstyle='->',font_size=36)

    # symmetrised graph
    ax[1].axis('off')
    node_colors = [
        'green' if g['digraph'].out_degree(n) > 0 else 'red' 
        for n in g['sym']['graph'].nodes()]
    draw_sym = lambda g: nx.draw_networkx(
        g,
        edge_color=[g[u][v]['color'] for u,v in g.edges()],
        pos=lay, node_size=2000,
        font_color='w',ax=ax[1],
        node_color=node_colors,
        font_size=36,width=3)
    draw_sym(g['sym']['graph'])

    ## latex output
    out_deg = [
        v for v in g['digraph'].nodes()
        if g['digraph'].out_degree(v) == 0
        ]
    str1 = str(out_deg).replace("[","$\\{").replace("]","\\}$")
    null, nG = net_sym.separate_graphs(g['sym']['eq_laplacian'])
    str2 = str(nG.edges()).replace("[","$\\{").replace("]","\\}$")
    print(str1 + " &" + str2 + "\\\\")

    # save image
    if save_pdf:
        k = n - np.linalg.matrix_rank(g['sym']['eq_laplacian'])
        plt.savefig(
            "sym_conjecture/n"+str(n)+"_k"+str(k)+".pdf",
            bbox_inches='tight',
            pad_inches=0.1,
            frameon=False)


#%% get all digraphs
for i in np.arange(4,7):
    for j in np.arange(2,i-1):
        g = pickle.load(open("sym_conjecture/n"+str(i)+"k"+str(j)+".pkl", "rb"))
        draw_sym_both(g, True)
