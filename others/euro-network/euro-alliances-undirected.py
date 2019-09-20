#%% Initialisation
import networkx as nx
import numpy as np
import pickle
euro_alliances = {}
n = 6
pos = dict(zip(np.arange(0,n), [np.array([np.sin(t),np.cos(t)]) for t in np.linspace(0,2*np.pi,7)]))
country_labels={0:"GB", 1:"AH", 2:"Ge", 3:"It", 4:"Ru", 5:"Fr"}
draw_colored_edges = lambda g: nx.draw(g, edge_color=[g[u][v]['color'] for u,v in g.edges()],labels=country_labels, pos=pos)
graph_list = pickle.load(
    open("/Users/kumarharsha/thesis/graph_data/digraph_sym_6.pkl", "rb"))

#%% Triple alliance 1882
G = nx.Graph()
G.add_edge(0,1,weight=-1,color='r')
G.add_edge(0,4,weight=-1,color='r')
G.add_edge(0,5,weight=-1,color='r')
G.add_edge(1,2,weight=1,color='b')
G.add_edge(1,3,weight=1,color='b')
G.add_edge(1,4,weight=1,color='b')
G.add_edge(1,5,weight=-1,color='r')
G.add_edge(2,3,weight=1,color='b')
G.add_edge(2,4,weight=1,color='b')
G.add_edge(2,5,weight=-1,color='r')
G.add_edge(4,5,weight=-1,color='r')
euro_alliances['1882'] = G
draw_colored_edges(G)

#%% German-Russian lapse 1890
G = nx.Graph()
G.add_edge(0,1,weight=-1,color='r')
G.add_edge(0,4,weight=-1,color='r')
G.add_edge(0,5,weight=-1,color='r')
G.add_edge(1,2,weight=1,color='b')
G.add_edge(1,3,weight=1,color='b')
G.add_edge(1,4,weight=-1,color='r')
G.add_edge(1,5,weight=-1,color='r')
G.add_edge(2,3,weight=1,color='b')
G.add_edge(2,4,weight=-1,color='r')
G.add_edge(2,5,weight=-1,color='r')
euro_alliances['1890'] = G
draw_colored_edges(euro_alliances['1890'])


#%% Search for directed graph
for k in graph_list.keys():
    for g in graph_list[k]:
        if nx.is_isomorphic(euro_alliances['1890'], g['sym']):
            print("found!!")
            break

#%%
