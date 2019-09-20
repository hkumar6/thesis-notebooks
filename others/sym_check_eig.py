#%% Check that the eigenvectors of 
# L(D_\textup{s}) and L(G_\textup{s}) are same

import networkx as nx
import numpy as np
import net_sym as ns

#%% Get a strongly connected digraph
N = 5
d = ns.get_strongly_connected_digraph(N)
# symmetrization
sym = ns.symmetrised_laplacian(d)
# check eigenvectors of laplacians
VD = np.linalg.eig(sym['laplacian'])
VU = np.linalg.eig(sym['eq_laplacian'])

#%% Pick a weakly connected digraph
