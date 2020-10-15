import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import linregress
from ProjectPAF.PAF_Generation import ba_graph_oi, ba_graph, paf_graph
from networkx.drawing.nx_agraph import write_dot


n = 1000
m = 25
m0 = 30

# G = ba_graph(n, m)
G = ba_graph_oi(n, m)
#G, fitnesses = paf_graph(n, m, m0)

degrees = list(dict(G.degree()).values())  # degree values for all vertices
deg_count = Counter(degrees)
deg_k = list(deg_count.keys())  # unique degrees
deg_v = list(deg_count.values())  # number of occurences of each unique degree

#labeldict = dict(zip(range(len(degrees)), degrees))
#nx.draw(G, node_color=degrees, labels=labeldict, with_labels=True, node_size=600, cmap=plt.cm.Reds_r)

# slope, intercept, _, _, _ = linregress(np.log10(deg_k), np.log10(deg_v))
#
# xfid = np.linspace(np.log10(min(deg_k)), np.log10(max(deg_k)))
#
# fig, axs = plt.subplots(2)
# axs[0].bar(deg_k, deg_v)
# axs[0].set_xlim([0, 100])
# axs[1].plot(np.log10(deg_k), np.log10(deg_v), 'k.')
# axs[1].plot(xfid, xfid * slope + intercept)
#
# plt.show()

