import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import linregress
from ProjectPAF.PAF_Generation import ba_graph_oi, ba_graph, paf_graph

n = 30
m = 3
m0 = 5

# G = ba_graph(n, m)
# G = ba_graph_oi(n, m, m0)
G = paf_graph(n, m, m0)


degrees = dict(G.degree()).values()

print(degrees)

deg_count = Counter(degrees)
deg_k = list(deg_count.keys())
deg_v = list(deg_count.values())

nx.draw(G, with_labels=True, node_color=list(degrees), cmap=plt.cm.Reds_r)

# slope, intercept, _, _, _ = linregress(np.log10(deg_k), np.log10(deg_v))
#
# xfid = np.linspace(np.log10(min(deg_k)), np.log10(max(deg_k)))
#
# fig, axs = plt.subplots(2)
# axs[0].bar(deg_k, deg_v)
# axs[0].set_xlim([0, 100])
# axs[1].plot(np.log10(deg_k), np.log10(deg_v), 'k.')
# axs[1].plot(xfid, xfid * slope + intercept)

plt.show()
