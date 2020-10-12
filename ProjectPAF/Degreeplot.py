import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import linregress
from ProjectPAF.PAF_Generation import ba_graph_oi, ba_graph, paf_graph

n = 100000
m = 50
m0 = 100

# G = ba_graph(n, m)
G = ba_graph_oi(n, m, m0)
#G = paf_graph(n, m, m0)

degrees = Counter(dict(G.degree()).values())
deg_k = list(degrees.keys())
deg_v = list(degrees.values())

# nx.draw(G, with_labels=True)


slope, intercept, _, _, _ = linregress(np.log10(deg_k), np.log10(deg_v))

xfid = np.linspace(np.log10(min(deg_k)), np.log10(max(deg_k)))

fig, axs = plt.subplots(2)
axs[0].bar(deg_k, deg_v)
axs[1].plot(np.log10(deg_k), np.log10(deg_v), 'k.')
axs[1].plot(xfid, xfid * slope + intercept)
plt.show()
