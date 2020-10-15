import matplotlib.pyplot as plt
from ProjectPAF.PAF_Generation import *
import numpy as np
import networkx as nx

m = 3

G = ba_graph_oi(10, m)

print(G.degree)

nx.draw(G)
plt.show()


