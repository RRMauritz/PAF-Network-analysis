import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from collections import Counter, OrderedDict
from scipy.spatial.distance import jensenshannon


# For the ER graphs I used the networkx package: https://networkx.github.io/

def ER_tri(n, l, I):
    """"
    Creates I ER(l/n) graphs and stores for each ER graph the number of triangles
    :returns a list of I elements with list[i] = number of triangles in the ith ER graph
    """
    p = l / n
    tri_list = np.zeros(I)

    for i in range(I):
        G = nx.fast_gnp_random_graph(n, p)                  # Create ER(n,p) graph
        tri_list[i] = sum(nx.triangles(G).values()) / 3     # Count the number of triangles in the graph and store it

    return tri_list


# Plots ----------------------------------------------------------------------------------------------------------------
l = 1.2
I = 2000
# Run the above method for 4 different values of n, each with the same value of l and I
tri_list1 = ER_tri(10, l, I)
tri_list2 = ER_tri(100, l, I)
tri_list3 = ER_tri(1000, l, I)
tri_list4 = ER_tri(10000, l, I)

tri_list = [tri_list1, tri_list2, tri_list3, tri_list4]

fig, axs = plt.subplots(2, 2)
axs = axs.ravel()

# Make 4 subplots, each containing the histogram and poisson pmf
# What's more, each plot contains the JSD divergence as a measure for closeness
for t in range(4):
    x = np.arange(max(tri_list[t])+1)
    mu = (l ** 3) / 6
    rv = poisson(mu)
    axs[t].plot(x, poisson.pmf(x, mu), 'ro', ms=8, label='poisson pmf')
    axs[t].vlines(x, 0, poisson.pmf(x, mu), colors='r', lw=5, alpha=0.5)

    tri_count = OrderedDict(sorted(Counter(tri_list[t]).items()))
    tri_list_prob = [e / sum(tri_count.values()) for e in tri_count.values()]
    axs[t].bar(tri_count.keys(), tri_list_prob, label='simulation frequency')
    axs[t].set_title('n=%i, JSD = %5.4f' % (10 ** (t+1), jensenshannon(tri_list_prob,poisson.pmf(x, mu))))

handles, labels = axs[3].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
plt.show()


