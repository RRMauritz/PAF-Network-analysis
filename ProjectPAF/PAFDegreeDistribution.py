from ProjectPAF.Graph_Generation import paf_graph
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx


def deg_compare_PAF(n, plot=False):
    """"

    """
    G, fitness = paf_graph(n)

    degrees = list(dict(G.degree()).values())  # list of degree values for all vertices
    deg_count = Counter(degrees)  # Count each occurrence
    deg_k = list(deg_count.keys())  # unique degrees (key)
    deg_v = list(deg_count.values())  # number of occurrences of each unique degree (value)

    deg_k, deg_v = zip(*sorted(zip(deg_k, deg_v)))  # sort the two together based on deg_k ascending
    deg_v = [v / n for v in deg_v]  # normalize the bars

    if plot:
        plt.bar(deg_k, deg_v, label='Degree frequency realization')
        plt.title('Degree sequence of PA network')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return G, fitness


n = 100
G, fitness = deg_compare_PAF(n, plot=False)
nx.draw(G)
plt.show()
