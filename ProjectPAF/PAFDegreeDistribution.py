from ProjectPAF.Graph_Generation import paf_graph
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx


def deg_compare_PAF(n, plot=False):
    """
    Creates a PAF network and computes its degree sequence
    :param n: the number of vertices in the PAF graph
    :param plot: if True, then the degree sequence will be plotted via a bar plot
    """
    G, fitness,_ = paf_graph(n)

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


def competition_compare_PAF(n, lamb_not, plot=False):
    # Create an instance of the PAF graph
    G, fitness, Q = paf_graph(n)
    # For each fitness value, store the total degree
    degrees = list(dict(G.degree()).values())
    fit_deg = {}
    for i in range(len(degrees)):
        if fitness[i] in fit_deg:
            fit_deg[fitness[i]] += degrees[i]
        else:
            fit_deg[fitness[i]] = degrees[i]
    # Make a list of the fitness values and the corresponding counts
    link_k = list(fit_deg.keys())
    link_v = list(fit_deg.values())
    link_k, link_v = zip(*sorted(zip(link_k, link_v)))  # sort ascending
    link_v = [l / n for l in link_v]  # scale by n

    nu = [lamb_not * Q[j] / (lamb_not - fitness[j]) for j in range(len(Q))]
    # Make a bar plot
    if plot:
        plt.plot([i for i in range(len(Q))], nu, 'ro', label='Nu sequence')
        plt.bar(link_k, link_v, label='Scaled link count')
        plt.title('Scaled link count per fitness value')
        plt.xlabel('Fitness value')
        plt.ylabel('Scaled link count')
        plt.legend()
        plt.show()

def plot_PAF(n):
    G,fitness,_ = paf_graph(n)
    nx.draw(G, node_color = fitness, cmap=plt.cm.Reds_r)
    plt.show()

n=1000
lambnot = 10.000
#plot_PAF(n)
competition_compare_PAF(n,lambnot,plot=True)
#deg_compare_PAF(n,plot=True)