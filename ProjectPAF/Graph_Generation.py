import networkx as nx
import numpy as np
from networkx.utils import py_random_state


def _random_subset(seq, m, rng):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets


@py_random_state(2)
def ba_graph_oi(n, m, seed=None):
    """
    OWN IMPLEMENTATION: use model from the CN book, page 256 (Bollobás & Riordan 2004)
    Here we use that delta = 0 (the Albert Barabási model)
    :param n = the number of edges that we end the graph with (equivalent is the time when we end the process)
    :param m = the number of links each new vertex makes with the network
    Note 1: this algorithm allows for self-loops and multi-edges!
    Note 2: instead of using a probability vector, we use an occurrence list. This means that if we have two particles
    where particle 1 has twice as high probability to be drawn than particle 2 (p = [2/3, 1/3]), particle 1 will occur
    twice as often in this occurrence list than particle 2
    """

    # Start with 1 node, m self-loops, a self-loop adds 2 to its degree
    G = nx.MultiGraph()
    G.add_edges_from(zip([0] * m, [0] * m))
    # Add the first node (label = 0) 2*m times to the repeated-degree list:
    rep_nodes = [0] * m * 2
    # Set the source to 1 as we only have one vertex yet
    source = 1

    while source < n:
        # Loop over the m targets in each time step (intermediate scaling)
        for e in range(m):
            # this list adds the source each time new links is created -> p = D_(t+1)(e-1, t)+1 for connecting to itself
            rep_nodes_inbetw = rep_nodes[:]  # make hardcopy
            rep_nodes_inbetw.append(source)
            # Sample a target vertex from the rep_nodes_inbetw list that contains the extra source vertex
            target = list(_random_subset(rep_nodes_inbetw, 1, seed))[0]  # TODO: check if this can be more efficient
            # Add an edge between the source and the target
            G.add_edges_from([(source, target)])
            # Add the source and vertex to the rep_nodes list as both their degree has increased by 1
            rep_nodes.extend([source, target])
        source += 1
    return G


@py_random_state(2)
def ba_graph(n, m, seed=None):
    """"
    Networkx implementation of Albert Barabási algorithm
    """
    # Start with graph of m nodes, 0 edges
    G = nx.empty_graph(m)
    targets = list(range(m))
    repeated_nodes = []
    source = m
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1
    return G


def paf_graph(n):
    """"
    Implementation of the Albert Barabási graph with fitness
    :param n = the total vertices after the network simulation is over
    """

    # Start with a single vertex and self-loop on it
    G = nx.MultiGraph()
    G.add_edges_from(zip([0], [0]))
    # Sample n fitness parameters corresponding to the (future) vertices
    fitnesses = np.random.binomial(10, 0.3, size=n) + 1 # TODO

    # list of degrees,
    deg = list(dict(G.degree()).values())
    # The scaled degrees that we use as distribution to sample targets from the existing nodes
    sc_deg = [deg[0] * fitnesses[0]]
    # Scale it another time to make it a prob. distribution
    sum_sc_deg = sum(sc_deg)
    sc_deg = [e / sum_sc_deg for e in sc_deg]
    # We start with the target being vertex 0
    target = 0
    # The new entering vertex, starting with label m0 (as Python is 0-based)
    source = 1
    while source < n:
        # Add edges from the source to the the m targets
        G.add_edge(source, target)
        sc_deg[target] += 1 / fitnesses[target]
        sc_deg.append(1 / fitnesses[source])
        # Scale the degrees again to make it a prob. distribution. (here we can maybe improve)
        sum_sc_deg = sum(sc_deg)
        sc_deg = [e / sum_sc_deg for e in sc_deg]
        # Sample m target nodes from the existing vertices with sc_deg as distribution
        target = np.random.choice(np.arange(len(sc_deg)), p=sc_deg, size=1, replace=False)[0]
        source += 1
    return G, fitnesses
