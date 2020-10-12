import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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


@py_random_state(3)
def ba_graph_oi(n, m, m0, seed=None):
    """
    OWN IMPLEMENTATION
    Returns a random graph according to the Barabási–Albert preferential
    Attachment model.
    :param n: the number of nodes that we end with
    :param m: the number of edges with which each incoming node is attached to the existing nodes
    Because each connection is to a different node (sampling without replacement), we could also say that m is the
    number of existing nodes to which a new vertex is connected
    :param m0: the number of nodes that we start the network with
    We should have that m<m0
    """
    if not m <= m0:
        raise ValueError("m should be less or equal to m0!")

    # Add m initial nodes
    G = nx.complete_graph(m0)  # TODO: can we create random non-complete graphs?
    # Target nodes for new edges, initialized to m random nodes from the first m0 nodes
    targets = list(np.random.choice(m0, m, replace=False))
    # List of existing nodes, with nodes repeated by the number of it's degree
    degrees = list(dict(G.degree()).values())
    repeated_nodes = list(np.repeat(G.nodes, degrees))
    # Start adding the other n-m0 nodes. The first node is m0 (this works because of 0-based ness)
    source = m0
    while source < n:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachement)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1
    return G


@py_random_state(2)
def ba_graph(n, m, seed=None):
    """"
    Networkx implementation of Albert Barabási algorithm
    """
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


def paf_graph(n, m, m0):
    """"
    Implementation of the Albert Barabási graph with fitness
    :param n = the total vertices after the network simulation is over
    :param m = the number of vertices each new vertex connects to
    :param m0 = the number of vertices that we start the network with
    """

    # Start with a complete graph #TODO: this is not necessary, a random connected graph is maybe better
    G = nx.complete_graph(m0)
    # Sample n fitness parameters corresponding to the (future) vertices
    fitnesses = np.random.binomial(10, 0.3, size=n) + 1

    # list of degrees where deg[i] corresponds to the degree of vertex i, i = 0, 1, ..., m0 for now
    deg = list(dict(G.degree()).values())
    # The scaled degrees that we use as distribution to sample targets from the existing nodes
    # Note that by doing it this way, a higher fitness value means less quality (as we divide by it)
    sc_deg = [deg[i] / fitnesses[i] for i in range(len(deg))]
    # Scale it another time to make it a prob. distribution
    sc_deg = [e / sum(sc_deg) for e in sc_deg]
    # Sample m target nodes (without replacement) from the existing vertices with sc_deg as distribution
    targets = np.random.choice(np.arange(m0), p=sc_deg, size=m, replace=False)
    # The new entering vertex, starting with label m0 (as Python is 0-based)
    source = m0
    while source < n:
        # Add edges from the source to the the m targets
        G.add_edges_from(zip([source] * m, targets))
        # Update the degrees (2 steps):
        # Add 1 to the degree of the targets and rescale
        for v in range(m):
            sc_deg[targets[v]] += 1 / fitnesses[targets[v]]
        # Add m for the degree of the new node, this can simply be done by appending m to the degree list
        sc_deg.append(m / fitnesses[source])
        # Scale the degrees again to make it a prob. distribution. (here we can maybe improve)
        sum_sc_deq = sum(sc_deg)
        sc_deg = [e / sum_sc_deq for e in sc_deg]
        # Sample m target nodes from the existing vertices with sc_deg as distribution
        targets = np.random.choice(np.arange(len(sc_deg)), p=sc_deg, size=m, replace=False)
        source += 1
    return G
