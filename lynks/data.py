from collections.abc import Callable
from copy import copy, deepcopy

from functools import reduce

import networkx as nx
import numpy as np
import pandas as pd

from .pipeline import create_pipeline
from .logging import get_logger

logger = get_logger(__name__)


def build_graph(edges, weights, years) -> nx.MultiGraph:
    M = nx.MultiGraph()

    attributes = np.concatenate(([weights], [years]), axis=0)
    edges_w_attributes = np.concatenate((edges, attributes.T), axis=1)

    for item in edges_w_attributes:
        M.add_edge(item[0], item[1], weight=item[2], year=item[3])

    # turn into simple graph
    # TODO should this be multidigraph?
    G = nx.Graph()
    for u, v, data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G
