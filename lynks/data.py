import networkx as nx
import numpy as np
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)

EDG = "edg"
SRC = "src"
TGT = "tgt"
EDGE_LABEL = "collaborated"
WEIGHT = "weight"
YEAR = "year"


def build_graph(triples, weights, years) -> nx.MultiGraph:
    M = nx.MultiGraph()

    attributes = np.concatenate(([weights], [years]), axis=0)
    edges_w_attributes = np.concatenate((triples, attributes.T), axis=1)

    for item in edges_w_attributes:
        M.add_edge(item[0], item[2], relation=item[1], weight=item[3], year=item[4])

    # turn into simple graph
    G = nx.Graph()
    for u, v, data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def edge_data_to_df(edge_data) -> pd.DataFrame:
    triples_df = pd.DataFrame(edge_data["edge"], columns=[SRC, TGT], dtype=str)
    triples_df[EDG] = EDGE_LABEL
    triples_df[WEIGHT] = edge_data["weight"]
    triples_df[YEAR] = edge_data["year"]

    return triples_df
