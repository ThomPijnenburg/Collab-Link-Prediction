from collections.abc import Callable
from copy import copy, deepcopy

from functools import reduce

import networkx as nx
import numpy as np
import pandas as pd

from .pipeline import create_pipeline


class Dataset(object):
    def __init__(self, lpdata: dict):
        print("I am custom you know")

        self.raw_edges = lpdata["edge"]

        self.edge_weight_label = "weight"
        self.edge_year_label = "year"
        self.edge_attributes = {
            self.edge_weight_label: lpdata[self.edge_weight_label],
            self.edge_year_label: lpdata[self.edge_year_label]
        }
        self.graph = None

        self.edges = self.raw_edges
        self.labels = None
        self.features = []
        self.feature_names = []


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def split_dataset(dataset: Dataset, n_slices: int) -> list:
    edges = dataset.edges
    n_edges = len(edges)
    split_edges = split_list(edges, n_slices)
    result = []

    for split in split_edges:
        dataset_copy = deepcopy(dataset)
        dataset_copy.edges = split
        result.append(dataset_copy)

    return result


def merge_datasets(datasets: list) -> Dataset:

    def lambda_merge(d0, d1):
        d0.edges = d0.edges + d1.edges
        d0.features = np.concatenate((d0.features, d1.features), axis=0)
        return d0

    return reduce(lambda_merge, datasets)


def _build_graph(edges, weights, years) -> nx.MultiGraph:
    M = nx.MultiGraph()

    attributes = np.concatenate(([weights],[years]), axis=0)
    edges_w_attributes = np.concatenate((edges, attributes.T), axis=1)

    for item in edges_w_attributes:
        M.add_edge(item[0], item[1], weight=item[2], year=item[3])

    # turn into simple graph

    G = nx.Graph()
    for u,v,data in M.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if G.has_edge(u,v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


def create_graph_builder() -> Callable:
    def build_graph(dataset: Dataset) -> Dataset:
        edges = dataset.edges
        weights = dataset.edge_attributes[dataset.edge_weight_label]
        years = dataset.edge_attributes[dataset.edge_year_label]

        dataset.graph = _build_graph(edges=edges, weights=weights, years=years)
        dataset.edges = list(dataset.graph.edges)
        return dataset
    return build_graph


def create_formatter() -> Callable:
    def formatter(edge_object) -> Dataset:
        # could do something custom here
        dataset = Dataset(lpdata=edge_object)

        labels = np.ones(dataset.edges.shape[0])
        edges = list(map(tuple, dataset.edges))

        dataset.edges = edges
        dataset.labels = labels
        return dataset

    return formatter
