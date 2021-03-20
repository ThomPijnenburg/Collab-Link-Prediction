import joblib
import networkx as nx
import numpy as np

from collections.abc import Callable
from collections.abc import Iterable
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from typing import Union

from .pipeline import create_pipeline
from .data import Dataset
from .logging import get_logger


logger = get_logger(__name__)


def pipeline_cartridge_for_feature(feature_function: Callable) -> Callable:
    def mutating_feature_function(graph: nx.Graph, edge_tuples: Iterable, features: Iterable, feature_labels: Iterable) -> tuple:
        feat_values, feat_slug = feature_function(graph, edge_tuples)
        features.append(feat_values)
        feature_labels.append(feat_slug)
        return graph, edge_tuples, features, feature_labels

    return mutating_feature_function


def compute_lp_feature(graph: nx.Graph, edge_tuples: Iterable, lp_callback: Callable, verbose: int = 0) -> np.array:

    try:
        feats = lp_callback(graph, edge_tuples)
        feat_values = np.array([p for _, _, p in feats]) # todo maybe return the nodes as well
    except:
        logger.warn("Could not compute {}".format(str(lp_callback)))
        feat_values = np.full(len(edge_tuples), np.nan)

    return feat_values


# def create_feature_common_neighbors_count(verbose: int = 0):
#     label = "common_neighbors_count"

#     # def feat_common_neighbor_count(G, edge_tuples):
#     #     for (x, y) in edge_tuples:
#     #         commons = sorted(nx.common_neighbors(x, y))
#     #         print(commons)
#     #     return [(len(sorted(nx.common_neighbors(x, y))), x, y) for (x, y) in edge_tuples]

#     def feat_common_neighbor_count(graph: nx.Graph, edge_tuples: Iterable) -> tuple:
#         if verbose > 0:
#             logger.info("Computing {}...".format(label))
#         feat_values = compute_lp_feature(graph, edge_tuples, feat_common_neighbor_count, verbose=verbose)
#         return feat_values, label

#     return feat_common_neighbor_count


def create_feature_common_neighbor_centrality(verbose: int = 0):
    label = "common_neighbor_centrality"

    def feat_common_neighbor_centrality(graph:nx.Graph, edge_tuples: Iterable) -> tuple:
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edge_tuples, nx.common_neighbor_centrality, verbose=verbose)
        return feat_values, label

    return feat_common_neighbor_centrality


def create_feature_jaccard_coefficient(verbose: int = 0):
    label = "jaccard_coefficient"

    def feat_jaccard_coefficient(graph:nx.Graph, edge_tuples: Iterable) -> tuple:
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edge_tuples, nx.jaccard_coefficient, verbose=verbose)
        return feat_values, label

    return feat_jaccard_coefficient


def create_feature_adamic_adar_index(verbose: int = 0):
    label = "adamic_adar_index"

    def feat_adamic_adar_index(graph:nx.Graph, edge_tuples: Iterable) -> tuple:
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edge_tuples, nx.adamic_adar_index, verbose=verbose)
        return feat_values, label

    return feat_adamic_adar_index


def create_feature_preferential_attachment(verbose: int = 0):
    label = "preferential_attachment"

    def feat_preferential_attachment(graph:nx.Graph, edge_tuples: Iterable) -> tuple:
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edge_tuples, nx.preferential_attachment, verbose=verbose)
        return feat_values, label

    return feat_preferential_attachment


def create_feature_resource_allocation_index(verbose: int = 0):
    label = "resource_allocation_index"

    def feat_resource_allocation_index(graph:nx.Graph, edge_tuples: Iterable) -> tuple:
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edge_tuples, nx.resource_allocation_index, verbose=verbose)
        return feat_values, label

    return feat_resource_allocation_index


def create_feature_formatter(verbose: int = 0):
    def feature_formatter(dataset: Dataset) -> Dataset:
        features = []
        for index, feat in enumerate(dataset.features):
            if not np.isnan(feat).any():
                features.append(np.array([feat]).T)
            else:
                dataset.feature_names.pop(index)
        feats_np = np.concatenate(features, axis=1)
        dataset.features = feats_np

        if verbose > 0:
            logger.info("Length feat list = {} by {}".format(len(features), len(features[0])))
            logger.info("Shape feat array {}".format(feats_np.shape))

        return dataset

    return feature_formatter



def create_graph_topology_featuriser(common_neighbors_count: bool, common_neighbor_centrality: bool, jaccard_coefficient: bool,
                      adamic_adar_index: bool, preferential_attachment: bool, resource_allocation_index: bool,
                      verbose:int=0) -> Callable:
    def graph_topology_featuriser(graph_edges: list, edges_to_feat: Union[None, Iterable] = None) -> tuple:
        features = []
        feature_labels = []
        
        feature_steps = []

        # if common_neighbors_count:
        #     feature_steps.append(create_feature_common_neighbors_count(verbose=verbose))

        if common_neighbor_centrality:
            feature_steps.append(create_feature_common_neighbor_centrality(verbose=verbose))

        if jaccard_coefficient:
            feature_steps.append(create_feature_jaccard_coefficient(verbose=verbose))

        if adamic_adar_index:
            feature_steps.append(create_feature_adamic_adar_index(verbose=verbose))

        if preferential_attachment:
            feature_steps.append(create_feature_preferential_attachment(verbose=verbose))

        if resource_allocation_index:
            feature_steps.append(create_feature_resource_allocation_index(verbose=verbose))

        mutating_feature_steps = map(pipeline_cartridge_for_feature, feature_steps)

        feat_pipe = create_pipeline(mutating_feature_steps)

        return feat_pipe(graph_edges, edges_to_feat, features, feature_labels)

    return graph_topology_featuriser


def create_feature_transform(scaling:str):
    def transform(dataset: Dataset, pretrained_scaler_path: Path = None) -> Dataset:
        feats_np = dataset.features

        if scaling == "normalise":
            scaler = StandardScaler()
            feats_np = scaler.fit_transform(feats_np)

        if scaling == "minmax":
            scaler = MinMaxScaler()
            feats_np = scaler.fit_transform(feats_np)
        dataset.features = feats_np

        return dataset

    return transform
