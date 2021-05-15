import networkx as nx
import numpy as np
import torch

from collections.abc import Iterable
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from typing import Union, Callable, Tuple

from .helpers import create_pipeline
from .logging import get_logger


logger = get_logger(__name__)


def pipeline_cartridge_for_feature(feature_function: Callable) -> Callable:
    """Construct standardised function from Networkx callback"""
    def mutating_feature_function(edges: Iterable, features: Iterable, feature_labels: Iterable) -> tuple:
        feat_values, feat_slug = feature_function(edges)
        features.append(feat_values)
        feature_labels.append(feat_slug)
        return edges, features, feature_labels

    return mutating_feature_function


def compute_lp_feature(graph: nx.Graph, edges: Iterable, lp_callback: Callable, verbose: int = 0) -> np.array:
    """Perform callback"""
    try:
        feats = lp_callback(graph, edges)
        feat_values = np.array([[p for _, _, p in feats]]).T  # todo maybe return the nodes as well
    except Exception as e:
        logger.warn(f"Could not compute {str(lp_callback)}")
        logger.warn(f"Error: {e}")

        feat_values = np.full((len(edges), 1), np.nan)

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


def create_feature_common_neighbor_centrality(graph: nx.Graph, verbose: int = 0) -> Callable:
    label = "common_neighbor_centrality"

    def feat_common_neighbor_centrality(edges: Iterable) -> tuple:
        """Compute common neighbor centrality. This algorithm is based on two properties of nodes:
        i) common neighbors, ii) their centrality. This measure is a weighted sum of the two parameters.

        ref: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx\
            .algorithms.link_prediction.common_neighbor_centrality.html#networkx.algorithms\
            .link_prediction.common_neighbor_centrality
        """
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edges, nx.common_neighbor_centrality, verbose=verbose)
        return feat_values, label

    return feat_common_neighbor_centrality


def create_feature_jaccard_coefficient(graph: nx.Graph, verbose: int = 0) -> Callable:
    label = "jaccard_coefficient"

    def feat_jaccard_coefficient(edges: Iterable) -> tuple:
        """Compute Jaccard Coefficient. This measure defines the overlap of the neighbors of u and v.

        ref: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx\
            .algorithms.link_prediction.jaccard_coefficient.html#networkx.algorithms\
            .link_prediction.jaccard_coefficient
        """
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edges, nx.jaccard_coefficient, verbose=verbose)
        return feat_values, label

    return feat_jaccard_coefficient


def create_feature_adamic_adar_index(graph: nx.Graph, verbose: int = 0):
    label = "adamic_adar_index"

    def feat_adamic_adar_index(edges: Iterable) -> tuple:
        """Compute Adamic Adar index. This measure reflects the connectivity of the
        common neighbors of u and v

        ref: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx\
            .algorithms.link_prediction.adamic_adar_index.html#networkx.algorithms.\
            link_prediction.adamic_adar_index
        """
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edges, nx.adamic_adar_index, verbose=verbose)
        return feat_values, label

    return feat_adamic_adar_index


def create_feature_preferential_attachment(graph: nx.Graph, verbose: int = 0):
    label = "preferential_attachment"

    def feat_preferential_attachment(edges: Iterable) -> tuple:
        """Compute Preferential Attachment. PA is the multiplication of the number of neighbors of
        u and v.

        ref: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.\
            algorithms.link_prediction.preferential_attachment.html#networkx.algorithms\
            .link_prediction.preferential_attachment
        """
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edges, nx.preferential_attachment, verbose=verbose)
        return feat_values, label

    return feat_preferential_attachment


def create_feature_resource_allocation_index(graph: nx.Graph, verbose: int = 0):
    label = "resource_allocation_index"

    def feat_resource_allocation_index(edges: Iterable) -> tuple:
        """Compute Resource Allocation index. A different measure of the sum of inverse connectivity
        of the common neighbors of u and v.

        ref: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx\
            .algorithms.link_prediction.resource_allocation_index.html#networkx.algorithms\
            .link_prediction.resource_allocation_index
        """
        if verbose > 0:
            logger.info("Computing {}...".format(label))
        feat_values = compute_lp_feature(graph, edges, nx.resource_allocation_index, verbose=verbose)
        return feat_values, label

    return feat_resource_allocation_index


def create_feature_formatter(verbose: int = 0):
    def feature_formatter(feature_list: list, feature_labels: list) -> np.array:
        features = []
        labels = []
        for index, feat in enumerate(feature_list):
            if not np.isnan(feat).all():
                features.append(feat)
                labels.append(feature_labels[index])

        feats_np = np.concatenate(features, axis=1)

        if verbose > 0:
            logger.info("Length feat list = {} by {}".format(len(features), len(features[0])))
            logger.info("Shape feat array {}".format(feats_np.shape))

        return feats_np, labels

    return feature_formatter


def triples_to_nx_tuples(mapped_triples):
    return [(x[0], x[2]) for x in mapped_triples.numpy()]


def create_graph_topology_featuriser(
        graph_backbone: nx.Graph, common_neighbors_count: bool, common_neighbor_centrality: bool,
        jaccard_coefficient: bool, adamic_adar_index: bool, preferential_attachment: bool,
        resource_allocation_index: bool, verbose: int = 0) -> Callable:
    """Create feature function given backbone graph and configuration. The backbone graph
    is used to compute all network features for mapped triples.
    """
    def graph_topology_featuriser(mapped_triples) -> tuple:
        features = []
        feature_labels = []

        feature_steps = []

        # if common_neighbors_count:
        #     feature_steps.append(create_feature_common_neighbors_count(verbose=verbose))

        if common_neighbor_centrality:
            feature_steps.append(create_feature_common_neighbor_centrality(graph_backbone, verbose=verbose))

        if jaccard_coefficient:
            feature_steps.append(create_feature_jaccard_coefficient(graph_backbone, verbose=verbose))

        if adamic_adar_index:
            feature_steps.append(create_feature_adamic_adar_index(graph_backbone, verbose=verbose))

        if preferential_attachment:
            feature_steps.append(create_feature_preferential_attachment(graph_backbone, verbose=verbose))

        if resource_allocation_index:
            feature_steps.append(create_feature_resource_allocation_index(graph_backbone, verbose=verbose))

        mutating_feature_steps = map(pipeline_cartridge_for_feature, feature_steps)

        feat_pipe = create_pipeline(mutating_feature_steps)

        return feat_pipe(triples_to_nx_tuples(mapped_triples), features, feature_labels)

    return graph_topology_featuriser


def create_feature_transform(scaling: str, load_scaler: Path = None):
    def transform(feats: np.array) -> np.array:
        # TODO: implement load_scaler
        feats_transformed = None

        if scaling == "normalise":
            scaler = StandardScaler()
            feats_transformed = scaler.fit_transform(feats)

        if scaling == "minmax":
            scaler = MinMaxScaler()
            feats_transformed = scaler.fit_transform(feats)

        return feats_transformed

    return transform
