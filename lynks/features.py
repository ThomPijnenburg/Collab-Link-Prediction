import joblib
import networkx as nx
import numpy as np

from collections.abc import Callable
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from .pipeline import create_pipeline
from .data import Dataset

def create_negative_sampler() -> Callable:
    def negative_sampler(dataset: Dataset) -> Dataset:
        return dataset

    return negative_sampler


def create_balancer(n_samples: int = -1) -> Callable:
    def balancer(dataset: Dataset) -> Dataset:
        if n_samples > 0:
            edges_ = dataset.edges[:n_samples]
            print(len(edges_))
            dataset.edges = edges_

        # TODO: do balancing

        return dataset

    return balancer


def compute_lp_feature_for_dataset(dataset:dict, lp_callback: Callable, feat_slug:str, verbose: int = 0) -> dict:
    if verbose > 0:
        print("Computing {}...".format(feat_slug))
    G = dataset.graph
    edge_tuples = dataset.edges

    try:
        preds = lp_callback(G, edge_tuples)
        preds_values = [p for _, _, p in preds]
    except:
        preds_values = ["NotImplemented"]

    dataset.features.append(preds_values)
    dataset.feature_names.append(feat_slug)
    return dataset


def create_feature_common_neighbors_count(verbose: int = 0):
    feat_slug = "common_neighbors_count"

    def feat_common_neighbor_count(G, edge_tuples):
        for (x, y) in edge_tuples:
            commons = sorted(nx.common_neighbors(x, y))
            print(commons)
        return [(len(sorted(nx.common_neighbors(x, y))), x, y) for (x, y) in edge_tuples]

    def feat_common_neighbor_count(dataset: Dataset) -> Dataset:
        dataset = compute_lp_feature_for_dataset(dataset, feat_common_neighbor_count, feat_slug=feat_slug, verbose=verbose)
        return dataset

    return feat_common_neighbor_count


def create_feature_common_neighbor_centrality(verbose: int = 0):
    feat_slug = "common_neighbor_centrality"

    def feat_common_neighbor_centrality(dataset: Dataset) -> Dataset:
        dataset = compute_lp_feature_for_dataset(dataset, nx.common_neighbor_centrality, feat_slug=feat_slug, verbose=verbose)
        return dataset

    return feat_common_neighbor_centrality


def create_feature_jaccard_coefficient(verbose: int = 0):
    feat_slug = "jaccard_coefficient"

    def feat_jaccard_coefficient(dataset: Dataset) -> Dataset:
        dataset = compute_lp_feature_for_dataset(dataset, nx.jaccard_coefficient, feat_slug=feat_slug, verbose=verbose)
        return dataset

    return feat_jaccard_coefficient


def create_feature_adamic_adar_index(verbose: int = 0):
    feat_slug = "adamic_adar_index"

    def feat_adamic_adar_index(dataset: Dataset) -> Dataset:

        dataset = compute_lp_feature_for_dataset(dataset, nx.adamic_adar_index, feat_slug=feat_slug, verbose=verbose)
        return dataset

    return feat_adamic_adar_index


def create_feature_preferential_attachment(verbose: int = 0):
    feat_slug = "preferential_attachment"

    def feat_preferential_attachment(dataset: Dataset) -> Dataset:
        dataset = compute_lp_feature_for_dataset(dataset, nx.preferential_attachment, feat_slug=feat_slug, verbose=verbose)
        return dataset

    return feat_preferential_attachment


def create_feature_resource_allocation_index(verbose: int = 0):
    feat_slug = "resource_allocation_index"

    def feat_resource_allocation_index(dataset: Dataset) -> Dataset:
        dataset = compute_lp_feature_for_dataset(dataset, nx.resource_allocation_index, feat_slug=feat_slug, verbose=verbose)
        return dataset

    return feat_resource_allocation_index


def create_feature_formatter(verbose: int = 0):
    def feature_formatter(dataset: Dataset) -> Dataset:
        features = [np.array([feat]).T for feat in dataset.features if "NotImplemented" not in feat]
        feats_np = np.concatenate(features, axis=1)
        dataset.features = feats_np

        if verbose > 0:
            print("Length feat list = {} by {}".format(len(features), len(features[0])))
            print("Shape feat array {}".format(feats_np.shape))

        return dataset

    return feature_formatter


def create_featuriser(common_neighbors_count: bool, common_neighbor_centrality: bool, jaccard_coefficient: bool,
                      adamic_adar_index: bool, preferential_attachment: bool, resource_allocation_index: bool,
                      verbose:int=0):
    feature_steps = []

    if common_neighbors_count:
        feature_steps.append(create_feature_common_neighbors_count(verbose=verbose))

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


    feature_steps.append(create_feature_formatter(verbose=verbose))
    return create_pipeline(feature_steps)


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
