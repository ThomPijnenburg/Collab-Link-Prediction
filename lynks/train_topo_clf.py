import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from ogb.linkproppred import LinkPropPredDataset

from lynks.helpers import load_config

from lynks.data import build_graph
from lynks.features import create_graph_topology_featuriser
from lynks.features import create_feature_formatter
from lynks.helpers import create_pipeline
from lynks.logging import get_logger
from lynks.model import create_train_RandomForestClassifier

from pykeen.sampling import BasicNegativeSampler
from pykeen.triples import TriplesFactory
from pykeen.triples.triples_factory import create_entity_mapping, create_relation_mapping

logger = get_logger(__file__)

EDG = "edg"
SRC = "src"
TGT = "tgt"
EDGE_LABEL = "collaborated"
WEIGHT = "weight"
YEAR = "year"


def triples_factory_from_edge_data(triples_df: pd.DataFrame, reuse_mappings: TriplesFactory or None = None) -> TriplesFactory:

    triples_np = triples_df[[SRC, EDG, TGT]].to_numpy()

    if reuse_mappings is not None:
        triples_factory = TriplesFactory.from_labeled_triples(
            triples_np,
            entity_to_id=reuse_mappings.entity_to_id,
            relation_to_id=reuse_mappings.relation_to_id)
    else:
        triples_factory = TriplesFactory.from_labeled_triples(triples_np)

    return triples_factory


def edge_data_to_df(edge_data) -> pd.DataFrame:
    triples_df = pd.DataFrame(edge_data["edge"], columns=[SRC, TGT], dtype=str)
    triples_df[EDG] = EDGE_LABEL
    triples_df[WEIGHT] = edge_data["weight"]
    triples_df[YEAR] = edge_data["year"]

    return triples_df


def featurise_triples(positive_triples, negative_triples, feat_pipe):
    feat_formatter = create_feature_formatter()

    # train features
    edges_pos, pos_feats, feat_labels_pos = feat_pipe(positive_triples)
    edges_neg, neg_feats, feat_labels_neg = feat_pipe(negative_triples)

    pos_feats_np, feat_labels_np = feat_formatter(pos_feats, feat_labels_pos)
    neg_feats_np, feat_labels_np = feat_formatter(neg_feats, feat_labels_neg)

    X_train = np.concatenate([pos_feats_np, neg_feats_np], axis=0)
    y_train = np.concatenate([
        np.ones(pos_feats_np.shape[0]), np.zeros(neg_feats_np.shape[0])], axis=0)

    return X_train, y_train


def main(config_path: str, out_dir: str):
    config_path = Path(config_path)

    configuration = load_config(config_path)

    logger.info(f"Loading dataset {configuration.get('dataset_name')}...")
    lp_dataset = LinkPropPredDataset(configuration['dataset_name'])

    split_edge = lp_dataset.get_edge_split()

    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]

    train_df = edge_data_to_df(train_edge).drop_duplicates(subset=[SRC, EDG, TGT])
    valid_df = edge_data_to_df(valid_edge).drop_duplicates(subset=[SRC, EDG, TGT])
    test_df = edge_data_to_df(test_edge).drop_duplicates(subset=[SRC, EDG, TGT])

    train_triples_factory = triples_factory_from_edge_data(train_df)
    valid_triples_factory = triples_factory_from_edge_data(valid_df)
    test_triples_factory = triples_factory_from_edge_data(test_df)

    # TODO: We are losing a set of triples because of duplicates (multiple collabs?)
    logger.info(f"Train triples {len(train_triples_factory.mapped_triples)}, \
        Valid triples {len(train_triples_factory.mapped_triples)}, \
        Test triples {len(train_triples_factory.mapped_triples)}")

    #
    # generate negatives
    #
    num_negs_per_pos = configuration.get("sampling").get("n_samples")
    logger.info(f"Sampling {num_negs_per_pos} negatives per positive.")

    sampler = BasicNegativeSampler(
        train_triples_factory, num_negs_per_pos=num_negs_per_pos)

    train_negatives, _ = sampler.sample(train_triples_factory.mapped_triples)
    valid_negatives, _ = sampler.sample(valid_triples_factory.mapped_triples)
    test_negatives, _ = sampler.sample(test_triples_factory.mapped_triples)

    # build graph backbone
    graph_backbone = build_graph(
        train_triples_factory.mapped_triples,
        train_df[WEIGHT].values,
        train_df[YEAR].values)

    logger.info(f"Graph with {graph_backbone.number_of_edges()} edges")

    #
    # Generate features
    #
    feature_config = configuration["features"]
    logger.info(f"Generating features: {feature_config}")

    topo_feature_pipeline = create_graph_topology_featuriser(
        graph_backbone=graph_backbone,
        common_neighbors_count=feature_config["common_neighbors_count"],
        common_neighbor_centrality=feature_config["common_neighbor_centrality"],
        jaccard_coefficient=feature_config["jaccard_coefficient"],
        adamic_adar_index=feature_config["adamic_adar_index"],
        preferential_attachment=feature_config["preferential_attachment"],
        resource_allocation_index=feature_config["resource_allocation_index"],
        verbose=1
    )

    # train features
    logger.info("Generating features for train...")
    X_train, y_train = featurise_triples(train_triples_factory.mapped_triples, train_negatives, topo_feature_pipeline)
    logger.info(f"Shape of X: {X_train.shape}, y: {y_train.shape}")

    # valid features
    logger.info("Generating features for valid...")
    X_valid, y_valid = featurise_triples(valid_triples_factory.mapped_triples, valid_negatives, topo_feature_pipeline)

    # test features
    logger.info("Generating features for test...")
    X_test, y_test = featurise_triples(test_triples_factory.mapped_triples, test_negatives, topo_feature_pipeline)

    #
    # Train clf
    #
    logger.info(f"Training classifier...")

    model_configuration = configuration.get("model")

    model = RandomForestClassifier(n_estimators=model_configuration.get("n_estimators"))
    model.fit(X_train, y_train)

    y_pred_valid = model.predict(X_valid)
    logger.info(classification_report(y_valid, y_pred_valid))


if __name__ == "__main__":
    parser = ArgumentParser(description="Full pipeline for LP model using Topological features")
    parser.add_argument("--config_path", type=str,
                        help="Path to experiment config toml file")
    parser.add_argument("--out_dir", type=str,
                        help="path to directory to write trained models")

    args = parser.parse_args()
    main(args.config_path, args.out_dir)
