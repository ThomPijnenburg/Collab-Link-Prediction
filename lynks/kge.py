import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pathlib import Path
from time import time

from ogb.linkproppred import LinkPropPredDataset

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
from pykeen.regularizers import LpRegularizer

from lynks.helpers import load_config
from lynks.logging import get_logger
from lynks.data import edge_data_to_df
from lynks.data import EDG, SRC, TGT


logger = get_logger(__file__)

ALLOWED_MODELS = {"TransE"}
TRIPLE_LABELS = [SRC, EDG, TGT]


def main(args):
    """"""
    config_path = Path(args.config_path)
    configuration = load_config(config_path)

    model = configuration.get("model")

    assert model in ALLOWED_MODELS, f"Model {model} not in allowed models: {ALLOWED_MODELS}"

    model_params = configuration.get(model)
    optimizer_params = configuration.get("optimizer")
    training_params = configuration.get("training")
    regularizer = configuration.get("regularizer")
    regularizer_params = configuration.get(regularizer)
    evaluator = configuration.get("evaluator")
    evaluator_params = configuration.get(evaluator)
    #
    # Load data
    #
    dataset = configuration.get('dataset')
    logger.info(f"Loading dataset {dataset}...")

    lp_dataset = LinkPropPredDataset(dataset)
    split_edge = lp_dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]

    train_df = edge_data_to_df(train_edge)
    train_triples = train_df[TRIPLE_LABELS].values

    valid_df = edge_data_to_df(valid_edge)
    valid_triples = valid_df[TRIPLE_LABELS].values

    test_df = edge_data_to_df(test_edge)
    test_triples = test_df[TRIPLE_LABELS].values

    training = TriplesFactory.from_labeled_triples(train_triples)
    valid = TriplesFactory.from_labeled_triples(
        valid_triples,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id)
    testing = TriplesFactory.from_labeled_triples(
        test_triples,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id)

    pipeline_result = pipeline(
        training=training,
        validation=valid,
        testing=testing,
        model=model,
        model_kwargs=model_params,
        training_kwargs=training_params,
        optimizer_kwargs=optimizer_params,
        regularizer=regularizer,
        regularizer_kwargs=regularizer_params,
        evaluator=evaluator,
        evaluator_kwargs=evaluator_params,
        device='cpu')

    timestamp = int(time())
    out_dir = Path(args.result_dir).joinpath(f'collab-transe/{timestamp}/')
    pipeline_result.save_to_directory(out_dir)
    # pipeline_result.plot()
    # plt.savefig(out_dir.joinpath('results.pdf'))


if __name__ == "__main__":
    parser = ArgumentParser(description="Full pipeline for LP model using KGE models")
    parser.add_argument("--config_path", type=str,
                        help="Path to experiment config toml file")
    parser.add_argument("--result_dir", type=str,
                        help="path to directory to write models and results")
    args = parser.parse_args()
    main(args)
