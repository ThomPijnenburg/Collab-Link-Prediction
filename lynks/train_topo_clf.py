import numpy as np
from argparse import ArgumentParser
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from lynks.helpers import load_config

from lynks.logging import get_logger


logger = get_logger(__file__)


def main(config_path: str, data_dir: str, result_dir: str, n_proc: int):
    config_path = Path(config_path)

    configuration = load_config(config_path)

    #
    # Write
    #
    data_dir = Path(data_dir)
    logger.info(f"Loading features from {data_dir}")

    X_train = np.load(data_dir.joinpath("X_train.npy"))
    y_train = np.load(data_dir.joinpath("y_train.npy"))
    X_valid = np.load(data_dir.joinpath("X_valid.npy"))
    y_valid = np.load(data_dir.joinpath("y_valid.npy"))
    X_test = np.load(data_dir.joinpath("X_test.npy"))
    y_test = np.load(data_dir.joinpath("y_test.npy"))

    #
    # Train clf
    #
    logger.info(f"Training classifier...")

    model_configuration = configuration.get("model")

    model = RandomForestClassifier(n_estimators=model_configuration.get("n_estimators"), n_jobs=n_proc)
    model.fit(X_train, y_train)

    #
    # Validate clf
    #
    y_pred_valid = model.predict(X_valid)
    logger.info(f"""
        {classification_report(y_valid, y_pred_valid)}
    """)

    logger.info(f"""
        {confusion_matrix(y_valid, y_pred_valid)}
    """)


if __name__ == "__main__":
    parser = ArgumentParser(description="Full pipeline for LP model using Topological features")
    parser.add_argument("--config_path", type=str,
                        help="Path to experiment config toml file")
    parser.add_argument("--data_dir", type=str,
                        help="path to directory with feature data")
    parser.add_argument("--result_dir", type=str,
                        help="path to directory to write models and results")
    parser.add_argument("--n_proc", type=int,
                        help="number of processes")
    args = parser.parse_args()
    main(args.config_path, args.data_dir, args.result_dir, args.n_proc)
