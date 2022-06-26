"""Prints performance statistics and graph for LogisticRegression().

This script takes model paramaters found in model_params.yml, and tests
against them for accuracy, precision, recall, f1. The decision region
is graphed with PCA dimensionality reduction
"""

from yaml import safe_load
from pathlib import Path
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from Helpers.cv_score_print import cv_score_print
from Helpers.print_decision_region import print_decision_region
from typing import Any


def test_logistic_regression(dataset: Any, params: dict[str, Any]) -> None:
    """Prints performance statistics and graph for LogisticRegression().

    This function takes model paramaters found in model_params.yml, and tests
    against them for accuracy, precision, recall, f1. The decision region
    is graphed with PCA dimensionality reduction

    Args:
        dataset (sklearn dataset): dataset, with "data" and "target"
        params: model parameters dict defined in model_params.yml

    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["data"], dataset["target"], test_size=0.3, random_state=42
    )

    # With cross validation on full data
    print("5-Fold Cross Validitation on Full Dataset:")
    log_reg = LogisticRegression(
        C=params["C"],
        penalty=params["penalty"],
        solver=params["solver"],
        max_iter=params["max_iter"],
    )

    # Use macro for multi class scoring - computes score for each class,
    #  then averages with classes unweighted
    scores = cross_validate(
        log_reg,
        dataset["data"],
        dataset["target"],
        cv=5,
        scoring=("accuracy", "precision_macro", "recall_macro", "f1_macro"),
    )

    cv_score_print(scores)
    print_decision_region(log_reg, dataset, X_train, X_test, y_train, y_test)


def main() -> None:
    """Handle getting required info for and executing the script."""
    # Get the model params
    base_path = Path(__file__).parent
    model_params_path = (
        base_path / "../algorithms/model_params.yml"
    ).resolve()
    with open(model_params_path, "r") as file:
        data = safe_load(file)
    model_params = data["algorithms"]["LogisticRegression()"]["params"][
        "params"
    ]

    iris = datasets.load_iris()

    test_logistic_regression(iris, model_params)


if __name__ == "__main__":
    main()
