"""Prints performance statistics and graph for sklearn model.

This script takes model paramaters found in model_params.yml, and tests
against them for accuracy, precision, recall, f1. The decision region
is graphed with PCA dimensionality reduction
"""

from yaml import safe_load
from pathlib import Path
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_validate
from Helpers.cv_score_print import cv_score_print
from Helpers.print_decision_region import print_decision_region
from typing import Any

# Models used
from sklearn.linear_model import LogisticRegression  # noqa, expected unused
from sklearn.svm import SVC  # noqa
from sklearn.ensemble import RandomForestClassifier  # noqa


def test_logistic_regression(dataset: Any, model_data: dict) -> None:
    """Prints performance statistics and graph for sklearn model.

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
    model_funcs = model_data["algorithms"].keys()

    for func_name in model_funcs:
        params = model_data["algorithms"][func_name]["params"]["params"]

        # Cut the () from func name string from yaml
        func_to_eval = func_name[:-2]

        print("5-Fold Cross Validitation on Full Dataset:")

        # Pass param dict as individual args
        clf = eval(func_to_eval)(**params)

        # Use macro for multi class scoring - computes score for each class,
        #  then averages with classes unweighted
        scores = cross_validate(
            clf,
            dataset["data"],
            dataset["target"],
            cv=5,
            scoring=(
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ),
        )

        cv_score_print(scores)
        print_decision_region(clf, dataset, X_train, X_test, y_train, y_test)


def main() -> None:
    """Handle getting required info for and executing the script."""
    # Get the model data to run with
    base_path = Path(__file__).parent
    model_params_path = (
        base_path / "../algorithms/model_params.yml"
    ).resolve()
    with open(model_params_path, "r") as file:
        data = safe_load(file)

    # Get dataset
    iris = datasets.load_iris()

    test_logistic_regression(iris, data)


if __name__ == "__main__":
    main()
