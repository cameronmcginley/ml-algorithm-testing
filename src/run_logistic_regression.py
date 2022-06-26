# Logistic Regression
# Despite max_iter warnings, 500 iter was found to be best by grid search


# # Prints decision region (dimensionality reduced)
# print_decision_region(log_reg)
from yaml import safe_load, safe_dump
from pathlib import Path
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from cv_score_print import cv_score_print
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from print_decision_region import print_decision_region


def run(dataset, params):
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

    # Use macro for multi class scoring - computes score for each class, then averages with classes unweighted
    scores = cross_validate(
        log_reg,
        dataset["data"],
        dataset["target"],
        cv=5,
        scoring=("accuracy", "precision_macro", "recall_macro", "f1_macro"),
    )

    # Print table of scores
    cv_score_print(scores)

    # Just test against the testing sets
    print(
        "\n\n-----------------------------------------------------\n\n30% Testing Set:"
    )
    # Uses hyperparams found by grid search
    log_reg = LogisticRegression(
        C=1, penalty="l2", solver="saga", max_iter=500
    )
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    # Prints decision region (dimensionality reduced)
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

    # Get dataset
    iris = datasets.load_iris()

    run(iris, model_params)

    # Find hyperparams for every model in model_defns.yml
    # Output results back to same yaml
    # find_hyperparams(model_defns_path, model_params_path, X_train, y_train)


if __name__ == "__main__":
    main()
