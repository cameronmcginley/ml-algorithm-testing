"""Algorithm optimal hyperparameter finding.

This script reads in the data of models/algorithms from
algorithms/model_defns.yml and performs hyperparameter testing. The
best params, along with the accuracy obtained, are printed back
into model_params.yml for each algorithm
"""

from numpy import ndarray
from yaml import safe_load, safe_dump
from pathlib import Path
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Models used
from sklearn.linear_model import LogisticRegression  # noqa, expected unused
from sklearn.svm import SVC  # noqa
from sklearn.ensemble import RandomForestClassifier  # noqa


def find_hyperparams(
    model_defns_path: str,
    model_params_path: str,
    X_train: ndarray,
    y_train: ndarray,
) -> None:
    """Run GridSeachCV on models stored in algorithms/model_defns.yml.

    This func reads in the data of models/algorithms from
    algorithms/model_defns.yml and performs hyperparameter testing. The
    best params, along with the accuracy obtained, are printed back
    into model_params.yml for each algorithm

    Args:
        algorithmsyml_path: File path to algorithms.yml
        X_train: ndarray of dataset's X values
        y_train: ndarray of dataset's y values

    """
    # Read in our models and param_grids defined in the yaml
    with open(model_defns_path, "r") as file:
        data = safe_load(file)

    # For each model in the yaml, grid search against it's param
    # grid and output the results into results yaml under 'params'
    for model_func in data["algorithms"]:
        # Grid search runs through combinations of the given hyperparams and
        # returns the params with the best accuracy found
        clf = GridSearchCV(
            # Key of each model is function ref string, eval it
            estimator=eval(model_func),
            param_grid=data["algorithms"][model_func]["param_grid"],
            scoring="accuracy",
            n_jobs=-1,
        )

        # Run on training set only - GridSearchCV implements its own cross
        # validation but we still want to ensure we have some data that has
        # not been seen before future testing
        results = clf.fit(X_train, y_train)

        print(f"Best Score: {results.best_score_}")
        print(f"Best Hyperparameters: {results.best_params_}")

        # Add results to the dictionary, then output to yaml
        data["algorithms"][model_func]["params"] = {
            "accuracy": float(results.best_score_),
            "params": results.best_params_,
        }
        # Delete the param_grid for the results file
        del data["algorithms"][model_func]["param_grid"]
        with open(model_params_path, "w") as file:
            safe_dump(data, file)


def main() -> None:
    """Handle getting required info for and executing find_hyperparams()."""
    # Get yaml paths for input (defns) and output (params)
    base_path = Path(__file__).parent
    model_defns_path = (
        base_path / "../../algorithms/model_defns.yml"
    ).resolve()
    model_params_path = (
        base_path / "../../algorithms/model_params.yml"
    ).resolve()

    # Get dataset
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris["data"], iris["target"], test_size=0.3, random_state=42
    )

    # Find hyperparams for every model in model_defns.yml
    # Output results back to same yaml
    find_hyperparams(model_defns_path, model_params_path, X_train, y_train)


if __name__ == "__main__":
    main()
