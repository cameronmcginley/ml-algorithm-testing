"""Algorithm optimal hyperparameter finding.

This script reads in the data of models/algorithms from
algorithms/algorithms.yml and performs hyperparameter testing. The
best params, along with the accuracy obtained, are printed back
into algorithms.yml under 'param_results' for each algorithm
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Models used
from sklearn.linear_model import LogisticRegression  # noqa, expected unused
from sklearn.svm import SVC  # noqa
from sklearn.ensemble import RandomForestClassifier  # noqa
import yaml

from pathlib import Path


def find_hyperparams(algorithmsyml_path: str) -> None:
    """Run GridSeachCV on models stored in algorithms/algorithms.yml.

    This func reads in the data of models/algorithms from
    algorithms/algorithms.yml and performs hyperparameter testing. The
    best params, along with the accuracy obtained, are printed back
    into algorithms.yml under 'param_results' for each algorithm

    Args:
        algorithmsyml_path: File path to algorithms.yml

    """
    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris["data"], iris["target"], test_size=0.3, random_state=42
    )

    # Use GridSearchCV to find optimal hyperparameters for the models

    # For the three classifiers we use LogisticRegression, SVC, and
    # RandomForestClassifier

    # Read in our models and param_grids defined in the yaml
    with open(algorithmsyml_path, "r") as file:
        data = yaml.safe_load(file)

    # For each model in the yaml, grid search against it's param
    # grid and output the results back into the yaml under 'param_results'
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

        print("Best Score: %s" % results.best_score_)
        print("Best Hyperparameters: %s" % results.best_params_)

        # Add results to the dictionary, then output to yaml
        data["algorithms"][model_func]["param_results"] = {
            "accuracy": float(results.best_score_),
            "params": results.best_params_,
        }
        with open(algorithmsyml_path, "w") as file:
            yaml.dump(data, file)


if __name__ == "__main__":
    # Get algorithms.yml
    base_path = Path(__file__).parent
    algorithmsyml_path = (base_path / "../algorithms/algorithms.yml").resolve()

    # Find hyperparams for every model in algorithms.yml
    # Output results back to same yaml
    find_hyperparams(algorithmsyml_path)
