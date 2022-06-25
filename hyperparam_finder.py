from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Models used
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Ignore the sklearn warnings when using gridsearch
from sklearn.utils._testing import ignore_warnings


def main():
    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris["data"], iris["target"], test_size=0.3, random_state=42
    )

    # Use GridSearchCV to find optimal hyperparameters for the models

    # For the three classifiers we use LogisticRegression, SVC, and RandomForestClassifier

    # Define the models along with paramaters to search
    models = [
        {
            "type": LogisticRegression(),
            "param_grid": {
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                "penalty": ["none", "l1", "l2", "elasticnet"],
                "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                "max_iter": [50, 100, 500, 1500, 5000],
            },
        },
        {
            "type": SVC(),
            "param_grid": {
                "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
                "gamma": ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
            },
        },
        {
            "type": RandomForestClassifier(),
            "param_grid": {
                "n_estimators": [10, 100, 500, 1000],
                "max_features": ["auto", "sqrt", "log2"],
                "max_depth": [10, 20, 60, 100, None],
                "n_jobs": [-1],
            },
        },
    ]

    for model in models:
        # Grid search runs through combinations of the given hyperparams and
        # returns the params with the best accuracy found
        clf = GridSearchCV(
            model["type"],
            model["param_grid"],
            scoring="accuracy",
            n_jobs=-1,
        )

        # Run on training set only - GridSearchCV implements its own cross
        # validation but we still want to ensure we have some data that has
        # not been seen before future testing
        results = clf.fit(X_train, y_train)

        print("Best Score: %s" % results.best_score_)
        print("Best Hyperparameters: %s" % results.best_params_)

        # Add results to the dictionary
        model["param_results"] = {
            "accuracy": results.best_score_,
            "params": results.best_params_,
        }


if __name__ == "__main__":
    main()
