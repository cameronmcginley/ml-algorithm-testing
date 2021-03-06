"""Output decision region graph for classifier and data."""

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from typing import Any


def print_decision_region(
    clf: Any,
    dataset: Any,
    X_train: list,
    X_test: list,
    y_train: list,
    y_test: list,
) -> None:
    """Output decision region graph for classifier and data.

    Args:
        clf (sklearn model): the fit classifier
        dataset (sklearn dataset): sklearn dataset
        X_train (numpy.ndarray): dataset splits
        X_test (numpy.ndarray):
        y_train (numpy.ndarray):
        y_test (numpy.ndarray):

    """
    # For the sake of visualizing the data, uses PCA for dimenionsality
    # reduction since we use more than 2 features
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.fit_transform(X_test)
    clf.fit(X_train_pca, y_train)

    # Uses mlxtend func to plot
    ax = plot_decision_regions(X_train_pca, y_train, clf=clf, legend=2)

    # PCA makes two compound features to capture as much of the original
    # features as possible
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.title("Decision Region (Dimensionality Reduced w/ PCA)", size=16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, dataset.target_names, framealpha=0.3, scatterpoints=1)
    plt.show()

    # Prints accuracy of this PCA reduced clf, since it's different from the
    # original clf
    y_pred = clf.predict(X_test_pca)
    print(
        "Accuracy of PCA reduced classifer (on test set):",
        accuracy_score(y_test, y_pred),
    )
