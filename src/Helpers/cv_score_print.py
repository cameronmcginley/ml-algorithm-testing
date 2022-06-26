"""Print classifier scores as a table."""

from tabulate import tabulate


def cv_score_print(scores: dict[str, float]) -> None:
    """Print classifier scores as a table.

    Args:
        scores: contains test_accuracy, test_precision_macro,
            test_recall_macro, test_f1_macro, fit_time

    """
    # Convert fit times to ms
    for i in range(len(scores["fit_time"])):
        scores["fit_time"][i] *= 1000

    print(
        tabulate(
            [
                [
                    "Accuracy",
                    scores["test_accuracy"].mean(),
                    scores["test_accuracy"].std(),
                    scores["test_accuracy"],
                ],
                [
                    "Precision",
                    scores["test_precision_macro"].mean(),
                    scores["test_precision_macro"].std(),
                    scores["test_precision_macro"],
                ],
                [
                    "Recall",
                    scores["test_recall_macro"].mean(),
                    scores["test_recall_macro"].std(),
                    scores["test_recall_macro"],
                ],
                [
                    "F1",
                    scores["test_f1_macro"].mean(),
                    scores["test_f1_macro"].std(),
                    scores["test_f1_macro"],
                ],
                [
                    "Fit time (ms)",
                    scores["fit_time"].mean(),
                    scores["fit_time"].std(),
                    scores["fit_time"],
                ],
            ],
            headers=["", "Average", "Std", "List"],
        )
    )
