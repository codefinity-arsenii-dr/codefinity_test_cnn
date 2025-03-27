import numpy as np
import tensorflow as tf
from utils import display_hint, display_solution, display_check


def hint4():
    hint = """
Ensure you evaluate the model using:
- `model.evaluate()` to compute accuracy and loss.
- `return_dict=True` to return a dictionary of metrics.
- Ensure that the returned dictionary contains valid float values.
"""
    display_hint(hint)


def solution4():
    code = """
metrics = model.evaluate(x_test, y_test, verbose=2, return_dict=True)
"""
    display_solution(code)


def check4(model, x_test, y_test, metrics):
    expected_metrics = model.evaluate(x_test, y_test, verbose=0, return_dict=True)

    if not isinstance(metrics, dict):
        display_check(False, "Evaluation results should be a dictionary.")

    elif set(metrics.keys()) != set(expected_metrics.keys()):
        display_check(False,
                      f"Metric keys mismatch: expected {set(expected_metrics.keys())} but got {set(metrics.keys())}")

    else:
        all_correct = True
        for metric, value in metrics.items():
            if not isinstance(value, float):
                display_check(False, f"Metric {metric} should be a float but got {type(value)}")
                all_correct = False
            elif np.isnan(value) or np.isinf(value):
                display_check(False, f"Invalid value for {metric}: {value}")
                all_correct = False

        if all_correct:
            display_check(True, "Correct! Here is the next part of the key: 4L99TR")
