import tensorflow as tf
from utils import display_hint, display_solution, display_check


def hint5():
    hint = """
Use `tf.math.confusion_matrix` to compute the confusion matrix:
- Pass the true labels as the first argument.
- Pass the predicted labels as the second argument.
- Optionally, specify the number of classes if necessary.
"""
    display_hint(hint)


def solution5():
    code = """
confusion_mtx = tf.math.confusion_matrix(y_test_classes, y_pred_classes)
"""
    display_solution(code)


def check5(y_test_classes, y_pred_classes, confusion_mtx):
    expected_confusion_mtx = tf.math.confusion_matrix(y_test_classes, y_pred_classes)

    if confusion_mtx.shape != expected_confusion_mtx.shape:
        display_check(False, f"Shape mismatch: expected {expected_confusion_mtx.shape} but got {confusion_mtx.shape}")
    elif confusion_mtx.dtype != expected_confusion_mtx.dtype:
        display_check(False,
                      f"Data type mismatch: expected {expected_confusion_mtx.dtype} but got {confusion_mtx.dtype}")
    else:
        display_check(True, "Correct! Here is the next part of the key: Z9X8V7")
