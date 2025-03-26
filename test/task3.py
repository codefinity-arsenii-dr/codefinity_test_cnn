import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from utils import display_hint, display_solution, display_check


def hint3():
    hint = """
Ensure you evaluate the model using:
- `model.evaluate()` to get accuracy.
- `tf.math.confusion_matrix()` for confusion matrix.
- `classification_report()` from sklearn for precision, recall, and F1-score.
"""
    display_hint(hint)


def solution3():
    code = """
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
report = classification_report(y_true, y_pred_classes)
"""
    display_solution(code)


def check3(model, x_test, y_test):
    test_results = model.evaluate(x_test, y_test, verbose=0)
    if test_results is None or not isinstance(test_results, (list, tuple)):
        display_check(False, "Model evaluation did not return valid results.")
        return

    y_pred = model.predict(x_test)
    if y_pred is None or not isinstance(y_pred, np.ndarray):
        display_check(False, "Model prediction did not return a valid numpy array.")
        return

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)
    report = classification_report(y_true, y_pred_classes, output_dict=True)

    if confusion_mtx is None or not isinstance(confusion_mtx, tf.Tensor):
        display_check(False, "Confusion matrix is not valid.")
    elif report is None or not isinstance(report, dict):
        display_check(False, "Classification report is not valid.")
    else:
        display_check(True, "Correct! Model evaluation has been properly conducted.")
