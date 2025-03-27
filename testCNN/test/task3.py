import numpy as np
from testCNN.utils import display_hint, display_solution, display_check


def hint3():
    hint = """
Ensure you evaluate the model using:
- `model.predict()` to generate predictions.
- `np.argmax()` to convert predictions to class labels.
- `classification_report()` from sklearn for precision, recall, and F1-score.
"""
    display_hint(hint)


def solution3():
    code = """
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

report = classification_report(y_test_classes, y_pred_classes, target_names=class_names)
print(report)
"""
    display_solution(code)

def check3(model, x_test, y_test, y_pred, y_pred_classes, y_test_classes):
    expected_y_pred = model.predict(x_test)
    expected_y_pred_classes = np.argmax(y_pred, axis=1)
    expected_y_test_classes = np.argmax(y_test, axis=1)

    if y_pred.shape != expected_y_pred.shape:
        display_check(False, "Shape mismatch: expected {} but got {}".format(expected_y_pred.shape, y_pred.shape))

    elif y_pred.dtype != expected_y_pred.dtype:
        display_check(False, "Data type mismatch: expected {} but got {}".format(expected_y_pred.dtype, y_pred.dtype))

    elif y_pred_classes.shape != expected_y_pred_classes.shape:
        display_check(False, "Shape mismatch in predicted and class labels: expected {} but got {}".format(expected_y_pred_classes.shape, y_pred_classes.shape))

    elif y_pred_classes.dtype != expected_y_pred_classes.dtype:
        display_check(False, "Data type mismatch in predicted classes: expected {} but got {}".format(expected_y_pred_classes.dtype, y_pred_classes.dtype))

    elif y_test_classes.shape != expected_y_test_classes.shape:
        display_check(False, "Shape mismatch in predicted and test class labels: expected {} but got {}".format(
            expected_y_test_classes.shape, y_pred_classes.shape))

    elif y_test_classes.dtype != expected_y_test_classes.dtype:
        display_check(False, "Data type mismatch in predicted classes: expected {} but got {}".format(expected_y_test_classes.dtype, y_pred_classes.dtype))

    else:
        display_check(True, "Correct! Here is the next part of the key: T543YU")
