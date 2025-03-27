import numpy as np
from testCNN.utils import display_hint, display_solution, display_check

def hint1():
    hint = """
Ensure that pixel values are scaled to [0,1] by dividing by 255.0.
Also, use `keras.utils.to_categorical` to convert labels into one-hot encoded vectors.
"""
    display_hint(hint)

def solution1():
    code = """
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
"""
    display_solution(code)

def check1(x_train, x_test, y_train, y_test):
    if not (np.max(x_train) <= 1.0 and np.min(x_train) >= 0.0):
        display_check(False, "x_train is not properly normalized.")
    elif not (np.max(x_test) <= 1.0 and np.min(x_test) >= 0.0):
        display_check(False, "x_test is not properly normalized.")
    elif y_train.shape[1] != 10 or y_test.shape[1] != 10:
        display_check(False, "Labels are not correctly one-hot encoded.")
    else:
        display_check(True, "Correct! Here is the next part of the key: BFA67U")
