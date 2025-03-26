import keras
from tensorflow.keras import layers
from utils import display_hint, display_solution, display_check


def hint2():
    hint = """
Ensure the model follows a VGG-like CNN structure with:
- Multiple Conv2D layers with ReLU activation.
- MaxPooling layers after convolution blocks.
- Dropout layers to prevent overfitting.
- A fully connected (Dense) output layer with 10 units (softmax activation).
"""
    display_hint(hint)


def solution2():
    code = """
def build_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # Output layer
    ])
    return model
"""
    display_solution(code)


def check2(model):
    expected_layers = [
        ('Conv2D', 32), ('Conv2D', 32), ('MaxPooling2D', None), ('Dropout', None),
        ('Conv2D', 64), ('Conv2D', 64), ('MaxPooling2D', None), ('Dropout', None),
        ('Conv2D', 128), ('Conv2D', 128), ('MaxPooling2D', None), ('Dropout', None),
        ('Flatten', None), ('Dense', 512), ('Dropout', None), ('Dense', 10)
    ]

    model_layers = [(layer.__class__.__name__, getattr(layer, 'units', None) or getattr(layer, 'filters', None)) for
                    layer in model.layers]

    if model_layers == expected_layers:
        display_check(True, "Correct! The model architecture is as expected.")
    else:
        display_check(False, "Model architecture does not match the expected VGG-like structure.")

