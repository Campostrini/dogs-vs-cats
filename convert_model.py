#    Copyright 2024 Stefano Campostrini
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse

import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

IMAGE_WIDTH = IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3

model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(
    Dense(2, activation="softmax")
)  # 2 because we have cat and dog classes

model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

model.summary()

print(type(model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model")
    parser.add_argument("output_model")
    args = parser.parse_args()

    model.load_weights(args.input_model)
    model.output_names = [output.name for output in model.outputs]

    input_signature = [
        tf.TensorSpec(
            [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
            tf.float32,
            name="input",
        )
    ]

    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_signature, output_path=args.output_model
    )
