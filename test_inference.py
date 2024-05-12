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

import numpy as np
import onnxruntime as ort
from PIL import Image


def prepare_image(image_path):
    with Image.open(image_path) as image:
        image_np = np.array(image.resize((128, 128))).astype("float32")
    image_np /= 255.0  # Normalize
    image_np = np.expand_dims(image_np, axis=0)  # Add a batch dimension
    return image_np


if __name__ == "__main__":

    cat_image_path = "images/cat.jpg"
    dog_image_path = "images/dog.jpg"

    paths = [cat_image_path, dog_image_path]

    images = [prepare_image(path) for path in paths]

    session = ort.InferenceSession("models/model.onnx")

    # Name of the input node
    input_name = session.get_inputs()[0].name

    classes = ["cat", "dog"]

    for i, image_np in enumerate(images, 1):
        # Perform inference
        result = session.run(None, {input_name: image_np})

        predicted_class_index = np.argmax(result[0])
        print(
            f"Image {i} {paths[i-1]} predicted class: "
            + f"{classes[predicted_class_index]}"
        )
