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
import streamlit as st
from PIL import Image, UnidentifiedImageError

classes = ["cat", "dog"]

uploaded_file = st.file_uploader("Choose a file")


def prepare_pil_image(image):
    image_np = np.array(image.resize((128, 128))).astype("float32")
    image_np /= 255.0  # Normalize
    image_np = np.expand_dims(image_np, axis=0)  # Add a batch dimension
    return image_np


@st.cache_resource
def get_inference_session():
    session = ort.InferenceSession("models/model.onnx")
    return session


if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as im:
            st.image(im, width=256)
        image_np = prepare_pil_image(im)
        session = get_inference_session()
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: image_np})
        predicted_class_index = np.argmax(result[0])
        st.write(
            f"This cute animal is a **{classes[predicted_class_index]}**."
        )
    except UnidentifiedImageError:
        file_extension = "." + uploaded_file.name.split(".")[-1]
        st.write(
            "Got an UnidentifiedImageError from what you uploaded."
            + f"Can't handle those pesky {file_extension} files."
        )
else:
    st.write("Nothing to show.")
