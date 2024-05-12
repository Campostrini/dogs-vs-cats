# Dogs vs. Cats

This repo contains the code for a web app that allows users to upload an image and get a prediction of whether the image is a dog or a cat.
The web app is built using streamlit and the model is built using Keras. The original notebook can be found [here](https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification/notebook),
whereas the h5 file containing the weights is taken from [here](https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification/output).

The model is also converted to ONNX format. See the models folder.

## How to run the webapp

After cloning the repo, navigate to the root directory and create a conda environment using the following command:
```bash
conda env create -f inference-environment.yml 
conda activate dogs-vs-cats 
```

Then, run the following command to start the web app:
```bash
streamlit run app.py
```

Using the web app is straightforward and left as an exercise to the user.

## Other code in this repo
### `convert_model.py`
This script converts the Keras model to ONNX format. But I computed it already and it's in the `models` folder.

Use the `conversion-environment.yml` for the conversion.

### `test_inference.py`
This script tests the inference using the ONNX model. The sample images are in the images folder.


## Credits
The images used in the web app are from Unsplash. The credits are as follows:

`images/cat.jpg` 
Photo by <a href="https://unsplash.com/@sajadnori?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Sajad Nori</a> on <a href="https://unsplash.com/photos/brown-tabby-cat-in-close-up-photography-s1puI2BWQzQ?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

`images/dog.jpg` 
Photo by <a href="https://unsplash.com/@victor_vector?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Victor Grabarczyk</a> on <a href="https://unsplash.com/photos/black-and-white-short-coated-dog-N04FIfHhv_k?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
  