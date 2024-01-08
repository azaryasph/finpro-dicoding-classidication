# Final Project Image Classification

## 📚 Introduction
In this project I'm developing a Deep Learning model to predict what hand gestrues (Rock, Paper, Scissors) that showed to the model

## 🛠️ Requirements

This project requires the following software and Python libraries:

- Python (3.6 or higher)
- Jupyter Notebook
- TensorFlow (2.0 or higher)
- Keras
- NumPy
- Matplotlib
- Scikit-Learn
- PIL
- os, shutil, zipfile

You can install the required packages using pip:

```bash
pip install jupyter tensorflow keras numpy matplotlib scikit-learn
```

## 🚀 Instruction
To run `Final Project Image Classification.ipynb`: <br>
- first, you have to download the notebook
- then you have to run it on jupyter notebook web so you can insert the picture on the model later

To run `Final_Project_Image_Classification_Colab.ipynb`:
- You can go to this google colaboratory link : https://colab.research.google.com/drive/1-Mcc5wKTCj2Euf8Gbp8Zzy9ZEo8-CtTz?usp=sharing
to download and run on your new jupyter file on colab.
- To Download the notebook on colab :<br> 
    * go to `File` on your upper left 
    * find `Download`
    * Download .ipynb
    * you ready to go to import the downloaded file on your own notebook.

## 📦 Dataset
The Dataset contains image of rock, paper, and siccors hand gestures.
Dataset Download link : https://github.com/azaryasph/finpro-dicoding-classification/raw/main/rockpaperscissors.zip?download=

## 📝 Data Preparation
The images are loaded from a dataset and some preprocessing is performed such as rescaling the images. The dataset is split into training and validation sets in a 60/40 ratio.

## 🏗️ Model Building
The image classification model utilized in this project is a Convolutional Neural Network (CNN), a type of model frequently employed for image classification tasks. This model is implemented using the Sequential Model from the Keras library, which is characterized by its linear stack of layers. This structure makes it an ideal choice for a straightforward stack of layers, where each layer has precisely one input tensor and one output tensor.

The architecture of this CNN model comprises several types of layers:

- Convolutional 2D layers: These layers form the foundation of the model. They create a convolution kernel that interacts with the layer input to generate a tensor of outputs. The model includes four of these layers, each followed by a MaxPooling layer.
  
- MaxPooling layers: These layers serve to downscale the input along its spatial dimensions (height and width). They achieve this by selecting the maximum value over an input window for each dimension along the features axis. The inclusion of these layers after each Convolutional 2D layer helps to reduce the computational cost by decreasing the dimensionality of the input and also helps in making the model invariant to small shifts and distortions.
  
- Dense layers: These are regular densely-connected Neural Network (NN) layers. They output an array of transformed data. The model concludes with three of these fully connected layers.

This combination of layers in the CNN model allows for effective image classification, with each layer performing a specific function that contributes to the overall task.

## 🎯 Model Training
The model is compiled with the 'adam' optimizer and the 'sparse_categorical_crossentropy' loss function. The 'adam' optimizer is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The 'sparse_categorical_crossentropy' loss function is used when there are two or more label classes and the classes are exclusive.

After training the model using the training data and validated using the validation data for several epochs, it achieved an accuracy of 98%. This high accuracy indicates that the model is very effective at predicting the hand gestures from the images.

📈 Results
The model achieved an accuracy of 98.26% on the training data and 99.47% on the validation data.

