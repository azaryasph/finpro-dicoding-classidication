# Final Project Image Classification

## Introduction
In this project I'm developing a Deep Learning model to predict what hand gestrues (Rock, Paper, Scissors) that showed to the model


## Instruction
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

## Dataset
The Dataset contains image of rock, paper, and siccors hand gestures.
Dataset Download link : https://github.com/azaryasph/finpro-dicoding-classification/raw/main/rockpaperscissors.zip?download=

## Model

The model used in this project is a Sequential Model from the Keras library. This model is a linear stack of layers that is very suitable for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

The model was trained using a dataset of images of rock, paper, and scissors hand gestures. The images were preprocessed and then fed into the model. The model consists of several layers including Convolutional 2D layers, MaxPooling layers, and Dense layers.

The Convolutional 2D layers are used to create a convolution kernel that is convolved with the layer input to produce a tensor of outputs. The MaxPooling layers are used to downscale the input along its spatial dimensions (height and width) by taking the maximum value over an input window for each dimension along the features axis. The Dense layers are regular densely-connected NN layers which output the array of transformed data.

The model was compiled with the 'adam' optimizer and the 'sparse_categorical_crossentropy' loss function. The 'adam' optimizer is an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The 'sparse_categorical_crossentropy' loss function is used when there are two or more label classes and the classes are exclusive.

After training the model for several epochs, it achieved an accuracy of 98%. This high accuracy indicates that the model is very effective at predicting the hand gestures from the images.