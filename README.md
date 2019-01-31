# Road Segmentation from Satellite Images using Convolutional Neural Network

The aim of this project was to train a CNN classifier to segment roads on Google Maps images, i.e. assign a label {road=1, background=0} to each 16x16 pixel patch images. 

## Getting the csv file

First, you have to put content of the zip file in a folder, as well as the "data" folder that was given for this project.

From a console, you should be able to get the csv file that gave us our best result (0.737) by running the following command (in the directory of run.py file): "python run.py". Be aware of having the correct path to load train and test files.

## Structure of the run.py file

You will find at the beginning of run.py two boolean variables: charge_model and charge_predic. Set charge_model to False if you want to train the 3 networks from scratch, otherwise the code charges the 3 weights. Set charge_predic to False if you want to generate the predictions from scratch, otherwise the code will charge the 3 predictions files and create the csv file. Both boolean are set to True by default (everything charged).

First part of the run.py file consists of creating our 3 models (charged from cnn_model.py files), and then preparing the data by extracting patches from input images, doing some operations on them, and finally by doing padding before training the networks.

Second part consists of creating predictions using the 3 models we trained with a majority voting rule system that you can find inside the "ensemble_model" function (that is in helpers.py).

## Structure of the code

Please find in the zip file our 3 cnn_models (in cnn_modelX.py, X = 1,2,3), helpers.py which contains all the functions that we used during our project (to train all the models), and run.py that is given the submission.csv file which gave us our best f-score. Note that the cnn_modelX.py files are very similar, but that is for a sake of simplicity and readability of our code. Note also that in helpers.py, we put some of the functions that were given at the beginning of the project.

## Built With

- [Numpy](http://www.numpy.org/)- Package for scientific computing with Python

- [Matplotlib.image](https://matplotlib.org/api/image_api.html)- Supports basic image loading, rescaling and display operations. 

- [Keras](https://keras.io/)- Keras is a high-level neural networks API, written in Python and capable of running on top of [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), or [Theano](https://github.com/Theano/Theano).  


## Authors

- **Jangwon Park**
- **Jean-Baptiste Beau**
- **Frédéric Myotte**

