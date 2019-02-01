# Road Segmentation from Satellite Images using Convolutional Neural Network

The aim of this project is to train a CNN classifier to segment roads on Google Maps images, i.e. assign a label {road=1, background=0} to each 16x16 pixel patch images. This task is done with Keras using both Tensorflow and Theano as backend. For full details of our work, refer to `Report.pdf`.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The required environment for running the code and reproducing the results is a computer with a valid installation of Python 3. More specifically, [Python 3.6](https://docs.python.org/3.6/) is used.

Besides that (and the built-in Python libraries), the following packages are used and have to be installed:

* [Keras 2.2.4](https://keras.io/) `pip3 install --user keras==2.2.4`
* [NumPy 1.13.3](http://www.numpy.org) `pip3 install --user numpy==1.13.3`
* [Matplotlib 2.0.2](https://matplotlib.org). `pip3 install --user matplotlib==2.0.2`
* [Pandas 0.23.4](https://pandas.pydata.org) `pip install --user pandas==0.23.4`
* [PIL 5.2.0](https://pillow.readthedocs.io/en/latest/releasenotes/5.2.0.html) `pip install --user pil==5.2.0`
* [Tensorflow 1.12.0](https://www.tensorflow.org) `pip install --user tensorflow==1.12.0`
* [Theano 1.0.3](http://deeplearning.net/software/theano_versions/dev/library/index.html) `pip install --user theano==1.0.3`

## Project Structure

The `project/` directory has the following folder (and file) structure:

* `data/`. Directory containing original dataset
    * `training/` Contains 50 400x400 pixel training images in .png format.
    * `test_set_images/` Contains 50 608x608 pixel test images in .png format.

* `models/`. Contains python files describing three different CNN architectures.
* `predictions/`. Contains saved predictions by each CNN architecture on the test images in .npy format.
* `weights/`. Contains weights of each CNN architecture trained on the training images in .hdf5 format.

* `Report.pdf`

## How to execute the files.
Neural networks in general are very complex and may take up to several hours to train. Therefore, to enable convenient replication of our work, we have saved the trained weights as well as the predictions made on the test images.

### Getting the .csv file

First, you have to put content of the zip file in a folder, as well as the "data" folder that was given for this project.

From a console, you should be able to get the csv file that gave us our best result (0.737) by running the following command (in the directory of run.py file): "python run.py". Be aware of having the correct path to load train and test files.

### Structure of the run.py file

You will find at the beginning of run.py two boolean variables: charge_model and charge_predic. Set charge_model to False if you want to train the 3 networks from scratch, otherwise the code charges the 3 weights. Set charge_predic to False if you want to generate the predictions from scratch, otherwise the code will charge the 3 predictions files and create the csv file. Both boolean are set to True by default (everything charged).

First part of the run.py file consists of creating our 3 models (charged from cnn_model.py files), and then preparing the data by extracting patches from input images, doing some operations on them, and finally by doing padding before training the networks.

Second part consists of creating predictions using the 3 models we trained with a majority voting rule system that you can find inside the "ensemble_model" function (that is in helpers.py).

### Structure of the code

Please find in the zip file our 3 cnn_models (in cnn_modelX.py, X = 1,2,3), helpers.py which contains all the functions that we used during our project (to train all the models), and run.py that is given the submission.csv file which gave us our best f-score. Note that the cnn_modelX.py files are very similar, but that is for a sake of simplicity and readability of our code. Note also that in helpers.py, we put some of the functions that were given at the beginning of the project.

## Authors

- **Jangwon Park**
- **Jean-Baptiste Beau**
- **Frédéric Myotte**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
