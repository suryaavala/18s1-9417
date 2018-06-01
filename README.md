# 18s1-9417

## Overview

The MNIST Data
Hand digit recognition has been a Hello, World! to Deep Learning. It is extremely easy and simple to get started with it and to build a classifier with a 90% accuracy. At the same time, MNIST problem can also be used to dive deeper into advanced Deep Learning concepts such as Convolutional Neural Networks, learning rate, dropouts and batch normalisation. The world record for the MNIST data is at about 99.6% accuracy and the model I have trained should give an accuracy around ~99.5%.

The MNIST data is hosted on Yann LeCun's website.

The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation). This split is very important: it's essential in machine learning that we have separate data which we don't learn from so that we can make sure that what we've learned actually generalizes!

As mentioned earlier, every MNIST data point has two parts: an image of a handwritten digit and a corresponding label. We'll call the images "x" and the labels "y". Both the training set and test set contain images and their corresponding labels; for example the training images are mnist.train.images and the training labels are mnist.train.labels.

Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers:

![alt text](https://www.tensorflow.org/images/MNIST-Matrix.png)

We can flatten this array into a vector of 28x28 = 784 numbers. It doesn't matter how we flatten the array, as long as we're consistent between images. From this perspective, the MNIST images are just a bunch of points in a 784-dimensional vector space, with a very rich structure (warning: computationally intensive visualizations).

Flattening the data throws away information about the 2D structure of the image. Isn't that bad? Well, the best computer vision methods do exploit this structure, and we will in later tutorials. But the simple method we will be using here, a softmax regression (defined below), won't.

The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784]. The first dimension is an index into the list of images and the second dimension is the index for each pixel in each image. Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.

![alt text](https://www.tensorflow.org/images/mnist-train-xs.png)

Each image in MNIST has a corresponding label, a number between 0 and 9 representing the digit drawn in the image.

For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension. In this case, the nth digit will be represented as a vector which is 1 in the nth dimension. For example, 3 would be [0,0,0,1,0,0,0,0,0,0]. Consequently, mnist.train.labels is a [55000, 10] array of floats.

![a;t text](https://www.tensorflow.org/images/mnist-train-ys.png)

We're now ready to actually make our model! [Tensorflow Turotial](https://www.tensorflow.org/get_started/mnist/beginners)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See Running for notes on how to deploy the project on a live system.

### Prerequisites

Following prerequisites are essential for this code to run.

```
git
python3
pip3
numpy
pandas
matplotlib
tensorflow
```

Following commands are just a reference to install the dependencies on a Ubuntu like linux system. Please refer to their respective websites to properly install the dependencies

```
sudo apt-get install git

sudo apt-get install python3

sudo apt-get install python3-pip

pip3 install numpy

pip3 install pandas

pip3 install matplotlib

pip3 install tensorflow
```

### Installing

A step by step series of examples that tell you have to get a development env running

Once the dependencies are installed, clone this github repository onto your local machine as follows

```
git clone git@github.com:suryaavala/18s1-cs9417.git
```

Then cd into the cloned directory

```
cd 18s1-cs9417
```

## Testing the Installation

Run the following commands to test if the code has been installed properly

```
cd code

python3 mnist_nn.py
```

Expected output

```
You did not run the program as designed
Please run the program again as follows(case sensitive Train!=train):
python3 mnist_nn.py train/predict
Exiting the program...
```

If you see the above output, then the installation is successful

## Running

The code can be run to either Train the classifier or to Predict the results using the pretrained classifier

### Train

Inorder to train the classifier run the following code

```
pwd
```

Make sure your current directory is `/18s1-cs9417/code/`

```
python3 mnist_nn.py train
```

The classifier takes around ~60min to run on a 1.7GHz Intel Dual Core i7 processor.
During the program execution you'll be seeing something like follows:

```
suryatherisingstar@instance-1:~/18s1-cs9417/dev$ python3 mnist_nn.py train
Extracting data from yann.lecun.com/exdb/mnist/
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Data extracted
step 0, training accuracy 0.16
step 100, training accuracy 0.9
step 200, training accuracy 0.96
step 300, training accuracy 0.96
step 400, training accuracy 0.88
step 500, training accuracy 0.96
step 600, training accuracy 0.92
step 700, training accuracy 0.92
step 800, training accuracy 0.96
step 900, training accuracy 1
step 1000, training accuracy 0.94
....
....
....
```

The training is terminated at step 30000 and you will see something like

```
step 29000, training accuracy 1
step 29100, training accuracy 1
step 29200, training accuracy 1
step 29300, training accuracy 1
step 29400, training accuracy 1
step 29500, training accuracy 1
step 29600, training accuracy 1
step 29700, training accuracy 1
step 29800, training accuracy 1
step 29900, training accuracy 1
test accuracy 0.9927
Model saved in file: ./model/model.ckpt
```

### Predict

Inorder to predict the labels using create a csv like this [test.csv](https://www.kaggle.com/c/digit-recognizer/data) and place it in

```
18s1-cs9417/code/data
```

And then run the following command

```
python3 mnist_nn.py predict
```

Predicted results will be saved in

```
18s1-cs9417/submission_softmax.csv
```

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/suryaavala/18s1-cs9417/tags).

## Authors

* **Surya Avala** - _Initial work_ - [SuryaAvala](https://github.com/suryaavala)

See also the list of [contributors](https://github.com/suryaavala/18s1-cs9417/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Tensorflow](https://tensorflow.org)
* [Martin Görner](https://twitter.com/martin_gorner)
* [Tensorflow Turotial](https://www.tensorflow.org/get_started/mnist/beginners)

##### < / > With :heart: Using [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/)
