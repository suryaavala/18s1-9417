# 18s1-9417

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See Running for notes on how to deploy the project on a live system.

### Prerequisites

Following prerequisites are essential for this code to run.

```
git
python3
pip3
```

and packages under `requirements.txt`

Following commands are just a reference to install the dependencies on a Ubuntu like linux system. Please refer to their respective websites to properly install the dependencies

```
sudo apt-get install git

sudo apt-get install python3

sudo apt-get install python3-pip

pip3 -r requirements.txt
```

### Installing

A step by step series of examples that tell you have to get a development env running

Once the dependencies are installed, download the submission directory which was submitted

Then cd into the directory

```
cd submission
```

## Testing the Installation

Run the following commands to test if the code has been installed properly

```
cd code

python3 assignment.py
```

Expected output

```
Please add an input argument while running the program. 1st and 2nd arguments are mandatory and 3rd is option. 1st args: <train/test>; 2nd args: <mnist/emnist>;
Exiting program..
```

If you see the above output, then the installation is successful

## Running

The code can be run to either Train the classifier or to Predict the results using the pretrained classifier

## Data

Unzip the data directory inside code and copy the data files inside `data/<mnist/emnist>` into `MNIST_data` directory depending on whether you are planning to run mnist (handwritten digits) or emnist (hand written digits plus alphabets).

You have to do copy the right data into `MNIST_data` everytime you are going to train the classifier. The data files for both mnist and emnist are named exactly the same.

```
$ ls ../../code/MNIST_data/
t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz  train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
```

### Train

Make sure you copied the right(mnist/emnist whichever you are planning to run) data into `MNIST_data` dir.

Inorder to train the classifier run the following code

```
pwd
```

Make sure your current directory is `/submission/code/`

```
python3 assignmetn.py train <mnist/emnist>
```

The classifier takes around ~60min to run on a 1.7GHz Intel Dual Core i7 processor.

### Test

Make sure you copied the right(mnist/emnist whichever you are planning to run) data into `MNIST_data` dir.

Inorder to test the classifier run the following code

```
pwd
```

Make sure your current directory is `/submission/code/`

```
python3 assignmetn.py test <mnist/emnist>
```

## Acknowledgments

* [Tensorflow](https://tensorflow.org)
* [Martin Görner](https://twitter.com/martin_gorner)
* [Tensorflow Turotial](https://www.tensorflow.org/get_started/mnist/beginners)

##### < / > With :heart: Using [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/)
