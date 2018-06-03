import tensorflow as tf
import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Please add an input argument while running the program. 1st and 2nd arguments are mandatory and 3rd is option. 1st args: <train/test>; 2nd args: <mnist/emnist>;")
    sys.exit("Exiting program..")
else:
    train = bool(sys.argv[1] == "train")
    data = sys.argv[2]
    architecture = sys.argv[-1]


def init_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def init_weights(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def init_conv_layer(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def init_pool_layer(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
if data == 'mnist':
    y_ = tf.placeholder(tf.float32, [None, 10])
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
elif data == 'emnist':
    y_ = tf.placeholder(tf.float32, [None, 62])
    from import_data import read_data_sets
if architecture == '2layer' or data == 'mnist':
    W_conv1 = init_weights([5, 5, 1, 32])
    b_conv1 = init_bias([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(init_conv_layer(x_image, W_conv1) +
                         b_conv1)
    h_pool1 = init_pool_layer(h_conv1)

    W_conv2 = init_weights([5, 5, 32, 64])
    b_conv2 = init_bias([64])

    h_conv2 = tf.nn.relu(init_conv_layer(h_pool1, W_conv2)+b_conv2)
    h_pool2 = init_pool_layer(h_conv2)

    W_fc1 = init_weights([7*7*64, 1024])
    b_fc1 = init_bias([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = init_weights([1024, 10])
    b_fc2 = init_bias([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
else:
    W_conv1 = init_weights([5, 5, 1, 32])
    b_conv1 = init_bias([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(init_conv_layer(x_image, W_conv1) +
                         b_conv1)
    h_pool1 = init_pool_layer(h_conv1)

    W_conv2 = init_weights([5, 5, 32, 64])
    b_conv2 = init_bias([64])

    h_conv2 = tf.nn.relu(init_conv_layer(h_pool1, W_conv2)+b_conv2)
    h_pool2 = init_pool_layer(h_conv2)

    W_conv3 = init_weights([5, 5, 64, 128])
    b_conv3 = init_bias([128])

    h_conv3 = tf.nn.relu(init_conv_layer(h_pool2, W_conv3)+b_conv3)
    h_pool3 = init_pool_layer(h_conv3)

    W_fc1 = init_weights([4*4*128, 1024])
    b_fc1 = init_bias([1024])

    h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1)+b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = init_weights([1024, 62])
    b_fc2 = init_bias([62])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("extracting data")
mnist = read_data_sets("MNIST_data/", one_hot=True)
print("extraction done")

if train:
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(450000):
        batch = mnist.train.next_batch(50)
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        if i % 10000 == 0:
            accuracies = []
            for i in range(2327):
                testSet = mnist.test.next_batch(50)
                accu = accuracy.eval(
                    feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0})
                accuracies.append(accu)

            print('Total accuracy: {}'.format(
                str(sum(accuracies)/len(accuracies))))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.45})
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()
else:
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/model.ckpt")
        print("Model restored")
        accuracies = []
        for i in range(2327):
            testSet = mnist.test.next_batch(50)
            accu = accuracy.eval(
                feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0})
            print("test accuracy %g" % accu)
            accuracies.append(accu)

        print('Total accuracy: {}'.format(str(sum(accuracies)/len(accuracies))))
        sess.close()
