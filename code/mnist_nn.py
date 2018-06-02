import tensorflow as tf
import pandas as pd
import numpy as np
import sys

#NOTE Checking the arg for training or test
if len(sys.argv) != 2:
    print("You did not run the program as designed\nPlease run the program again as follows(case sensitive Train!=train): ")
    print("python3 mnist_nn.py train/predict")
    sys.exit("Exiting the program...")
else:
    train_model = bool(sys.argv[1]=="train")


#NOTE Variable declaration
x = tf.placeholder(tf.float32,[None, 784]) #variable for features
y_ = tf.placeholder(tf.float32, [None, 62]) #variable for labels


#NOTE function declaration to genrate Weight/bias variables
def weight_variable(shape):
    '''
    Function to initialize weigth matrix variable
    Input: Takes shape of the desired variable
    Output: Outputs the variable of that shape
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Function to initialize bias matrix variable
    Input: Takes shape of the desired variable
    Output: Outputs the variable of that shape
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#NOTE function declaration to generate convolutional and maxpooling layers
def conv2d(x,W):
    '''
    Function to create a 2d convolutional layer
    Input: Takes feature vector (x) and weight matrix (W)
    Output: Returns a conv
    '''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    '''
    Function to create a maxpooling layer
    Input: Takes feature vector (x)
    Output: Returns a max pool
    '''
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#NOTE: convolutional layer 1 on 5x5 pixels with 32 output features
W_conv1 = weight_variable([5,5,1,32])   #weights for conv1
b_conv1 = bias_variable([32])   #biases for conv1

x_image = tf.reshape(x, [-1, 28, 28, 1])    #reshaping x into 28x28

#ReLU is used as activation
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1) #output from conv layer1
h_pool1 = max_pool_2x2(h_conv1)     #max pool 1

#NOTE: convolutional layer 2 on 5x5 pixels with 32 input and 64 output features
W_conv2 = weight_variable([5,5,32,64])  #weights for conv2
b_conv2 = bias_variable([64]) #biases for conv2

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2) #output from conv layer1
h_pool2 = max_pool_2x2(h_conv2) #max pool2

#NOTE: convolutional layer 2 on 5x5 pixels with 32 input and 64 output features
W_conv3 = weight_variable([5,5,64,128])  #weights for conv2
b_conv3 = bias_variable([128]) #biases for conv2

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3)+b_conv3) #output from conv layer1
h_pool3 = max_pool_2x2(h_conv3) #max pool2

W_fc1 = weight_variable([4*4*128, 1024])
b_fc1 = bias_variable([1024])

print(h_pool2.shape, h_pool3.shape)
h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1)+b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 62])
b_fc2 = bias_variable([62])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#NOTE variable declaration for variables used in training
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))  #cross entropy

#adam optimizer with learning rate of 1e-4 to minimize cross_entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#list of booleans comparing te correct y and predicted labels
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#NOTE Train model function
if train_model:
    '''
    Training the model
    '''
    #NOTE Downloading the dataset
    print("Extracting data from yann.lecun.com/exdb/mnist/")
    from mnist import read_data_sets
    mnist = read_data_sets("MNIST_data/", one_hot=True)
    print("Data extracted")

    #Running the session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #training the model
    for i in range(42000):
      batch = mnist.train.next_batch(50)
      if i%1000 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      if i%10000 == 0:
        accuracies = []
        for i in range(2327):
            testSet = mnist.test.next_batch(50)
            accu = accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0})
            #print("test accuracy %g"%accu)
            accuracies.append(accu)
        
        print('Total accuracy: {}'.format(str(sum(accuracies)/len(accuracies))))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.45})

    # print("test accuracy %g"%accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    #https://stackoverflow.com/questions/39076388/tensorflow-deep-mnist-resource-exhausted-oom-when-allocating-tensor-with-shape?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    for i in range(10):
        testSet = mnist.test.next_batch(50)
        print("test accuracy %g"%accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0}))
    #test accuracy 0.9918

    #saving the nn
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/model_62/model_2/model.ckpt")
    print("Model saved in file: %s" % save_path)
    sess.close()


else:
    saver = tf.train.Saver()
    #restore
    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, "./model/model_62/model.ckpt")
        print("Model restored.")
        #NOTE Downloading the dataset
        print("Extracting data from yann.lecun.com/exdb/mnist/")
        from mnist import read_data_sets
        mnist = read_data_sets("MNIST_data/", one_hot=True)
        print("Data extracted")
        accuracies = []
        for i in range(2327):
            testSet = mnist.test.next_batch(50)
            accu = accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0})
            print("test accuracy %g"%accu)
            accuracies.append(accu)
        
        print('Total accuracy: {}'.format(str(sum(accuracies)/len(accuracies))))
        #test accuracy 0.9918
        # predict = tf.argmax(y_conv,1)
        # # read test data from CSV file
        # test_images = pd.read_csv('./data/test.csv').values
        # test_images = test_images.astype(np.float)
        # print('test_images({0[0]},{0[1]})'.format(test_images.shape))
        # # using batches is more resource efficient
        # predicted_lables = np.zeros(test_images.shape[0])
        # for i in range(0,test_images.shape[0]//100):
        #     predicted_lables[i*100 : (i+1)*100] = predict.eval(feed_dict={x: test_images[i*100 : (i+1)*100],keep_prob: 1.0})

        # print('predicted_lables({0})'.format(len(predicted_lables)))

        # # save results
        # np.savetxt('submission_softmax.csv',np.c_[range(1,len(test_images)+1),predicted_lables], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')



        sess.close()
