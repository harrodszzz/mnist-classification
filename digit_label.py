import mnist_rot
import numpy as np
import tensorflow as tf 

# load data
X,y = mnist_rot.load_mnist_rot(which='train')

# make the labels one-hot
y_size = y.size
y_oneHot = np.zeros([y_size,10])
for i in range(y_size):
	num = y[i]
	y_oneHot[i,num] = 1

# test 
X_test,y_test = mnist_rot.load_mnist_rot(which='test')
# make the labels one-hot
y_test_size = y_test.size
y_test_oneHot = np.zeros([y_test_size,10])
for i in range(y_test_size):
	num = y_test[i]
	y_test_oneHot[i,num] = 1

# define batch fcn
def batch(images,labels,batch_size):
	sizeOfSamples = images.shape[0]
	rand = np.random.random(batch_size) * sizeOfSamples
	rand_ind = rand.astype(int)
	return images[rand_ind,:],labels[rand_ind,:]

# CNN fcn define
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Placeholders
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None,10])

# --- CNN build start --- #

## Build
# 1st layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# 2nd layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# FC layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
# Readout layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
# --- CNN build end --- #

## Train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
	x_batch, y_batch = batch(X,y_oneHot,batch_size=50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
		print("step %d, training accuracy %g" %(i, train_accuracy))
	train_step.run(feed_dict={x: x_batch, y_:y_batch, keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, y_: y_test_oneHot, keep_prob: 1.0}))
