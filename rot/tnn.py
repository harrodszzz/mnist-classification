import numpy as np
import tensorflow as tf

# load data from .npy
X = np.load('rot_images.npy')
y = np.load('rot_labels.npy')

# split to train and test
train_ratio = 0.3
size = X.shape[0]

# train set
train_num = int(train_ratio * size)
rand = np.random.random([train_num])*size
train_ind = rand.astype(int)
X_tr = X[train_ind,:]
y_tr = y[train_ind,:]

# test set
total = range(size)
test_ind = np.array(list(set(total) - set(train_ind)))
X_te = X[test_ind,:]
y_te = y[test_ind,:]

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
y_ = tf.placeholder(tf.float32, shape=[None,12])


# --- CNN build start --- #

# 1st fc layer
W_fc1 = weight_variable([784,1024])
b_fc1 = weight_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(x,W_fc1) + b_fc1)
# 2nd fc layer
W_fc2 = weight_variable([1024,1024])
b_fc2 = weight_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)
# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob)
# Readout layer
W_fc3 = weight_variable([1024,12])
b_fc3 = bias_variable([12])
y_conv = tf.matmul(h_fc2_drop,W_fc3) + b_fc3

# --- CNN build end --- #

## Train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
	x_batch, y_batch = batch(X_tr,y_tr,batch_size=50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
		print("step %d, training accuracy %g" %(i, train_accuracy))
	train_step.run(feed_dict={x: x_batch, y_:y_batch, keep_prob: 0.4})

print("test accuracy %g" % accuracy.eval(feed_dict={x: X_te, y_: y_te, keep_prob: 1.0}))


