# this file is to create the experiment dataset.
# images are rotated 2 and 4, and labels are merely their rotation degrees

import dataset
import numpy as np
import matplotlib.pyplot as plt 
import cv2

# load data
X,y = dataset.load_mnist()

y_size = y.size
X_n = np.zeros([y_size,784])
y_n = np.zeros([y_size,1])

# pick up digits 2 and 4
index = 0
for i in range(y.size):
	if y[i] == 2 or y[i] == 4:
		X_n[index,:] = X[i,:]
		y_n[index,:] = y[i]
		index += 1
X_n = np.delete(X_n,range(index,y_size+1),0)
y_n = np.delete(y_n,range(index,y_size+1),0)

cols = 28
rows = 28

num = X_n.shape[0]

X_r = np.zeros([num*12,784])
y_r = np.zeros([num*12,12])

# Make new dataset
for i in range(11):
	M = cv2.getRotationMatrix2D((cols/2,rows/2),30*i,1)
	for j in range(num):
		x = X_n[j,:].reshape(28,28)
		dst = cv2.warpAffine(x,M,(cols,rows))
		X_r[i*num+j,:] = dst.reshape(1,784)
		y_r[i*num+j,i] = 1

np.save('rot_images',X_r)
np.save('rot_labels',y_r)
