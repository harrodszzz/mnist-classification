# mnist_classification
this repository is to classify mnist datas using labels of digits and rotating degrees
to explore whether a new manifold exists to classify two digits in rotation parameters

this repository consists of two parts, labels of digits and of rotation parameters

## labels of digits
digits_labels.py is the main code

the input data are mnist_rot(mnist data after rotation of different angles),
labels of data are the digits of images

I didn't implement data augmentation yet (06/19/2017)

test accuracy is 0.945

## labels of rotation
test accuracy is around 0.26, this method did not work.(06/20/2017)
traditional 2-layer nn, accuracy, 0.88
