"""hw1/apps/simple_ml.py"""

import needle as ndl
import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filename) as f:
        image_bin = f.read()

    # Check the magic number
    assert (int.from_bytes(image_bin[0:4], 'big') == 2051)
    image_cnt = int.from_bytes(image_bin[4:8], 'big')
    row_cnt = int.from_bytes(image_bin[8:12], 'big')
    col_cnt = int.from_bytes(image_bin[12:16], 'big')
    pixels = []
    pixels_iter = struct.iter_unpack('>B', image_bin[16:])
    for pixel in pixels_iter:
        pixels.append(pixel[0] / 255.0)
    # Check the num of pixels
    assert (len(pixels) == image_cnt * row_cnt * col_cnt)
    # Generate the array of images
    images = np.array(pixels, dtype=np.float32).reshape(
        image_cnt, row_cnt * col_cnt)

    with gzip.open(label_filename) as f:
        label_bin = f.read()

    # Check the magic number
    assert (int.from_bytes(label_bin[0:4], 'big') == 2049)
    label_cnt = int.from_bytes(label_bin[4:8], 'big')
    labels = list(label_bin[8:])
    # Check the num of labels
    assert (len(labels) == label_cnt)
    # Generate the array of labels
    labels = np.array(labels, dtype=np.uint8)

    assert (len(images) == len(labels))
    return images, labels


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    return ndl.summation(ndl.log(ndl.summation(ndl.exp(Z), axes=(1,))) - ndl.summation(Z * y_one_hot, axes=(1,))) / Z.shape[0]


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    iterations = (y.size + batch - 1) // batch
    for i in range(iterations):
        x = ndl.Tensor(X[i * batch : (i+1) * batch, :])
        Z = ndl.relu(x.matmul(W1)).matmul(W2)
        yy = y[i * batch : (i+1) * batch]
        y_one_hot = np.zeros((batch, y.max() + 1))
        y_one_hot[np.arange(batch), yy] = 1
        y_one_hot = ndl.Tensor(y_one_hot)
        loss = softmax_loss(Z, y_one_hot)
        loss.backward()
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())
    return W1, W2


# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
