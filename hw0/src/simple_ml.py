import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    return x + y


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
    assert(int.from_bytes(image_bin[0:4], 'big') == 2051)
    image_cnt = int.from_bytes(image_bin[4:8], 'big')
    row_cnt = int.from_bytes(image_bin[8:12], 'big')
    col_cnt = int.from_bytes(image_bin[12:16], 'big')
    pixels = []
    pixels_iter = struct.iter_unpack('>B', image_bin[16:])
    for pixel in pixels_iter:
      pixels.append(pixel[0] / 255.0)
    # Check the num of pixels
    assert(len(pixels) == image_cnt * row_cnt * col_cnt)
    # Generate the array of images
    images = np.array(pixels, dtype=np.float32).reshape(image_cnt, row_cnt * col_cnt)

    with gzip.open(label_filename) as f:
      label_bin = f.read()
    
    # Check the magic number
    assert(int.from_bytes(label_bin[0:4], 'big') == 2049)
    label_cnt = int.from_bytes(label_bin[4:8], 'big')
    labels = list(label_bin[8:])
    # Check the num of labels
    assert(len(labels) == label_cnt)
    # Generate the array of labels
    labels = np.array(labels, dtype=np.uint8)

    assert(len(images) == len(labels))
    return images, labels


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    return np.average(np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(y.size), y])


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    iterations = y.size // batch
    for i in range(iterations):
      x_ = X[i * batch : min((i + 1) * batch, y.size), :]
      y_ = y[i * batch : min((i + 1) * batch, y.size)]
      logits_exp = np.exp(np.matmul(x_, theta))
      normalized = logits_exp / np.sum(logits_exp, axis=1, keepdims=True)
      one_hot_y = np.zeros((y_.size, y.max()+1))
      one_hot_y[np.arange(y_.size), y_] = 1
      gradient = np.matmul(x_.T, normalized - one_hot_y) / batch
      theta -= lr * gradient
    return

    


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    iterations = y.size // batch
    for i in range(iterations):
        x_ = X[i * batch : min((i + 1) * batch, y.size), :]
        y_ = y[i * batch : min((i + 1) * batch, y.size)]
        Z1 = np.matmul(x_, W1)
        Z1[Z1 < 0] = 0 #RELU
        G2 = np.exp(np.matmul(Z1, W2))
        G2 = G2 / np.sum(G2, axis=1, keepdims=True)
        Y = np.zeros((batch, y.max() + 1))
        Y[np.arange(batch), y_] = 1
        G2 -= Y
        G1 = np.zeros_like(Z1)
        G1[Z1 > 0] = 1
        G1 = G1 * np.matmul(G2, W2.T)
        grad1 = x_.T @ G1 / batch
        grad2 = Z1.T @ G2 / batch
        W1 -= lr * grad1
        W2 -= lr * grad2



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
