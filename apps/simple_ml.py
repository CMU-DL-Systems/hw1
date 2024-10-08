"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


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
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, 'rb') as image_file:
        image_data = image_file.read()
        num_images = int.from_bytes(image_data[4:8])
        image_array = np.frombuffer(image_data[16:], dtype=np.uint8)
        image_array = image_array.reshape(num_images, -1)
        image_array = image_array.astype(np.float32)
        min_val = image_array.min()
        max_val = image_array.max()
        image_array = (image_array - min_val) / (max_val - min_val)

    with gzip.open(label_filename, 'rb') as label_file:
        label_data = label_file.read()
        label_array = np.frombuffer(label_data[8:], dtype=np.uint8)

    return image_array, label_array
    ### END YOUR CODE


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
    ### BEGIN YOUR SOLUTION
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    z_y = ndl.summation(Z * y_one_hot, axes=(1,))
    return ndl.summation(log_sum_exp - z_y) / Z.shape[0]
    ### END YOUR SOLUTION


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

    ### BEGIN YOUR SOLUTION
    index = 0
    while index < X.shape[0]:
        X_batch = X[index:index+batch, :]
        y_batch = y[index:index+batch]
        index += batch
        m = X_batch.shape[0]

        X_batch = ndl.Tensor(X_batch)
        y_batch = ndl.Tensor(np.eye(W2.shape[1])[y_batch])

        Z = ndl.matmul(ndl.relu(ndl.matmul(X_batch, W1)), W2)
        loss = softmax_loss(Z, y_batch)
        loss.backward()

        W1_grad = W1.grad.numpy()
        W2_grad = W2.grad.numpy()
        W1 = ndl.Tensor(W1.cached_data - lr * W1_grad)
        W2 = ndl.Tensor(W2.cached_data - lr * W2_grad)

    return W1, W2


    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
