import gzip
import numpy as np

def get_data(inputs_file_path, labels_file_path, num_examples):
    """
    Takes in an inputs file path and labels file path, unzips both files, 
    normalizes the inputs, and returns (NumPy array of inputs, NumPy array of labels). 
    
    Read the data of the file into a buffer and use 
    np.frombuffer to turn the data into a NumPy array. Keep in mind that 
    each file has a header of a certain size. This method should be called
    within the main function of the model.py file to get BOTH the train and
    test data. 
    
    If you change this method and/or write up separate methods for 
    both train and test data, we will deduct points.
    
    :param inputs_file_path: file path for inputs, e.g. 'MNIST_data/t10k-images-idx3-ubyte.gz'
    :param labels_file_path: file path for labels, e.g. 'MNIST_data/t10k-labels-idx1-ubyte.gz'
    :param num_examples: used to read from the bytestream into a buffer. Rather 
    than hardcoding a number to read from the bytestream, keep in mind that each image
    (example) is 28 * 28, with a header of a certain number.
    :return: NumPy array of inputs (float32) and labels (uint8)
    """
    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        # get rid of header
        bytestream.read(16) 
        # Creates an array to store inputs
        # Go image by image, adding as a row, normalizing values
        # np.float32, cast
        X = (1/255.0)*(np.reshape(np.frombuffer(bytestream.read(
            784*num_examples), dtype=np.uint8), (-1, 784)).astype(np.float32))
    
    with open(labels_file_path, 'rb') as fl, gzip.GzipFile(fileobj=fl) as bytestream:
        bytestream.read(8) # get rid of header
        # use np.uint8
        L = np.reshape(np.frombuffer(bytestream.read(num_examples), dtype=np.uint8), (-1, 1))
    return X, L

## NOTE you may want to introduce batching method here
def get_next_batch(X, L, start_index, batch_size):
    """
    Returns a slice of data and a slice of labels, given a starting index
    for the slice and the batch size. These two slices represent a batch of data
    and a abatch of labels. MAKE SURE NOT TO CALL IN MAIN
    """
    return X[start_index: start_index + batch_size], L[start_index: start_index + batch_size]
