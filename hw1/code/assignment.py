from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data
from preprocess import get_next_batch

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    sizes. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = 0.5

        # Initialize weights and biases
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros((1,self.num_classes))

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities for each class per image # (batch_size x 10)
        """
        # Linear layer and exp of softmax
        probabilities = np.exp(np.matmul(inputs, self.W) + self.b)
        # divide each row by sum_k(e^jk)
        probabilities = (1/np.sum(probabilities, axis=1, keepdims=True))*probabilities
        return probabilities
    
    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step). 
        NOTE: This function is not actually used for gradient descent 
        in this assignment, but is a sanity check to make sure model 
        is learning.

        :param probabilities: matrix that contains the probabilities 
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        return np.sum(-np.log(probabilities[np.arange(0, self.batch_size), labels])/self.batch_size)
    
    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. The learning 
        algorithm for updating weights and biases mentioned in 
        class works for one image, but because we are looking at 
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.

        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights, and gradient for biases
        """
        # 
        indicator = np.zeros((labels.size, self.num_classes))
        indicator[np.arange(labels.size), labels] = 1
        # ASK ABOUT THE AVERAGING, check axes etc.
        # make a separate bias gradient to return
        weights_gradient = (1/self.batch_size) * (np.matmul(inputs.transpose(), (probabilities-indicator)))

        bias_input = np.ones((self.batch_size, 1))
        bias_gradient = (1/self.batch_size) * (np.matmul(bias_input.transpose(), (probabilities-indicator)))

        return weights_gradient, bias_gradient
    
    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: batch accuracy (float [0,1])
        """
        # Create a matrix where I_i,j = 1 if j = c, 0 otherwise
        indicator = np.zeros_like(probabilities)
        indicator[np.arange(len(probabilities)), probabilities.argmax(1)] = 1
        # Sum the elements at each correct label, divide by the total number in the batch
        return np.sum(indicator[np.arange(0, labels.size), np.reshape(labels, (-1, 1))])/labels.size

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''

        self.W = self.W - (self.learning_rate * (gradW))
        self.b = self.b - (self.learning_rate * (gradB))

        return None
    
    def train(model, train_inputs, train_labels):
        '''
        Trains the model on all of the inputs and labels.
        :param model: the initialized model to use for the forward 
        pass and backward pass
        
        :param train_inputs: train inputs (all inputs to use for training)
        :param train_inputs: train labels (all labels to use for training)
        :return: None
        '''
        # Iterate over the training inputs and labels, in model.batch_size increments and do forward pass
        losses = np.empty(int(train_labels.size / model.batch_size))
        for i in range(0, train_labels.size, model.batch_size):
            # Get the batched inputs and labels
            inputs, labels = get_next_batch(train_inputs, train_labels, i, model.batch_size)
            # Forward pass, create the probabilities matrix
            probabilities = model.call(inputs)   
            # Calculate the gradients
            gradW, gradB = model.back_propagation(inputs, probabilities, labels)
            # Update the weights and biases
            model.gradient_descent(gradW, gradB)
            # Visualize losses
            losses[int(i/model.batch_size)] = model.loss(probabilities, labels)
        visualize_loss(losses)
        return None
        

    def test(model, test_inputs, test_labels):
        """
        Tests the model on the test inputs and labels. For this assignment, 
        the inputs should be the entire test set, but in the future we will
        ask you to batch it instead.
        
        :param test_inputs: MNIST test data (all images to be tested)
        :param test_labels: MNIST test labels (all corresponding labels)
        :return: accuracy (float [0,1])
        """
        # Return accuracy across testing set
        return model.accuracy(model.call(test_inputs), test_labels)

def visualize_loss(losses):
    """
    NOTE: DO NOT EDIT

    Uses Matplotlib to visualize loss per batch. Call this in train().
    When you observe the plot that's displayed, think about:
    1. What does the plot demonstrate or show?
    2. How long does your model need to train to reach roughly its best accuracy so far, 
    and how do you know that?
    Optionally, add your answers to README!
    
    :param losses: an array of loss value from each batch of train
    :return: does not return anything, a plot should pop-up
    """
    x = np.arange(1, len(losses)+1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses)
    plt.show()

def visualize_results(image_inputs, probabilities, image_labels):
    """
    NOTE: DO NOT EDIT

    Uses Matplotlib to visualize the results of our model.
    
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()
    :return: does not return anything, a plot should pop-up 
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    
    :return: None
    '''
    # load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels
    training_data = get_data('C:/Users/moasi/Desktop/CSCI 1470/hw1-mnist-liamoconno/data/train-images-idx3-ubyte.gz', 
    'C:/Users/moasi/Desktop/CSCI 1470/hw1-mnist-liamoconno/data/train-labels-idx1-ubyte.gz', 60000)

    test_data = get_data('C:/Users/moasi/Desktop/CSCI 1470/hw1-mnist-liamoconno/data/t10k-images-idx3-ubyte.gz', 
    'C:/Users/moasi/Desktop/CSCI 1470/hw1-mnist-liamoconno/data/t10k-labels-idx1-ubyte.gz', 10000)
    
    train_inputs = training_data[0]
    train_labels = training_data[1]

    test_inputs = test_data[0]
    test_labels = test_data[1]
    # Create Model
    model = Model()
    # Train model by calling train() ONCE on all data
    model.train(train_inputs, train_labels)
    # Test the accuracy by calling test() after running train()
    print(model.test(test_inputs, test_labels))
    # Visualize the data by using visualize_results()
    visualize_results(train_inputs[:10], model.call(train_inputs[:10]), train_labels[:10])
if __name__ == '__main__':
    main()