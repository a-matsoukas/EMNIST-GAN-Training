"""
This module contains helper functions for making the GAN trainer functional.

Specifically, this module contains functions to save the parameters of a GAN
trainer to a .npz file, recover the data from a .npz file, generate training and
testing data for the discriminator, generate 100-dimensional random samples from
a normal distribution to pass through the generator, and to create visuals of
the data.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

def save_network_data(filename, trainer):
    """
    Save the parameters of a GAN trainer to a .npz file.

    Args:
        filename: a string containing the name of the file to which to save data
        trainer: the GAN trainer whose parameters will be saved
    Returns:
        No Return Value
    """
    # Save all of the weight matrices and the arrays containing the value
    # functions for each batch
    np.savez(filename, Theta1 = trainer.Theta1, Theta2 = trainer.Theta2, 
        Theta3 = trainer.Theta3, Theta4 = trainer.Theta4, Phi1 = trainer.Phi1, 
        Phi2 = trainer.Phi2, Phi3 = trainer.Phi3, Phi4 = trainer.Phi4, 
        Phi5 = trainer.Phi5, disc_val = trainer.disc_value_array, 
        gen_val = trainer.gen_value_array)

def recover_network_data(filename, trainer):
    """
    Recover the parameters of a GAN trainer by overwriting the parameters of a
    currently initialized trainer.

    Args:
        filename: a string containing the name of the file that will be loaded
            in
        trainer: the trainer that will be overwritten by the data in the file
    Returns:
        No Return Value
    """
    # Read the data in the file and overwrite the weight matrices and the value
    # function info of the given GAN trainer
    saved_data = np.load(filename)
    trainer._Theta1 = saved_data['Theta1']
    trainer._Theta2 = saved_data['Theta2']
    trainer._Theta3 = saved_data['Theta3']
    trainer._Theta4 = saved_data['Theta4']
    trainer._Phi1 = saved_data['Phi1']
    trainer._Phi2 = saved_data['Phi2']
    trainer._Phi3 = saved_data['Phi3']
    trainer._Phi4 = saved_data['Phi4']
    trainer._Phi5 = saved_data['Phi5']
    trainer._disc_value_array = saved_data['disc_val']
    trainer._gen_value_array = saved_data['gen_val']

def load_disc_train_data(digit):
    """
    Load EMNIST training data for a specific digit 0 through 9, which will be
    used to train the discriminator.

    Args:
        digit: an int representing the digit that the GAN will be trained on
    Returns:
        train_data_digit: a numpy array containing all of the training examples
    """
    # Load the Training Data
    train_set = torchvision.datasets.EMNIST(
        root='./emnistdata', split='digits', train=True, download=False, 
        transform=transforms.ToTensor())
    
    # Transpose all of the images to make them right-side-up
    train_data_unvec = train_set.data.transpose(1, 2).numpy()

    # Vectorize the images to be rows of a numpy array and save their labels
    train_data_vec = np.reshape(train_data_unvec, (-1, 28*28))
    train_labels = train_set.targets.numpy()

    # Isolate only the chosen digit
    train_data_digit = train_data_vec[train_labels == digit]

    return train_data_digit

def load_disc_test_data(digit):
    """
    Load  EMNIST testing data for a specific digit 0 through 9, which will be
    used to test the discriminator.

    Args:
        digit: an int representing the digit that the GAN will be tested with
    Returns:
        test_data_digit: a numpy array containing all of the testing examples
    """
    # Load the Testing Data
    test_set = torchvision.datasets.EMNIST(
        root='./emnistdata', split='digits', train=False, download=False,
        transform=transforms.ToTensor())
    
    # Transpose all of the images to make them right-side-up
    test_data_unvec = test_set.data.transpose(1, 2).numpy()

    # Vectorize the images to be rows of a numpy array and save their labels
    test_data_vec = np.reshape(test_data_unvec, (-1, 28*28))
    test_labels = test_set.targets.numpy()

    # Isolate only the chosen digit
    test_data_digit = test_data_vec[test_labels == digit]

    return test_data_digit

def generator_input(mu=0, sigma=1):
    """
    Generate random noise to pass through the generator from a normal
    distribution with default mean of 0 and default st. dev. of 1.

    Args:
        mu: a float representing the mean of the normal distribution to sample
            from
        sigma: a float representing the st. dev. of the normal distribution to
            sample from
    Returns:
        a 100 x 1 numpy array sampled from the defined distribution
    """
    return np.random.normal(mu, sigma, size=(100,1))

def plot_image(data_vector):
    """
    Plot the image given in vevtorized form.

    Args:
        data_vector: a 784 x 1 numpy array representing a grayscale image
    Returns:
        No Return Value
    """
    plt.axis('off')

    # display the image in grayscale after reshaping it back to 28 x 28
    plt.imshow(np.reshape(data_vector, (28,28)), cmap='gray')

def plot_subplot_images(data_matrix, height, width, fig_size=(10,5)):
    """
    Sample random images from a numpy array of vectorized images and plot them
    in a figure with the given number of subplots.

    Args:
        data_matrix: a numpy array containing vectorized images to plot
        height: an int representing the number of vertical subplots
        width: an int representing the number of horizontal subplots
        fig_size: a tuple representing the overall size of the figure - set to
            (10,5) by default'
    Returns:
        No Return Value
    """
    # Find the max number of images that can be plotted and the number of images
    # needed
    max_images = data_matrix.shape[0]
    num_images = height * width

    # sample the number needed from the max number possible
    image_indeces = random.sample(range(max_images), num_images)

    # define the figure with the correct number of subplots
    plt.subplots(height, width, figsize=fig_size)
    
    # iterate through each subplot and the image indeces and plot a new image in
    # each subplot
    for image_num in range(height * width):
        plt.subplot(height, width, image_num + 1)
        plot_image(data_matrix[image_indeces[image_num]][np.newaxis].T)
    plt.show()

def plot_value_functions(trainer):
    """
    
    """
    f, (ax1, ax2) = plt.subplots(1, 2)
    
    disc_y = trainer.disc_value_array
    disc_x = list(range(1, disc_y.shape[0] + 1))

    gen_y = trainer.gen_value_array
    gen_x = list(range(1, gen_y.shape[0] + 1))
    
    ax1.plot(disc_x, disc_y)
    ax1.set_title('Discriminator Value')
    ax1.set_xlabel('Batch Number')
    ax1.set_ylabel('Value Function')
    
    ax2.plot(gen_x, gen_y)
    ax2.set_title('Generator Value')
    ax2.set_xlabel('Batch Number')
