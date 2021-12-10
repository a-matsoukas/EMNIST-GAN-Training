"""

"""
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

def save_network_data(filename, trainer):
    """
    
    """
    np.savez(filename, Theta1 = trainer.Theta1, Theta2 = trainer.Theta2, 
        Theta3 = trainer.Theta3, Theta4 = trainer.Theta4, Phi1 = trainer.Phi1, 
        Phi2 = trainer.Phi2, Phi3 = trainer.Phi3, Phi4 = trainer.Phi4, 
        disc_val = trainer.disc_value_array, gen_val = trainer.gen_value_array)

def recover_network_data(filename, trainer):
    """
    
    """
    saved_data = np.load(filename)
    trainer._Theta1 = saved_data['Theta1']
    trainer._Theta2 = saved_data['Theta2']
    trainer._Theta3 = saved_data['Theta3']
    trainer._Theta4 = saved_data['Theta4']
    trainer._Phi1 = saved_data['Phi1']
    trainer._Phi2 = saved_data['Phi2']
    trainer._Phi3 = saved_data['Phi3']
    trainer._Phi4 = saved_data['Phi4']
    trainer._disc_value_array = saved_data['disc_val']
    trainer._gen_value_array = saved_data['gen_val']

def load_disc_train_data(digit):
    """
    
    """
    # Load the Training Data
    train_set = torchvision.datasets.EMNIST(
        root='./emnistdata', split='digits', train=True, download=False, 
        transform=transforms.ToTensor())
    
    train_data_unvec = train_set.data.transpose(1, 2).numpy()
    train_data_vec = np.reshape(train_data_unvec, (-1, 28*28))
    train_labels = train_set.targets.numpy()

    # Isolate only the chosen digit
    train_data_digit = train_data_vec[train_labels == digit]

    return train_data_digit

def load_disc_test_data(digit):
    """
    
    """
    # Load the Testing Data
    test_set = torchvision.datasets.EMNIST(
        root='./emnistdata', split='digits', train=False, download=False,
        transform=transforms.ToTensor())
    
    test_data_unvec = test_set.data.transpose(1, 2).numpy()
    test_data_vec = np.reshape(test_data_unvec, (-1, 28*28))
    test_labels = test_set.targets.numpy()

    # Isolate only the chosen digit
    test_data_digit = test_data_vec[test_labels == digit]

    return test_data_digit

def generator_input(mu=0, sigma=3):
    """
    
    """
    return np.random.normal(mu, sigma, size=(49,1))

def plot_image(data_vector):
    """
    
    """
    plt.imshow(np.reshape(data_vector, (28,28)), cmap='gray')

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
