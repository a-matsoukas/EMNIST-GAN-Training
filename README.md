# EMNIST GAN Training 
### Project Goal
The goal of this project was to gain some intuition for backpropogation and to explore the functionality of GANs. All of the implementation and training of the GAN was done from scratch.


### GAN Architecture
The generator is a multilayer perceptron with six layers. Including the bias term, the input layer has 101 nodes, the first hidden layer has 250 nodes, the second hidden layer has 350 nodes, the third and fourth hidden layer have 500 nodes each, and the output layer has 784 nodes. The discriminator is a multilayer perceptron with five layers. Including the bias term, the input layer has 785 nodes, the first hidden layer has 500 nodes, the second hidden layer has 250 nodes, the third hidden layer has 50 nodes, and the output layer has one node.


### Implementation Details
Each layer of the GAN uses a leaky ReLU activation function, with an alpha value of .1. The generator samples from a normal distribution with a mean of 0 and a standard deviation of 1. Both the generator and the discriminator are trained via backpropogation, using Wasserstein Loss.


### Training Details
The generator is trained to replicate the digit 0 from the EMNIST digits dataset. Some of the training examples are shown below. The digit that the generator is trained on can be changed in the training notebook, but it can only be trained on one digit at a time. 

![image](/readme_images/train_data.png)


### Results
The following images are generated images after training the GAN for 1 epoch, 60 epochs, and 120 epochs.


1 Epoch:

![1_epoch](/readme_images/1epoch.png)


60 Epochs:

![60_epoch](/readme_images/60epoch.png)


120 Epochs:

![120_epochs](/readme_images/120epoch.png)


### Necessary Libraries
To run this code, the following Python libraries are required:
- numpy
- random
- matplotlib.pyplot
- torchvision
- torchvision.transforms

**NOTE: Due to GitHub file size constraints, I am unable to push the folder with the EMNIST data to GitHub. When running this code for the first time, you must navigate to ``helper_functions.py`` and edit the functions called ``load_disc_train_data`` and ``load_disc_test_data`` to download the data.**

**``load_disc_train_data``:**
In the following line, set the download parameter to true. ``train_set = torchvision.datasets.EMNIST(root='./emnistdata', split='digits', train=True, download=False, transform=transforms.ToTensor())``

**``load_disc_test_data``:**
In the following line, set the download parameter to true. ``test_set = torchvision.datasets.EMNIST(root='./emnistdata', split='digits', train=False, download=False,transform=transforms.ToTensor())``

After the data is successfully downloaded by running the main training notebook (you should see a folder called emnistdata appear in your directory), you can go back and set these parameters to false to avoid downloading the data every time you run the notebook.
