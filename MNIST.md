# MNIST

## Data Preparation and Visualization
#### Began by importing the necessary libraries and modules, including NumPy for numerical computations, Matplotlib for plotting, and Torch for deep learning. We also installed and imported the wandb library for experiment tracking.

```
%%capture
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from skimage.util import montage
!pip install wandb
import wandb as wb
from skimage.io import imread
```
## Loading and Processing the Dataset
#### Defined two functions, GPU and GPU_data, to convert the data into Torch tensors and transfer them to the GPU, with and without enabling gradient computation, respectively. 
```
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))
def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))
```
#### Another function plot was created to plot data in grayscale, and if the data is a Torch tensor, it converts it to a numpy array first.
```
def plot(x):
    if type(x) == torch.Tensor :
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')    # Plot in grayscale 
    ax.axis('off')    # Hide axes ticks and labels
    fig.set_size_inches(7, 7)   # Size of plot 
    plt.show()   #Display the plot
```
```
def montage_plot(x):
    x = np.pad(x, pad_width=((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)   # Pad the image
    plot(montage(x))   # calling plot function to display the montage
```
#### The MNIST dataset is loaded into the train_set and test_set variables. The data is automatically downloaded if it’s not already present in the './data' directory.
```
#MNIST
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

#### The training and testing data and their corresponding labels are extracted and converted to numpy arrays.
```
X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()
```
#### The images are normalized to have pixel values between 0 and 1.
```
X = X[:,None,:,:]/255    
X_test = X_test[:,None,:,:]/255 
```
```
montage_plot(X[127:176,0,:,:])
```
## Conclusion
#### This code is an example of how to load, process, and visualize the MNIST dataset using Python, Numpy, PyTorch, and Matplotlib.

















