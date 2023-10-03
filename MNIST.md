# Data Preparation and Visualization
##### We began by importing the necessary libraries and modules, including NumPy for numerical computations, Matplotlib for plotting, and Torch for deep learning. We also installed and imported the wandb library for experiment tracking.

```
%%capture
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
!pip install wandb
import wandb as wb
from skimage.io import imread
```

##### We defined two functions, GPU and GPU_data, to convert the data into Torch tensors and transfer them to the GPU, with and without enabling gradient computation, respectively. 
```
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))
def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))
```
##### Another function plot was created to plot data in grayscale, and if the data is a Torch tensor, it converts it to a numpy array first.
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
```
#MNIST
train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

#KMNIST
#train_set = datasets.KMNIST('./data', train=True, download=True)
#test_set = datasets.KMNIST('./data', train=False, download=True)

#Fashion MNIST
# train_set = datasets.FashionMNIST('./data', train=True, download=True)
# test_set = datasets.FashionMNIST('./data', train=False, download=True)
```
```
X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:,None,:,:]/255    
X_test = X_test[:,None,:,:]/255 
```
```
x = X[5,0,:,:]
```
```
plt.imshow(x)
```
```
plot(x)
```
```
plot(X[31,0,:,:])
```
```
montage_plot(X[127:176,0,:,:])
```

















