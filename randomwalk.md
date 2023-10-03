# Random Walk 
## Importing Libraries and Preparing Data
#### We start by importing the necessary libraries. torch and numpy are used for numerical operations, torchvision.datasets for loading datasets, and matplotlib.pyplot for plotting images.
```
import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
```
#### A function plot is defined to visualize the images. It can handle both tensors and numpy arrays and also allows an optional title for the plot.
```
def plot(x,title=None):
    if type(x) == torch.Tensor :
        x = x.cpu().detach().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap = 'gray')
    ax.axis('off')
    fig.set_size_inches(7, 7)
    plt.title(title)  
    plt.show()
```
#### The KMNIST dataset is loaded, and the data is extracted and normalized to have values between 0 and 1.
```
##MNIST
# train_set = datasets.MNIST('./data', train=True, download=True)
# test_set = datasets.MNIST('./data', train=False, download=True)

#KMNIST
train_set = datasets.KMNIST('./data', train=True, download=True)
test_set = datasets.KMNIST('./data', train=False, download=True)

# Fashion MNIST
#train_set = datasets.FashionMNIST('./data', train=True, download=True)
#test_set = datasets.FashionMNIST('./data', train=False, download=True)
```
#### The images are reshaped to have each image's pixels in a flat array to facilitate the matrix multiplication during model training.
```
X = train_set.data.numpy()
X_test = test_set.data.numpy()
Y = train_set.targets.numpy()
Y_test = test_set.targets.numpy()

X = X[:,None,:,:]/255
X_test = X_test[:,None,:,:]/255
```
### Generating Random Weights and Making Predictions
m = np.random.standard_normal((10,784))

X = np.reshape(X, (X.shape[0],X.shape[2]*X.shape[3]))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[2]*X_test.shape[3]))

x = X[0:2,:]

x = x.T

y = m@x

np.max(y, axis=0)

y = np.argmax(y, axis=0)

y_ans = Y[0:2]

np.sum((y == y_ans))/len(y)
```
























