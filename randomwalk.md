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
## Generating Random Weights and Making Predictions
#### A random matrix 'm' with a standard normal distribution is generated to act as the initial weights. A subset of data is selected, matrix multiplication is performed with 'm', and class labels are predicted. The accuracy on this subset is calculated.

```
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
```
X = X.T
```
```
X_test = X_test.T
```
```
y = m@X
```
```
def GPU(data):
    return torch.tensor(data, requires_grad=True, dtype=torch.float, device=torch.device('cuda'))
def GPU_data(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float, device=torch.device('cuda'))
```
#### Data is transferred to the GPU for faster computation. Functions GPU and GPU_data are defined to facilitate this, allowing gradient computations to be enabled or disabled as needed.
```
X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)
```
#### An optimization loop is executed 100,000 times to find the optimal 'm' matrix using random search. The best accuracy and corresponding matrix 'm' are updated whenever a better accuracy is found.
##### Got up to 68%
```
m_best = 0
acc_best = 0

for i in range(100000):

    step = 0.0000000001  

    m_random = GPU_data(np.random.randn(10,784))   

    m = m_best  + step*m_random

    y = m@X

    y = torch.argmax(y, axis=0)

    acc = ((y == Y)).sum()/len(Y)

    if acc > acc_best:
        print(acc.item()) 
        m_best = m
        acc_best = acc
```
```
m_random = GPU_data(np.random.randn(10,784))
```
#### A 3D tensor M is created on the GPU with random weights. Each set of weights in M is used to make predictions, and the scores are calculated. The scores are then sorted to identify the best set of weights.
```
M = GPU_data(np.random.random((100,10,784)))

M.shape,X.shape

(M@X).shape

(torch.argmax((M@X), axis=1) == Y).shape

Y.shape

y = torch.argmax((M@X), axis=1)

score = ((y == Y).sum(1)/len(Y))

s = torch.argsort(score,descending=True)

score[s]
```
#### A second optimization loop is implemented, running for 1,000,000 iterations. A form of genetic algorithm is used to optimize a population of 'm' matrices. The best matrix and accuracy are updated, and the population evolves over time.
##### Got up to 88%
```
N = 100  
M = GPU_data(np.random.rand(N,10,784)) 

m_best = 0
acc_best = 0

step = 0.00000000001   

for i in range(1000000):

    y = torch.argmax((M@X), axis=1)
    score = ((y == Y).sum(1)/len(Y))
    s = torch.argsort(score,descending=True)

    M = M[s]

    M[50:100] = 0  
    M[0:50] = M[0] 
    M[1:] += step*GPU_data(np.random.rand(N-1,10,784))  


    acc = score[s][0].item()

    if acc > acc_best:

        m_best = M[0]
        acc_best = acc

        print(i,acc)
```
## Conclusion
#### The code demonstrates various stages of preparing, processing, and evaluating data using random matrices, moving data to the GPU, and attempting to optimize the weights using random search and a form of a genetic algorithm. 




















