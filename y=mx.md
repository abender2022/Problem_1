# Reshaping and Transferring Data to GPU
#### The training and test data are reshaped to have each image's pixels in a flat array. This flattening is necessary to facilitate matrix multiplication when making predictions.
```
X = X.reshape(X.shape[0],784)
X_test = X_test.reshape(X_test.shape[0],784)
```
#### The reshaped data and labels are then moved to the GPU for faster computations.
```
X = GPU_data(X)
Y = GPU_data(Y)
X_test = GPU_data(X_test)
Y_test = GPU_data(Y_test)
```
```
X = X.T
```
# Making Predictions
#### A batch of 64 examples is selected from the training data.
```
x = X[:,0:64]
```
```
Y[0:64]
```
```
M = GPU(np.random.rand(10,784))
```
```
y = M@x
```
```
torch.max(y,0)
```
```
y = torch.argmax(y,0)
```
```
y
```
```
Y[0:64]
```
```
y == Y[0:64]
```
```
torch.sum((y == Y[0:64]))
```
```
torch.sum((y == Y[0:64]))/64
```
```
batch_size = 64
```
#### A weight matrix M is initialized with random values and used to make predictions on the batch of data. The predicted class labels are determined by identifying the index of the maximum value in each column of the output matrix.
```
M = GPU(np.random.rand(10,784))

y = M@x

y = torch.argmax(y,0)
```
```
torch.sum((y == Y[0:batch_size]))/batch_size
```
#### A loop runs for 100,000 iterations to optimize the weight matrix. In each iteration, a new random weight matrix is created, predictions are made, and the accuracy is calculated. If the current score is better than the best recorded so far, the best score and the corresponding weight matrix are updated.
```
M_Best = 0
Score_Best = 0

for i in range(100000):

    M_new = GPU(np.random.rand(10,784))

    y = M_new@x

    y = torch.argmax(y,0)


    Score = (torch.sum((y == Y[0:batch_size]))/batch_size).item()
    if Score > Score_Best:

        Score_Best = Score
        M_Best = M_new

        print(i,Score_Best)
```











