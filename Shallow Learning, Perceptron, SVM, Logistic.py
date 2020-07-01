# -*- coding: utf-8 -*-
"""

Shallow learning: Perceptron algorithm, SVM, Logstic regression

@author: Kyungho Lee at SNU Econ

Note: Some starter codes are provided by the lecturer.

"""

"""

Perceptron algorithm with a Kernel method.

"""

import numpy as np
import matplotlib.pyplot as plt

# Generate random data
N =30
np.random.seed(0)
X = np.random.randn(2, N)
y = np.sign(X[0,:]**2 + X[1,:]**2 - 0.7)
theta = 0.5
c , s = np.cos(theta), np.sin(theta)
X = np.array([[ c , -s], [ s , c]])@X
X = X + np.array([[1] ,[1]])

# Observe that the data are not linearly separable
plt.cla()
plt.plot(X[0,y==1],X[1,y==1],color='blue',marker='+',linestyle='None')
plt.plot(X[0,y==-1],X[1,y==-1],color='red',marker='x',linestyle='None')
plt.show()

def Transformation(X_i):
    return np.asarray([1, X_i[0], X_i[0]**2, X_i[1], X_i[1]**2])


w = np.zeros(5)
for _ in range(1000):
    for i in range(N):
        if np.sign(Transformation(X[:,i]).T@w) != y[i]:
            x_curr, y_curr = X[:,i], y[i]
            break
    else : 
        print("No misclassificaion. Converged")
        break
    w += y_curr*Transformation(x_curr)

    #Plot results
    plt.cla()
    plt.plot(X[0,y==1],X[1,y==1],color='blue',marker='+',linestyle='None')
    plt.plot(X[0,y==-1],X[1,y==-1],color='red',marker='x',linestyle='None')
    xx = np.linspace ( -4 ,4 ,1024)
    dd = ( w[3]**2 -4 * w[4]*(w[0]+ xx * (w[1]+ w[2]* xx )))
    yy = ( - w[3] + np.sqrt(dd [ dd >=0]))/(2* w[4])
    plt.plot(xx[dd >=0], yy, color = 'green')
    yy = ( - w [3] - np . sqrt ( dd [ dd >=0]))/(2* w [4])
    plt.plot( xx[dd >=0] , yy , color = 'green')


"""

Binary Image Classification:
    
MNIST dataset

Support Vector Machine using PyTorch.

"""

import torch
import torch.nn as nn

# torchvision: popular datasets, model architectures, and common image transformations for computer vision.
from torchvision import datasets
from torchvision.transforms import transforms

from random import randint
import numpy as np
import matplotlib.pyplot as plt


'''
Step 1: Prepare dataset
'''

# Use data with only 4 and 9 as labels: which is hardest to classify
label_1, label_2 = 4, 9

# MNIST training data
train_set = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)

# Use data with two labels
idx = (train_set.targets == label_1) + (train_set.targets == label_2)
train_set.data = train_set.data[idx]
train_set.targets = train_set.targets[idx]
train_set.targets[train_set.targets == label_1] = -1
train_set.targets[train_set.targets == label_2] = 1

# # MNIST testing data
test_set = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor())
# # Use data with two labels
idx = (test_set.targets == label_1) + (test_set.targets == label_2)
test_set.data = test_set.data[idx]
test_set.targets = test_set.targets[idx]
test_set.targets[test_set.targets == label_1] = -1
test_set.targets[test_set.targets == label_2] = 1

'''
Step 2: Define the neural network class.
'''

## This step is the same to Logistic Regression case

class SVM(nn.Module) :
    '''
    Initialize model
        input_dim : dimension of given input data
    '''
    # MNIST data is 28x28 images
    def __init__(self, input_dim=28*28) :
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        # The network parameters are part of the neural network object.
        # therefore they are initialized as class attributes in the __init__ method
        
    ''' forward given input x '''
    def forward(self, x) :
        # The forward network defines how the neurons are combined to form the network.
        # How the connections are formed, and what activation functions are used are defined here.
        # The definition here should not introduce new parameters.
        x = self.linear(x.float().view(-1, 28*28))
        return x
    
    
'''
Step 3: Create the model, specify loss function and optimizer. In here, we use SVM Loss function
'''

model = SVM()                                   # Define a Neural Network Model

def SVM_loss(output, target):
    return torch.mean(torch.nn.functional.relu(1-target*output))

C = 5
loss_function = SVM_loss                                                   # Specify loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)   # specify SGD with learning rate

'''
Step 4: Train model with SGD
'''
for _ in range(1000) :
    # Sample a random data for training
    ind = randint(0, len(train_set.data)-1)
    image, label = train_set.data[ind], train_set.targets[ind]
    
    # Clear previously computed gradient
    optimizer.zero_grad()
    
    # then compute gradient with forward and backward passes
    train_loss = C*loss_function(model(image), label.float())
    weight = model.linear.weight.view(-1)
    train_loss += torch.sum(weight*weight) # weight sum
    train_loss.backward()
    
    #(This syntax will make more sense once we learn about minibatches)

    # perform SGD step (parameter update)
    optimizer.step()

'''
Step 5: Test model (Evaluate the accuracy)
'''
test_loss, correct = 0, 0
misclassified_ind = []
correct_ind = []

# Evaluate accuracy using test data
for ind in range(len(test_set.data)) :
    
    image, label = test_set.data[ind], test_set.targets[ind]
    
    # evaluate model
    output = model(image)
    
    # Calculate cumulative loss
    test_loss += loss_function(output, label.float()).item()

    # Make a prediction
    if output.item() * label.item() >= 0 :
        correct += 1
        correct_ind += [ind]
    else:
        misclassified_ind += [ind]

# Print out the results
print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss /len(test_set.data), correct, len(test_set.data),
        100. * correct / len(test_set.data)))

'''
Step 6: Show some incorrectly classified images and some correctly classified ones
''' 
# Misclassified images
fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Misclassified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[misclassified_ind[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[misclassified_ind[k]]
    
    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_2))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_1))
    plt.imshow(image, cmap='gray')
plt.show()

# Correctly classified images
fig = plt.figure(2, figsize=(15, 6))
fig.suptitle('Correctly-classified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[correct_ind[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[correct_ind[k]]

    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_1))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_2))
    plt.imshow(image, cmap='gray')
plt.show()



"""


Binary Image Classification:
    
MNIST dataset

Logistic Regression with Sum of Square Loss using PyTorch.
 

"""



'''
Step 2: Define the neural network class.
'''

class LR_no_bias(nn.Module) :
    '''
    Initialize model
        input_dim : dimension of given input data
    '''
    # MNIST data is 28x28 images
    def __init__(self, input_dim=28*28) :
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)
        # The network parameters are part of the neural network object.
        # therefore they are initialized as class attributes in the __init__ method
        
    ''' forward given input x '''
    def forward(self, x) :
        # The forward network defines how the neurons are combined to form the network.
        # How the connections are formed, and what activation functions are used are defined here.
        # The definition here should not introduce new parameters.
        x = self.linear(x.float().view(-1, 28*28))
        return x
    
    
'''
Step 3: Create the model, specify loss function and optimizer. In here, we use Sum of Square Loss function
'''
del model
model = LR_no_bias()                                   # Define a Neural Network Model

def sum_of_square_loss(output, target):
    
    if target == -1:
        p_1 = 1
        p_2 = 0
    else:
        p_1 = 0
        p_2 = 1
    
    return torch.square(p_1 - nn.functional.sigmoid(-output)) + torch.square(p_2 - nn.functional.sigmoid(output))

loss_function = sum_of_square_loss                                                   # Specify loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)   # specify SGD with learning rate

'''
Step 4: Train model with SGD
'''

for _ in range(2000) :
    # Sample a random data for training
    ind = randint(0, len(train_set.data)-1)
    image, label = train_set.data[ind], train_set.targets[ind]
    
    # Clear previously computed gradient
    optimizer.zero_grad()
    
    # then compute gradient with forward and backward passes
    train_loss = loss_function(model(image), label.float())
    train_loss.backward()
    
    #(This syntax will make more sense once we learn about minibatches)

    # perform SGD step (parameter update)
    optimizer.step()


'''
Step 5: Test model (Evaluate the accuracy)
'''


test_loss, correct = 0, 0
misclassified_ind = []
correct_ind = []

# Evaluate accuracy using test data
for ind in range(len(test_set.data)) :
    
    image, label = test_set.data[ind], test_set.targets[ind]
    
    # evaluate model
    output = model(image)
    
    # Calculate cumulative loss
    test_loss += loss_function(output, label.float()).item()

    # Make a prediction
    if output.item() * label.item() >= 0 :
        correct += 1
        correct_ind += [ind]
    else:
        misclassified_ind += [ind]

# Print out the results
print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss /len(test_set.data), correct, len(test_set.data),
        100. * correct / len(test_set.data)))

'''
Step 6: Show some incorrectly classified images and some correctly classified ones
''' 
# Misclassified images
fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Misclassified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[misclassified_ind[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[misclassified_ind[k]]
    
    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_2))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_1))
    plt.imshow(image, cmap='gray')
plt.show()

# Correctly classified images
fig = plt.figure(2, figsize=(15, 6))
fig.suptitle('Correctly-classified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[correct_ind[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[correct_ind[k]]

    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_1))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_2))
    plt.imshow(image, cmap='gray')
plt.show()



"""

Learning Rate Comparison of binary image claasifiction:
the MNIST dataset using logistic regression

"""

class LR(nn.Module) :
    '''
    Initialize model
        input_dim : dimension of given input data
    '''
    # MNIST data is 28x28 images
    def __init__(self, input_dim=28*28) :
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        # The network parameters are part of the neural network object.
        # therefore they are initialized as class attributes in the __init__ method
        
    ''' forward given input x '''
    def forward(self, x) :
        # The forward network defines how the neurons are combined to form the network.
        # How the connections are formed, and what activation functions are used are defined here.
        # The definition here should not introduce new parameters.
        x = self.linear(x.float().view(-1, 28*28))
        return x
    
# Define a Neural Network Model
model_standard = LR()
model_big_lr = LR()                                   
model_small_lr = LR()                                   

def logistic_loss(output, target):
    return -torch.nn.functional.logsigmoid(target*output)

loss_function = logistic_loss                                                   # Specify loss function
optimizer_standard = torch.optim.SGD(model_standard.parameters(), lr=1e-4)    # specify SGD with learning rate
optimizer_big_lr = torch.optim.SGD(model_big_lr.parameters(), lr=10000)        # specify SGD with learning rate
optimizer_small_lr = torch.optim.SGD(model_small_lr.parameters(), lr=1e-13)    # specify SGD with learning rate


'''
Step 4: Train model with SGD
'''
for _ in range(1000) :
    # Sample a random data for training
    ind = randint(0, len(train_set.data)-1)
    image, label = train_set.data[ind], train_set.targets[ind]
    
    # Clear previously computed gradient
    optimizer_standard.zero_grad()
    optimizer_big_lr.zero_grad()
    optimizer_small_lr.zero_grad()
    
    # then compute gradient with forward and backward passes
    train_loss_standard = loss_function(model_standard(image), label.float())
    train_loss_big_lr = loss_function(model_big_lr(image), label.float())
    train_loss_small_lr = loss_function(model_small_lr(image), label.float())

    train_loss_standard.backward()
    train_loss_big_lr.backward()
    train_loss_small_lr.backward()
    
    #(This syntax will make more sense once we learn about minibatches)

    # perform SGD step (parameter update)
    optimizer_standard.step()
    optimizer_big_lr.step()
    optimizer_small_lr.step()

'''
Step 5: Test model (Evaluate the accuracy)
'''
test_loss_standard, correct_standard = 0, 0
test_loss_big_lr, correct_big_lr = 0, 0
test_loss_small_lr, correct_small_lr = 0, 0

misclassified_ind_standard = []
misclassified_ind_big_lr = []
misclassified_ind_small_lr = []

correct_ind_standard = []
correct_ind_big_lr = []
correct_ind_small_lr = []


# Evaluate accuracy using test data
for ind in range(len(test_set.data)) :
    
    image, label = test_set.data[ind], test_set.targets[ind]
    
    # evaluate model
    output_standard = model_standard(image)
    output_big_lr = model_big_lr(image)
    output_small_lr = model_small_lr(image)
    
    # Calculate cumulative loss
    test_loss_standard += loss_function(output_standard, label.float()).item()
    test_loss_big_lr += loss_function(output_big_lr, label.float()).item()
    test_loss_small_lr += loss_function(output_small_lr, label.float()).item()

    # Make a prediction

    if output_standard.item() * label.item() >= 0 :
        correct_standard += 1
        correct_ind_standard += [ind]
    else:
        misclassified_ind_standard += [ind]

    if output_big_lr.item() * label.item() >= 0 :
        correct_big_lr += 1
        correct_ind_big_lr += [ind]
    else:
        misclassified_ind_big_lr += [ind]
        
    
    if output_small_lr.item() * label.item() >= 0 :
        correct_small_lr += 1
        correct_ind_small_lr += [ind]
    else:
        misclassified_ind_small_lr += [ind]

# Print out the results
print('[Test set: Standard lr] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss_standard /len(test_set.data), correct_standard, len(test_set.data),
        100. * correct_standard / len(test_set.data)))
print('[Test set: Big lr] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss_big_lr /len(test_set.data), correct_big_lr, len(test_set.data),
        100. * correct_big_lr / len(test_set.data)))
print('[Test set: Small lr] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss_small_lr /len(test_set.data), correct_small_lr, len(test_set.data),
        100. * correct_small_lr / len(test_set.data)))


"""

If learning rate is too big or too small, the accuracy of the model tends to get lower. 
Especially, when lr is too small, the accuracy of the model is very low.

"""


'''
Step 6: Show some incorrectly classified images and some correctly classified ones
''' 
# Misclassified images
fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Misclassified Figures: Standard lr', fontsize=16)

for k in range(3) :
    image = test_set.data[misclassified_ind_standard[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[misclassified_ind_standard[k]]
    
    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_2))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_1))
    plt.imshow(image, cmap='gray')
plt.show()


fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Misclassified Figures: Big lr', fontsize=16)

for k in range(3) :
    image = test_set.data[misclassified_ind_big_lr[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[misclassified_ind_big_lr[k]]
    
    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_2))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_1))
    plt.imshow(image, cmap='gray')
plt.show()

fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Misclassified Figures: Small lr', fontsize=16)

for k in range(3) :
    image = test_set.data[misclassified_ind_small_lr[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[misclassified_ind_small_lr[k]]
    
    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_2))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_1))
    plt.imshow(image, cmap='gray')
plt.show()

# Correctly classified images
fig = plt.figure(2, figsize=(15, 6))
fig.suptitle('Correctly-classified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[correct_ind_standard[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[correct_ind_standard[k]]

    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_1))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_2))
    plt.imshow(image, cmap='gray')
plt.show()

fig = plt.figure(2, figsize=(15, 6))
fig.suptitle('Correctly-classified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[correct_ind_big_lr[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[correct_ind_big_lr[k]]

    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_1))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_2))
    plt.imshow(image, cmap='gray')
plt.show()

fig = plt.figure(2, figsize=(15, 6))
fig.suptitle('Correctly-classified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[correct_ind_small_lr[k]].cpu().numpy().astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[correct_ind_small_lr[k]]

    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_1, label_1))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(label_2, label_2))
    plt.imshow(image, cmap='gray')
plt.show()