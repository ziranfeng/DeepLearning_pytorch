#!usr/bin/env python
# -*- coding: utf-8 -*-

###############################################
# Make a Small Neural Net Classification Model
###############################################


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # Set seed to make results reproducible

# Generating some dummy data;
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
# x, y = Variable(x), Variable(y)

# Show plot of dummy data
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
plt.show()


# Construct the First Neural Net
class Net(torch.nn.Module):  # Inherit torch's Module

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # Inherit __init__

        # Define the structure of each layer (here only 1 layer)
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer linear output
        self.predict = torch.nn.Linear(n_hidden, n_output)  # predict layer linear output

    def forward(self, x):  # forward is also a function inherited from Module
        # Forward the input and get NN output
        x = F.relu(self.hidden(x))  # Activation Function
        x = self.predict(x)  # Output
        return x


# Make a sample net
net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)  # Print the structure of above net
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

# Now create a training iteration loop
# Initiate a plot engine

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()   # something about plotting

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # back propagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 1 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

