#!usr/bin/env python
# -*- coding: utf-8 -*-

##############################################
# Make a Small Neural Net Regression Model
##############################################


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # Set seed to make results reproducible


# Generating some dummy data; Note: unsqueeze() makes 2 dims data to 1 dim
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1);
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
# The code below is deprecated in Pytorch 0.4. Now, autograd directly supports tensors
x, y = Variable(x), Variable(y)

# Show plot of dummy data
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


# Construct the First Neural Net
class Net(torch.nn.Module):                                 # Inherit torch's Module

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()                         # Inherit __init__

        # Define the structure of each layer (here only 1 layer)
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer linear output
        self.predict = torch.nn.Linear(n_hidden, n_output)  # predict layer linear output

    def forward(self, x):                                   # forward is also a function inherited from Module
        # Forward the input and get NN output
        x = F.relu(self.hidden(x))                   # Activation Function
        x = self.predict(x)                                 # Output
        return x


# Make a sample net
net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)  # Print the structure of above net
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""


# Now create a training iteration loop
# Initiate a plot engine
plt.ion()
plt.show()

# Define an optimiser
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# Define the loss function that will be used in the training
loss_func = torch.nn.MSELoss()

# Create a 100 steps training iteration
for t in range(100):
    # Get the initial prediction of the input
    prediction = net(x)
    # Calculate the cost
    loss = loss_func(prediction, y)

    optimizer.zero_grad()   # Empty the remaining gradients from the last step
    loss.backward()         # backwards the loss and update parameters
    optimizer.step()        # update the parameters to net's

    # Plot every 5 steps
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    



