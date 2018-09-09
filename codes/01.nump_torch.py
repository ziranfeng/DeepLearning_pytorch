#!usr/bin/env python
# -*- coding: utf-8 -*-

####################################
# A small example of basic torch
####################################

import torch
from torch.autograd import Variable

# Construct a tensor and a tensor variable
tensor = torch.FloatTensor([[1, 2],[3, 4]])
variable = Variable(tensor, requires_grad=True)

print(tensor)
print(variable)

# Some calculations
t_out = torch.mean(tensor*tensor)        # x^2
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)


# Try backward
v_out.backward()
# Backend Math
# v_out = 1/4*sum(variable*variable)
# d(v_out)/d(variable) = 1/4*2*variable = variable/2

# Variable gradient
print(variable.grad)
# Variable data in tensor format
print(variable.data)
# Variable data converted to numpy format
print(variable.data.numpy())















