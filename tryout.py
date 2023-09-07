import torch
import numpy as np

#Empty tensor
x=torch.empty(2)
print(x)

# multidimensional tensor
x=torch.empty(2, 3)
print(x)

# 0 tensor
x=torch.zeros(2, 3)
print(x)

#random tensor
x=torch.rand(2,3)
print(x)

x = torch.rand(5, 3)
print(x)

print(x[0,:])
print(x[:,0])
print(x[0,0])
print(x[0,0].item())

# reshape tensor

y=x.view(15)
print(y)

y=x.view(-1,5)
print(y.size())

### Tensor <-> Numpy

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

### use GPU

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)

z = z.to("cpu")
z = z.numpy()