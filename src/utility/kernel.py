import torch

def tangentKernel(network, input):
    ker = torch.Tensor()
    pass

def newPointKernel(x, network, input):
    pass

class NeuralTangentKernel:
    def __init__(self, X, net):
        out = net(X)

        n_samples, dim_out = out.shape

        kernel = torch.zeros([n_samples * dim_out] * 2, dtype=torch.float)
        for x in range(n_samples):
            for x1 in range(x, n_samples):
                for i in range(dim_out):
                    for j in range(i, dim_out):
                        net.zero_grad()
                        out = net(x)
                        direction = torch.zeros_like(out, dtype=torch.float)
                        direction[:, i] = 1.



    pass

