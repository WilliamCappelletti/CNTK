import torch

# from .util import one_hot
# import copy

# def tangentKernel(network, input):
#     ker = torch.Tensor()
#     pass

# def newPointKernel(x, network, input):
#     pass

class NeuralTangentKernel:
    def __init__(self, X, net, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        net = net.to(self.device)
        X = X.to(self.device)

        net.zero_grad()
        out = net(X)

        n_samples, dim_out = out.shape
        n_params = sum(par.numel() for par in net.parameters() if par.requires_grad)

    #     kernel = torch.zeros([n_samples * dim_out] * 2, dtype=torch.float).to(device)
        
        with torch.no_grad():
            # compute the jacobians
            self.Jac = torch.zeros(n_samples, dim_out, n_params)
            for x in range(n_samples):
                for i in range(dim_out):
                    direction = torch.zeros_like(out, dtype=torch.float)
                    direction[x, i] = 1.
                    # Note, if we manage to flatten in one tensor each dic entry, we can use it a tensor as well.
                    JacList = torch.autograd.grad(out, net.parameters(), direction, retain_graph=True)
                    
                    self.Jac[x, i] = torch.cat(tuple(map(lambda par: par.reshape(-1) , JacList)))
                    
                    
            # compute the kernel value
            self.kernel = torch.einsum('abp, cdp -> abcd', self.Jac, self.Jac).reshape(n_samples*dim_out, n_samples*dim_out)
    
    def regression_exact(self, input_train, target_train):
        self.coefficients = torch.cholesky_solve(target_train, self.kernel.unsqueeze(0).to(self.device))
        return self

    def prediction_exact(self, input_test):
        pass
