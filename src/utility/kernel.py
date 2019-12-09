import torch
import time

class NeuralTangentKernel:
    def __init__(self, X, net, device=None, verbose=False):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.net = net

        self.n_samples = X.shape[0]
        self.dim_out = list(net.modules())[-1].out_features
        self.n_params = sum(par.numel() for par in net.parameters() if par.requires_grad)

        start_time = time.time()

        self.Jac = self.compute_jacobians(X)

        if verbose: print("Computed jacobians in {}s".format(time.time() - start_time))
                    
    
    def kernel(self):
        Jac = self.Jac.to(self.device)
        # compute the kernel value
        return torch.einsum('abp, cdp -> abcd', Jac, Jac).reshape(self.n_samples*self.dim_out, self.n_samples*self.dim_out)

    def compute_jacobians(self, X):
        """Returns the jacobians of the neural network for each parameter and each input.
        
        Parameters
        ----------
        X : tensor
            Inputs for the network
        
        Returns
        -------
        tensor
            The jacobian encoded as a tensor of size (n_samples, dim_out, n_param)
        """
        X = X.to(self.device)
        net = self.net.to(self.device)

        net.zero_grad()
        out = net(X)

        Jac = torch.zeros(self.n_samples, self.dim_out, self.n_params)#, device=self.device)

        with torch.no_grad():
            for x in range(self.n_samples):
                for i in range(self.dim_out):
                    direction = torch.zeros_like(out, dtype=torch.float)
                    direction[x, i] = 1.
                    # Note, if we manage to flatten in one tensor each dic entry, we can use it a tensor as well.
                    JacList = torch.autograd.grad(out, self.net.parameters(), direction, retain_graph=True)
                    
                    Jac[x, i] = torch.cat(tuple(map(lambda par: par.reshape(-1) , JacList)))
        return Jac
        

    def regression_exact(self, input_train, target_train):
        """Compute the kernel regression coefficients by inverting the kernel.
        
        Parameters
        ----------
        input_train : tensor
        target_train : tensor
        
        Returns
        -------
        self
        """
        self.coefficients = torch.cholesky_solve(target_train.to(self.device), self.kernel().unsqueeze(0).to(self.device))
        return self

    def predict(self, input_test):
        """Compute the predicted output using the previously computed coefficients.
        
        Parameters
        ----------
        input_test : tensor
        """
        n_test = input_test.shape[0]
        Jac_te = self.compute_jacobians(input_test)
        Jac_tr = self.Jac

        kernel_te = torch.einsum('abp, cdp -> abcd', Jac_te, Jac_tr).reshape(n_test * self.dim_out, self.n_samples * self.dim_out)
        
        output = kernel_te @ self.coefficients.to('cpu')
        return output.reshape(-1, self.dim_out)
