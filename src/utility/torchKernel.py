import torch
import time
from tqdm.auto import tqdm

class NeuralTangentKernel(torch.nn.Module):
    def __init__(self, X, net, device=None, verbose=False):
        super(NeuralTangentKernel, self).__init__()

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # NOTE: this is a list as torch does not look inside to see the parameters
        self.nets = [net]

        self.n_samples = X.shape[0]

        with torch.no_grad():
            self.dim_out = net(X).shape[1]
        
        self.n_params = sum(par.numel() for par in net.parameters() if par.requires_grad)

        start_time = time.time()

        self.Jac = self.compute_jacobians(X, prog_bar=True)
        if verbose: print("Computed jacobians in {}s".format(time.time() - start_time))

        self.kernel = self.train_kernel()
        
        # TODO: choose an std for initialization.
        self.coefficients = torch.nn.Parameter(torch.empty(self.dim_out * self.n_samples).normal_(std=1.))
                    
    def train_kernel(self):
        Jac = self.Jac.to(self.device)
        # compute the kernel value
        return torch.einsum('abp, cdp -> abcd', Jac, Jac).reshape(self.n_samples*self.dim_out, self.n_samples*self.dim_out)

    def compute_jacobians(self, X, prog_bar=False):
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
        net = self.nets[0].to(self.device)

        net.zero_grad()
        out = net(X)

        Jac = torch.zeros(self.n_samples, self.dim_out, self.n_params)#, device=self.device)

        with torch.no_grad():
            for x in tqdm(range(self.n_samples), desc="Computing Jacobians") if prog_bar else range(self.n_samples):
                for i in range(self.dim_out):
                    direction = torch.zeros_like(out, dtype=torch.float)
                    direction[x, i] = 1.
                    JacList = torch.autograd.grad(out, net.parameters(), direction, retain_graph=True)
                    
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
        coeffs = torch.cholesky_solve(target_train.to(self.device), self.kernel().unsqueeze(0).to(self.device))
        self.coefficients = coeffs
        return self

    def predict(self, input_test):
        """Compute the predicted output using the previously computed coefficients.
        
        Parameters
        ----------
        input_test : tensor
        """
        n_test = input_test.shape[0]
        if self.training:
            kernel_te = self.kernel
        else:
            Jac_te = self.compute_jacobians(input_test)
            Jac_tr = self.Jac
            kernel_te = torch.einsum('abp, cdp -> abcd', Jac_te, Jac_tr).reshape(n_test * self.dim_out, self.n_samples * self.dim_out)
            kernel_te = kernel_te.to(self.device)

        output = kernel_te @ self.coefficients.to(self.device)
        return output.reshape(-1, self.dim_out)

    def forward(self, X):
        return self.predict(X)