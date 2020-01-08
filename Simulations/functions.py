"""
Sample code automatically generated on 2019-11-27 10:35:13

by www.matrixcalculus.org

from input

d/dW W*x = x'\otimes eye

where

W is a matrix
x is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(W, x):
    assert isinstance(W, np.ndarray)
    dim = W.shape
    assert len(dim) == 2
    W_rows = dim[0]
    W_cols = dim[1]
    assert isinstance(x, np.ndarray)
    dim = x.shape
    assert len(dim) == 1
    x_rows = dim[0]
    assert x_rows == W_cols

    functionValue = (W).dot(x)
    gradient = np.einsum('ij, k', np.eye(W_rows, W_rows), x)

    return functionValue, gradient

def checkGradient(W, x):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3, 3)
    f1, _ = fAndG(W + t * delta, x)
    f2, _ = fAndG(W - t * delta, x)
    f, g = fAndG(W, x)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    W = np.random.randn(3, 3)
    x = np.random.randn(3)

    return W, x

if __name__ == '__main__':
    W, x = generateRandomData()
    print('matrix W :\n{}'.format(W))
    print('vector x :\n{}'.format(x))
    functionValue, gradient = fAndG(W, x)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(W, x)
