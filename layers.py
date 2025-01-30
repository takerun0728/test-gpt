import numpy as np
from utility import *

EPS = 10e-7

class Layer:
    def __init__(self):
        self.params, self.grads = [], []
        pass

    def zero_grads(self):
        for grad in self.grads:
            grad[...] = 0

class MatMul(Layer):
    def __init__(self, W):
        super().__init__()
        self.params += [W]
        self.grads += [np.zeros_like(W)]
        self.x = []

    def forward(self, x, train=False):
        W = self.params[0]
        out = x @ W
        if train: self.x.append(x)
        return out

    def backward(self, dout):
        W = self.params[0]
        dx = dout @ W.T
        for x in self.x:
            self.grads[0][...] += x.T @ dout
        self.x = []
        return dx
    
    def zero_grads(self):
        super().zero_grads()

class Affine(MatMul):
    def __init__(self, W, b):
        super().__init__(W)
        self.params += [b]
        self.grads += [np.zeros_like(b)]

    def forward(self, x, train=False):
        W, b = self.params
        out = self.call_super_func(W, b, super().forward, x, train) + b
        return out

    def backward(self, dout):
        W, b = self.params
        dx = self.call_super_func(W, b, super().backward, dout)
        self.grads[1][...] += np.sum(dout, axis=0)
        return dx
    
    def call_super_func(self, W, b, super_func, *args):
        self.params = [W]
        out = super_func(*args)
        self.params = [W, b]
        return out
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, train=False):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        return dout * (1 - self.out) * self.out
    
class SoftmaxWithLoss(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, t, train=False): #t mast be an index
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.sm_out = e / e.sum(axis=1, keepdims=True)

        self.t = t.flatten()
        return -np.log(self.sm_out[np.arange(x.shape[0]), self.t] + EPS)

    def backward(self, dout):
        g = self.sm_out
        g[np.arange(g.shape[0]), self.t] -= 1
        return g * dout
    
class SigmoidWithLoss(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, t):
        self.sig_out = 1 / (1 + np.exp(-x))
        self.t = t
        return -t * np.log(self.sig_out) - (1 - t) * np.log(1 - self.sig_out)

    def backward(self, dout):
        return (self.sig_out - self.t) * dout
       
class Embedding(Layer):
    def __init__(self, W):
        super().__init__()
        self.params += [W]
        self.grads += [np.zeros_like(W)]
        self.idx = []
        
    def forward(self, idx):
        W = self.params
        self.idx += idx
        
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW = self.grads
        dW[...] = 0
        dW[self.idx] += dout
        self.idx = []

class EmbeddingDot(Embedding):
    def __init__(self, W):
        super().__init__(W)

    def forward(self, h, idx):
        self.target = super().forward(idx)
        self.h = h
        return np.sum(self.target, self.h, axis=1, keepdims=True)

    def backward(self, dout):
        super().backward(dout * self.h)
        return dout * self.target
