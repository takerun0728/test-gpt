import numpy as np

EPS = 10e-7

class Layer:
    def __init__(self):
        self.params, self.grads = [], []
        pass

    def zero_grads(self):
        for grad in self.grads:
            grad[...] = 0

class Affine(Layer):
    def __init__(self, W, b):
        super().__init__()
        self.params += [W, b]
        self.grads += [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = x @ W + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = dout @ W.T
        self.grads[0][...] += self.x.T @ dout
        self.grads[1][...] += np.sum(dout, axis=0)
        return dx

class MatMul(Layer):
    def __init__(self, W):
        super().__init__()
        self.params += [W]
        self.grads += [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W = self.params
        out = x @ W
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = dout @ W.T
        self.grads[0][...] += self.x.T @ dout
        return dx
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        return dout * (1 - self.out) * self.out
    
class SoftmaxWithLoss(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, t): #t mast be an index
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.sm_out = e / e.sum(axis=1, keepdims=True)

        self.t = t.flatten()
        return -np.log(self.sm_out[np.arange(x.shape[0]), self.t] + EPS)

    def backward(self, dout):
        g = self.sm_out * dout
        g[np.arange(g.shape[0]), self.t] -= 1
        return g

        