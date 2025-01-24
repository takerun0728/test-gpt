import numpy as np
from model import Model

EPS = 10e-7

class Optimizer:
    def __init__(self, model: Model):
        self.model = model

    def update(self):
        self.model.zero_grads()
        self.model.backward()

class SGD(Optimizer):
    def __init__(self, model, lr=0.01):
        super().__init__(model)
        self.lr = lr

    def update(self):
        super().update()
        for param, grad in zip(self.model.params, self.model.grads):
            param -= self.lr * grad

class Adam(Optimizer):
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999):
        super().__init__(model)
        self.lr = lr
        self.beta1=beta1
        self.beta2=beta2

        self.ms = [np.zeros_like(param) for param in self.model.params]
        self.vs = [np.zeros_like(param) for param in self.model.params]
        self.iter = 1

    def update(self):
        super().update()
        lr = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for param, grad, m, v in zip(self.model.params, self.model.grads, self.ms, self.vs):
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad**2 - v)
            param -= lr * m / (np.sqrt(v) + EPS)
        
        self.iter += 1