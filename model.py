import numpy as np
from layers import Affine, Sigmoid, SoftmaxWithLoss

class Model:
    def __init__(self):
        self.params, self.grads, self.layers = [], [], []

    def post_init(self):
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_loss(self, x, t):
        y = self.forward(x)
        return self.loss_layer.forward(y, t)
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def zero_grads(self):
        for layer in self.layers:
            layer.zero_grads()

class TwoLayerSoftMaxNet(Model):
    def __init__(self, input_size, hidden_size, output_size, sigma=0.01):
        super().__init__()
        W1 = sigma * np.random.randn(input_size, hidden_size)
        b1 = sigma * np.zeros(hidden_size)
        W2 = sigma * np.random.randn(hidden_size, output_size)
        b2 = sigma * np.zeros(output_size)
        self.layers += [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]
        self.loss_layer = SoftmaxWithLoss()
        super().post_init()

    