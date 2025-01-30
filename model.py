import numpy as np
from layers import *
from utility import *

class Model:
    def __init__(self):
        self.params, self.grads, self.layers = [], [], []

    def post_init(self):
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x, train=False):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def forward_loss(self, x, t):
        y = self.forward(x, True)
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

class SimpleCBOW(Model):
    def __init__(self, word_dim, hidden_size, sigma=0.01):
        super().__init__()
        Win = sigma * np.random.randn(word_dim, hidden_size)
        Wout = sigma * np.random.randn(hidden_size, word_dim)
        self.in_layer = MatMul(Win)
        self.out_layer = MatMul(Wout)

        self.loss_layer = SoftmaxWithLoss()

        self.params += self.in_layer.params
        self.params += self.out_layer.params

        self.grads += self.in_layer.grads
        self.grads += self.out_layer.grads

    def forward(self, x, train=False):
        x1 = self.in_layer.forward(x[:,0,:], train)
        x1 += self.in_layer.forward(x[:,1,:], train)
        x1 *= 0.5
        return self.out_layer.forward(x1, train)

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        dout = self.out_layer.backward(dout)
        dout = self.in_layer.backward(dout*0.5)

class CBOW(Model):
    def __init__(self, word_dim, hidden_size, corpus, power=0.75, sample_size=5, sigma=0.01):
        super().__init__()
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power)

        Win = sigma * np.random.randn(word_dim, hidden_size)
        Wout = sigma * np.random.randn(hidden_size, word_dim)
        self.in_layer = Embedding(Win)
        self.out_layer = EmbeddingDot(Wout)

        self.loss_layer = SigmoidWithLoss()

    def forward(self, x):
        x1 = self.in_layer.forward(x[:,0])
        x2 = self.in_layer.forward(x[:,1])
        self.sampler.get_samples(self.sample_size)
        samples = self.sampler(self.sample_size)


        