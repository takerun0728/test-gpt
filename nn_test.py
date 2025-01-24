import numpy as np
import matplotlib.pyplot as plt

from model import TwoLayerSoftMaxNet
from optimizer import SGD, Adam
from dataset import spiral
from utility import DataLoader

MAX_EPOCH = 3000
BATCH_SIZE = 30
HIDDEN_SIZE = 5

x, t = spiral.load_data()
out_dim = t.shape[1]
model = TwoLayerSoftMaxNet(x.shape[1], HIDDEN_SIZE, out_dim)
optimizer = Adam(model, lr=0.005)
t = np.argmax(t, axis=1, keepdims=True)
loader = DataLoader(x, t, BATCH_SIZE)

fig = plt.figure(figsize=(9, 4))
ax_l = fig.add_subplot(1, 2, 1)
ax_r = fig.add_subplot(1, 2, 2)
t_flat = t.flatten()
for i in range(out_dim):
    idxes = t_flat==i
    ax_l.scatter(x[idxes, 0], x[idxes, 1])
bx = np.linspace(-1, 1, 100)
by = np.linspace(-1, 1, 100)
bx, by = np.meshgrid(bx, by)
bx = np.hstack([bx.reshape([-1, 1]), by.reshape([-1, 1])])

for i in range(MAX_EPOCH):
    cond = True
    loss = 0
    while cond:
        data_x, data_y, end = loader.load()
        loss += model.forward_loss(data_x, data_y).sum()
        optimizer.update()
        cond = not end
    ax_r.plot(i, loss, c='g', ls='-', marker='.')

    ax_l.cla()
    bt = np.argmax(model.forward(bx), axis=1)
    for i in range(out_dim):
        idxes = bt==i
        c = [0.5, 0.5, 0.5]
        c[i] = 1.
        ax_l.scatter(bx[idxes, 0], bx[idxes, 1], c=c, s=2)
    for i in range(out_dim):
        c = [0., 0., 0.]
        c[i] = 1.
        idxes = t_flat==i
        ax_l.scatter(x[idxes, 0], x[idxes, 1], c=c)

    plt.pause(0.001)




pass