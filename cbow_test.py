import matplotlib.pyplot as plt

from utility import *
from model import *
from optimizer import *

HIDDEN_SIZE = 5
BATCH_SIZE = 3
MAX_EPOCH = 500

text = 'You say goodbye and I say hello.'
corpus, word2id, id2word = preprocess(text)
x, y = create_contexts_target(corpus)

model = SimpleCBOW(len(word2id), HIDDEN_SIZE)
optimizer = SGD(model, lr=0.001)

x1 = id2onehot(x[:, 0], len(word2id))
x2 = id2onehot(x[:, 1], len(word2id))
x = np.hstack([x1, x2])
x = x.reshape(-1, 2, len(word2id))

loader = DataLoader(x, y, BATCH_SIZE)
for epoch in range(MAX_EPOCH):
    cond = True
    loss = 0
    while cond:
        xd, yd, end = loader.load()
        loss += model.forward_loss(xd, yd).sum()
        optimizer.update()
        cond = not end
    plt.plot(epoch, loss, c='g', ls='-', marker='.')
    plt.pause(0.01)

y2 = np.argmax(model.forward(x), axis=1)
pass
