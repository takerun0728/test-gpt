import numpy as np
import matplotlib.pyplot as plt

from utility import preprocess, create_co_matrix, most_similar, ppmi

text = 'You say goodbye and I say hello.'
corpus, word2id, id2word = preprocess(text)
vocab_size = len(word2id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word2id, id2word, C, top=6)

W = ppmi(C)
U, S, V = np.linalg.svd(W)

ax = plt.figure().add_subplot(projection='3d')
for word, word_id in word2id.items():
    ax.text(U[word_id, 0], U[word_id, 1], U[word_id, 2], word)
ax.scatter(U[:,0], U[:,1], U[:, 2])
plt.show()