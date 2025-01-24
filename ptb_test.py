from sklearn.utils.extmath import randomized_svd
import numpy as np

from dataset import ptb
from utility import create_co_matrix, ppmi, most_similar

WINDOWS_SIZE = 2
WORDVEC_SIZE = 10

corpus, word2id, id2word = ptb.load_data('train')
vocab_size = len(word2id)
C = create_co_matrix(corpus, vocab_size, WINDOWS_SIZE)
W = ppmi(C)
U, S, V = randomized_svd(W, n_components=WORDVEC_SIZE, n_iter=5, random_state=None)
U /= np.sqrt((U**2).sum(axis=1)).reshape(-1, 1)

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word2id, id2word, U, top=6)
pass

king_vec = U[word2id['king']]
man_vec = U[word2id['man']]
woman_vec = U[word2id['woman']]
most_similar(king_vec - man_vec + woman_vec, word2id, id2word, U, top=6, vec_mode=True)
