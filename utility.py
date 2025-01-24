import numpy as np

class DataLoader:
    def __init__(self, x, y, batch_size, seed=0):
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.idxes = np.arange(x.shape[0])
        self.start = 0
        self.batch_size = batch_size
        np.random.shuffle(self.idxes)

    def load(self):
        end = self.start + self.batch_size
        idxes = self.idxes[self.start:end]
        x = self.x[idxes]
        y = self.y[idxes]

        if end >= self.x.shape[0]:
            self.start = 0
            np.random.shuffle(self.idxes)
            end = True
        else:
            self.start = end
            end = False
        return x, y, end

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word2id = {}
    id2word = {}
    for word in words:
        if word not in word2id:
            new_id = len(word2id)
            word2id[word] = new_id
            id2word[new_id] = word

    corpus = [word2id[w] for w in words]

    return corpus, word2id, id2word

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

def cos_similarities(vec, mat, eps=1e-8):
    nvec = vec / np.sqrt(np.sum(vec**2) + eps)
    nmat = mat / np.sqrt(np.sum(mat**2, axis=1) + eps).reshape(-1, 1)
    return nmat @ nvec

def most_similar(query, word2id, id2word, word_matrix, top=5, vec_mode=False):
    if vec_mode:
        query_vec = query
    else:
        if query not in word2id:
            print('%s is not found % query')
            return
        print(f'query:{query}')
        query_id = word2id[query]
        query_vec = word_matrix[query_id]
    similarities = cos_similarities(query_vec, word_matrix)
    idxes = np.argsort(similarities)[::-1]
    
    for i in range(top):
        print(f"{id2word[idxes[i]]}:{similarities[idxes[i]]}")
    pass

def ppmi(C, eps=1e-8):
    N = np.sum(C)
    S = np.sum(C, axis=0).reshape(-1, 1)
    S = S @ S.T
    return np.maximum(0, np.log2(C * N / (S + eps)))


