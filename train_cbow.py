from utility import *

text = 'You say goodbye and I say hello.'
corpus, word2id, id2word = preprocess(text)
x, y = create_contexts_target(corpus)

sampler = UnigramSampler(corpus, 0.75)
a = sampler.get_samples(3)

pass