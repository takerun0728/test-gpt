import gensim
from itertools import combinations

model = gensim.models.keyedvectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
queen = model["King"] - model["Man"] + model["Woman"]
print(model.most_similar(queen, topn=10))

fruits = ["pen", "pineapple", "banana"]
for comb in combinations(fruits, 2):
    print(comb[0], comb[1], model.similarity(comb[0], comb[1]))