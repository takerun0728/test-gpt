from janome import tokenizer
from collections import defaultdict

class NgramLM:
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(int))
        self.tokenizer = tokenizer.Tokenizer()

    def tokenize(self, text, word=True):
        if word:
            return [w for w in self.tokenizer.tokenize(text, wakati=True)]
        else:
            return text

    def ngram(self, text, n):
        return zip(*[text[i:] for i in range(self.n)])

    def train(self, text, n, word=True):
        self.word = word
        self.n = n
        text = self.tokenize(text, word)
        
        for w in self.ngram(text, n):
            if w[-1] != '\n':
                self.model[w[:-1]][w[-1]] += 1

    def predict(self, prefix):
        prefix = self.tokenize(prefix, self.word)
        
        next_words = []
        for next, cnt in self.model[tuple(prefix[-(self.n-1):])].items():
            next_words.append((next, cnt))

        return max(next_words, key=lambda x: x[1])[0]

if __name__ == "__main__":
    ngram_lm = NgramLM()
    ngram_lm.train("吾輩は猫、吾輩は犬、私は人間、吾輩は猫。", 3, True)
    print(ngram_lm.predict("私は"))


pass