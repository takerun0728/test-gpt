from datasets import load_dataset
from ngram import NgramLM

dataset = load_dataset("QEU/databricks-dolly-16k-line_ja-1_of_4")
strs = ""
for i, r in enumerate(dataset['train']):
    strs += r['output']

ngram_lm = NgramLM()
ngram_lm.train(strs, 4)
test = "猫にペットの"
for _ in range(100):
    tmp = ngram_lm.predict(test)
    test += tmp
print(test)
