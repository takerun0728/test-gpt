import torch
from sentence_transformers import SentenceTransformer

model_st = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

sentences = ["I went river fishing and stood still near the bank for a while.", "I went to the bank for withdrawing money.", "I played fishing near the ocean."]
embeddings = model_st.encode(sentences)

s1 = torch.tensor(embeddings[0])
s2 = torch.tensor(embeddings[1])
s3 = torch.tensor(embeddings[2])

print(s1)

similarity = torch.cosine_similarity(s1, s2, dim=0)
print(f"s1-s2:{similarity}")
similarity = torch.cosine_similarity(s2, s3, dim=0)
print(f"s2-s3:{similarity}")
similarity = torch.cosine_similarity(s1, s3, dim=0)
print(f"s1-s3:{similarity}")