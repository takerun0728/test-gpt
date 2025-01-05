import torch

text_data = """むかしむかしあるところに、
お爺さんとお婆さんがいました。
お爺さんは山へ柴刈りに、
お婆さんは川へ洗濯に行きました。"""
char_to_int = {c:i for i, c in enumerate(sorted(set(text_data)))}
int_to_char = {i:c for c, i in char_to_int.items()}
encoded = [char_to_int[c] for c in text_data]
decoded = [int_to_char[i] for i in encoded]

seq_length = len(text_data.split('\n')[0])
sequences = [encoded[i:i+seq_length+1] for i in range(len(encoded) -seq_length)]

X = torch.Tensor([seq[:-1] for seq in sequences])
Y = torch.Tensor([seq[1:] for seq in sequences])

pass