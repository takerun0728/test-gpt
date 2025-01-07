import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset, dataloader

class SimpleLSTM(nn.Module):
    def __init__(self, num_embedding, embedding_dim, hidden_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embedding, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_embedding)
        self.output = nn.Softmax(dim=1)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return self.output(x)

class TrainDataset(dataset.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

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

X = torch.tensor([seq[:-1] for seq in sequences], dtype=torch.int)
Y = torch.tensor([seq[1:] for seq in sequences], dtype=torch.int)

model = SimpleLSTM(len(char_to_int), 10, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
data = dataset.TensorDataset(X, Y)
loader = dataloader.DataLoader(dataset, batch_size=64, shuffle=True)

epoch = 10
for i in range(epoch):
    predict = loader.

pass