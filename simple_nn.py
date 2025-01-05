import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 1
EPOCH = 1000000

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

digit = load_digits()
scaler = StandardScaler()
x = scaler.fit_transform(digit.data)
y = digit.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = SimpleNN(x.shape[1], y.max()+1)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for i in range(EPOCH):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 100 == 0:
        
        with torch.no_grad():
            predicted = outputs.argmax(dim=1)
            accuracy_train = accuracy_score(y_train, predicted)
            outputs = model(x_test)
            predicted = outputs.argmax(dim=1)
            accuracy_test = accuracy_score(y_test, predicted)
        print(f"{i}, {loss}, {accuracy_train}, {accuracy_test}")

print(predicted)
print(y_test)
