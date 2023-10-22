from torch.utils.data import Dataset
import torch.nn as nn
import torch

class Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.LongTensor(self.data[idx])  # Convert data to a tensor
        label_tensor = torch.LongTensor(self.labels[idx])  # Convert labels to a tensor
        return data_tensor, label_tensor

class NLPModel(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim, num_layers, dropout):
        super(NLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.LongTensor(self.data[idx])  # Convert data to a tensor
        return data_tensor