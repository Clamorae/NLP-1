from torch.utils.data import Dataset
import torch.nn as nn
import torch
import func_helper as fh

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
    def __init__(self, vocab_size, num_classes, embedding_dim, hidden_dim, num_layers, dropout,word_emb,d_model):
        super(NLPModel, self).__init__()
        self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(word_emb))
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = fh.positional_encoding(vocab_size ,d_model)

    def forward(self, x):
        seq_len = x.size(1)
        embedded = self.word_emb(x)
        embedded = (embedded.cpu() + self.pos_encoding[:, :seq_len, :]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        embedded = self.dropout(embedded)
        embedded = embedded.to(torch.float)
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