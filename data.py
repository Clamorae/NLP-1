import torch
from torch.utils.data.dataset import Dataset

def data_loader(path):
    sentences = []
    label = []

    with open(path+'train.txt','r') as f:
        lines = f.readlines()
    current_word = []
    current_label = []
    for line in lines:
        if line=='\n': 
            sentences.append(current_word)
            label.append(current_label)
            current_word = []
            current_label = []
        else:
            separate = line.split('\t')
            current_word.append(separate[0])
            current_label.append(separate[1].split('\n')[0])
    return sentences, label

def test_loader(path):
    with open(path+'example.txt','r') as f:
        lines = f.readlines()
    test_sentences = []

    current_word = []
    for line in lines:
        if line=='\n': 
            test_sentences.append(current_word)
            current_word = []
        else:
            separate = line.split('\t')
            current_word.append(separate[0])
    return test_sentences

class Dataset(Dataset):
    def __init__(self, words, targets=None):
        self.words = words
        self.targets = targets

    def __getitem__(self, index):
        if self.targets == None:
            return torch.Tensor(self.words[index]).long()
        return torch.Tensor(self.words[index]).long(), torch.Tensor(self.targets[index]).long()

    def __len__(self):
        return len(self.words)

def create_padding_mask(seq):
  seq = seq.bool().float()
  return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)