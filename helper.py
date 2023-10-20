import torch.nn as nn
import torch

#TODO improve and understand everything here

class CoNLLDataset(Dataset):
    def __init__(self, words, targets=None):
        self.words = words
        self.targets = targets

    def __getitem__(self, index):
        if self.targets == None:
            return torch.Tensor(self.words[index]).long()
        return torch.Tensor(self.words[index]).long(), torch.Tensor(self.targets[index]).long()

    def __len__(self):
        return len(self.words)

def PadCollate(batch):
    if type(batch[0]) is tuple: #train, evaluate
      x = [t for t, _ in batch]
      y = [t for _, t in batch]
      x = nn.utils.rnn.pad_sequence(x, batch_first=True)
      y = nn.utils.rnn.pad_sequence(y, batch_first=True)
      return x, y
    else: #test
      x = nn.utils.rnn.pad_sequence(batch, batch_first=True)
      return x