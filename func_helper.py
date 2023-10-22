import torch

def padCollate(batch):
    if type(batch[0]) is tuple: #train, evaluate
      x = [t for t, _ in batch]
      y = [t for _, t in batch]
      x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
      y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
      return x, y
    else: #test
      x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
      return x