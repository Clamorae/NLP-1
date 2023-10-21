import torch
import torch.nn as nn
import computational_func as cf
from torch.utils.data.dataset import Dataset

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

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads

    assert d_model % num_heads == 0

    self.depth = d_model // num_heads

    self.wq = torch.nn.Linear(d_model,d_model)
    self.wk = torch.nn.Linear(d_model,d_model)
    self.wv = torch.nn.Linear(d_model,d_model)

    self.wo = torch.nn.Linear(d_model,d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = x.view(batch_size, -1, self.num_heads, self.depth)
    return x.transpose(1, 2)

  def forward(self, q, k, v, mask):
    batch_size = q.size(0)
    print(mask)

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q =  self.split_heads(q,batch_size) # (batch_size, num_heads, seq_len_q, depth)
    k =  self.split_heads(k,batch_size) # (batch_size, num_heads, seq_len_k, depth)
    v =  self.split_heads(v,batch_size) # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = cf.scaled_dot_product_attention(q,k,v,mask)

    scaled_attention =  scaled_attention.transpose(1,2).contiguous() # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = scaled_attention.view(batch_size, -1, self.num_heads * self.depth)  # (batch_size, seq_len_q, d_model)

    output =  self.wo(concat_attention) # (batch_size, seq_len_q, d_model)

    return output, attention_weights