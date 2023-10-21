import torch
import math
import numpy as np
import torch.nn as nn

#TODO remove weird comment

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

def scaled_dot_product_attention(q, k, v, mask):
 
  matmul_qk = torch.matmul(q, k.transpose(-1,-2))  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = k.size(dim = 1)
  scaled_attention_logits = (matmul_qk/math.sqrt(dk))

  # add the mask to the scaled tensor.
  if mask is not None:
    mask = -1 * np.power(10,9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = nn.functional.softmax(scaled_attention_logits,dim = 1)  # (..., seq_len_q, seq_len_k)

  output = torch.matmul(attention_weights,v)# (..., seq_len_q, depth_v)

  return output, attention_weights