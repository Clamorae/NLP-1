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

def point_wise_feed_forward_network(d_model, dff):
  return nn.Sequential(
      nn.Linear(d_model,dff), # (batch_size, seq_len, dff)
      nn.ReLU(),
      nn.Linear(dff,d_model)  # (batch_size, seq_len, d_model)
  )

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

#   return tf.cast(pos_encoding, dtype=tf.float32)
  return pos_encoding

def create_padding_mask(seq):
  seq = seq.bool().float()

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)

def evaluate(eval_loader, model, loss_object):
  model.eval()

  with torch.no_grad():

    running_loss = 0.0
    correct = 0
    total = 0

    for (batch, (inp, tar)) in enumerate(eval_loader):
      inp = inp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      tar = tar.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      enc_padding_mask = create_padding_mask(inp).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

      # predictions.shape == (batch_size, seq_len, label_size)
      predictions = model(inp, enc_padding_mask)

      _, predictions_id = torch.max(predictions, -1)

      predictions_id *= tar.bool().long()

      correct += (predictions_id == tar).sum().item() - (tar == 0).sum().item()
      total += (tar != 0).sum().item()

      loss = loss_object(predictions.transpose(2,1),tar)

      running_loss += loss.item()

    print('Evaluate loss: {:.4f} acc: {:.4f}%\n'.format(running_loss / len(eval_loader), 100 * correct / total))