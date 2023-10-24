import compute

import math
import torch

import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def point_wise_feed_forward_network(d_model, dff):
  return nn.Sequential(
      nn.Linear(d_model,dff), # (batch_size, seq_len, dff)
      nn.ReLU(),
      nn.Linear(dff,d_model)  # (batch_size, seq_len, d_model)
  )

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
    scaled_attention, attention_weights = compute.scaled_dot_product_attention(q,k,v,mask)

    scaled_attention =  scaled_attention.transpose(1,2).contiguous() # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = scaled_attention.view(batch_size, -1, self.num_heads * self.depth)  # (batch_size, seq_len_q, d_model)

    output =  self.wo(concat_attention) # (batch_size, seq_len_q, d_model)

    return output, attention_weights

class LayerNorm(nn.Module): #不用修改
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads).to(device)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = LayerNorm(d_model)
    self.layernorm2 = LayerNorm(d_model)

    self.dropout1 = nn.Dropout(rate)
    self.dropout2 = nn.Dropout(rate)

  def forward(self, x, mask):
    x = x.float()
    attn_output, _ = self.mha(x,x,x,mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads).to(device)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = LayerNorm(d_model)
    self.layernorm2 = LayerNorm(d_model)

    self.dropout1 = nn.Dropout(rate)
    self.dropout2 = nn.Dropout(rate)

  def forward(self, x, mask):
    x = x.float()
    attn_output, _ = self.mha(x,x,x,mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output)
    out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class Encoder(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, word_emb, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.word_emb = nn.Embedding.from_pretrained(torch.FloatTensor(word_emb))
    self.embedding = nn.Embedding.from_pretrained
    self.pos_encoding = compute.positional_encoding(maximum_position_encoding ,d_model)

    self.enc_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,dff,rate) for j in range(self.num_layers)])

    self.dropout = nn.Dropout(rate)

  def forward(self, x, mask):

    seq_len = x.size(1)

    # adding embedding and position encoding.
    x = self.word_emb(x)  # (batch_size, input_seq_len, d_model)

    x *= math.sqrt(self.d_model)
    x = (x.cpu() + self.pos_encoding[:, :seq_len, :]).to(device)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x,mask)

    return x  # (batch_size, input_seq_len, d_model)

class TaggingTransformer(nn.Module):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               label_size, pe_input, word_emb):
    super(TaggingTransformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads,dff, input_vocab_size,
                         pe_input, word_emb=word_emb).to(device)

    self.final_layer = nn.Linear(d_model,label_size)

  def forward(self, inp, enc_padding_mask):

    enc_output = self.encoder(inp,enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

    final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, label_size)

    return final_output
