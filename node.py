import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, use_last=True, embedding_tensor=None, padding_index=0, hidden_size=64, num_layers=1, batch_first=True):
        super(RNN, self).__init__()
        self.use_last = use_last
        # embedding
        self.encoder = None
        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index, _weight=embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)

        self.drop_en = nn.Dropout(p=0.6)
        self.rnn = nn.LSTM( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, num_output)

    def forward(self, x, seq_lengths):

        x = self.encoder(x)
        x = self.drop_en(x)
        #packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(),batch_first=True)
        x, ht = self.rnn(x)
        #out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)
        x = self.bn2(x)
        out = self.fc(x)
        return out
