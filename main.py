import data
import torch

import torch.nn as nn

from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from torch.autograd import Variable

# --------------------------------- CONSTANT --------------------------------- #
PATH = "./NLP/NLP-1/"
WORD_EMB_DIM = 128
EPOCHS = 20

learning_rate = 0.01
emb_size = 128
layer_dim = 4
hidden_dim = 4

# ------------------------------- DATA CREATION ------------------------------ #
dataset = data.data_loader(PATH)
splitted = train_test_split(dataset)
train_set = splitted[0]
test_set = splitted[1]

train_loader = data.Loader(train_set)
test_loader = data.Loader(test_set)

train_embedding_model = Word2Vec(sentences=train_loader.getItem(), vector_size=128, window=5, min_count=1, workers=8, sg = 1)
train_embedding = train_embedding_model.wv[list(train_embedding_model.wv.key_to_index.keys())].tolist()
train_embedding = [[0.5] * WORD_EMB_DIM] + [[0.] * WORD_EMB_DIM] + train_embedding

test_embedding_model = Word2Vec(sentences=test_loader.getItem(), vector_size=128, window=5, min_count=1, workers=8, sg = 1)
test_embedding = test_embedding_model.wv[list(test_embedding_model.wv.key_to_index.keys())].tolist()
test_embedding = [[0.5] * WORD_EMB_DIM] + [[0.] * WORD_EMB_DIM] + test_embedding

vocab = ["<PAD>","<UNK>"] + list(set(train_loader.getLabel()))
vocab_size = len(vocab)
vocab_tensor = []
for i in range(vocab_size):
    vocab_tensor.append(torch.tensor([0.0]*i + [1.0] + [0.0]*(vocab_size-i-1)))


train_data = [train_embedding,train_loader.getLabel()]
test_data = [test_embedding,test_loader.getLabel()]

# ------------------------------- RNN CREATION ------------------------------- #

class RNN(nn.Module):
    def __init__(self,hidden_dim,layer_dim,input_dim,output_dim):
        super(RNN,self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    
    def forward(self,x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, self.hidden_dim))
        x,hn = self.rnn(x,h0)
        x = self.fc(x)
        return x

rnn = RNN(hidden_dim,layer_dim,emb_size,vocab_size)        
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# --------------------------------- TRAINING --------------------------------- #

for epoch in range(EPOCHS):
    true = 0
    o = 0
    for i in range(len(train_data[0])):
        optimizer.zero_grad()   
        outputs = rnn(torch.tensor(train_data[0][i])[None, ...])
        lab = (vocab_tensor[vocab.index(train_data[1][i])])
        loss = criterion(outputs,lab.view(-1,23))
        loss.backward()
        optimizer.step()
        if vocab[torch.argmax(outputs)] == train_data[1][i]:
            true+=1
        if train_data[1][i] == "O":
            o+=1
        
    print(f"{true}/{i}, O = {o}")
    print(f"EPOCH : {epoch}, accuracy = {true/i*100}%")

