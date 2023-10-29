import torch
import string

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from sklearn.metrics import f1_score

# --------------------------------- CONSTANT --------------------------------- #
PATH = "./NLP/NLP-1/"
BATCH_SIZE = 16

word_emb_dim = 45
epochs = 20
learning_rate = 0.01
windows = 7
layer_dim = 128
hidden_dim = 128
dropout = 0.3

# ------------------------------- DATA LOADING ------------------------------ #
class Loader():
    def __init__(self, path) -> None:
        self.item = []
        self.target = []

        with open(path,'r') as f:
            lines = f.readlines()
        sentence = []
        label = []
        for line in lines:
            if line!='\n': 
                separate = line.split('\t')
                if separate[0] not in string.punctuation and separate[0][0]!="@":
                    sentence.append(separate[0])
                    label.append(separate[1].split('\n')[0])

            else:
                if all(i == label[0] for i in label) == False:
                    self.item.append(sentence)
                    self.target.append(label)
                sentence = []
                label = []
        
    def getItem(self):
        return self.item
    
    def getLabel(self):
        return self.target
    
    def setItem(self,item):
        self.item=item
    
    def setLabel(self, label):
        self.target = label

train_loader = Loader(PATH+"train.txt")
val_loader = Loader(PATH+"dev.txt")

# ---------------------------- DATA EMBEDDING ---------------------------- #

embedding_model = Word2Vec(sentences=train_loader.getItem(),vector_size=word_emb_dim,window=windows,min_count=1,workers=8,sg=1)
word_embedding = [[0.5] * word_emb_dim, [0.] * word_emb_dim]

vocab = list(embedding_model.wv.index_to_key)
for word in vocab:
    word_embedding.append(embedding_model.wv[word].tolist())

vocab = ['<PAD>', '<UNK>'] + vocab
vocab_size = len(vocab)

index2word = vocab
word2index = dict((j,i) for i,j in enumerate(vocab))

index2target = ['<PAD>','O','B-tvshow','I-tvshow','B-geo-loc','I-geo-loc','B-company','I-company','B-person','I-person','B-movie','I-movie','B-facility','I-facility','B-sportsteam','I-sportsteam','B-musicartist','I-musicartist','B-product','I-product','B-other','I-other']
nb_class = len(index2target)

target2index = dict((j,i) for i,j in enumerate(index2target))

def encoder(data,word2index,word_embedding):
    dataset = []
    for line in data:
        current_sentence =[]
        for word in line:
            if word in word2index:
                index = word2index.get(word)
            else:
                index = word2index.get("<UNK>")
            current_sentence.append(word_embedding[index])
        dataset.append(current_sentence)
    return(dataset)

# ---------------------------------- PADDING --------------------------------- #

max_size = max(len(seq) for seq in train_loader.getItem())

def addPadding(data, max_size, windows):
    padded_data = []
    for line in data:
        diff = max_size - len(line)
        padded_line = ['<PAD>'] * windows + line + ['<PAD>'] * (windows + diff)
        padded_data.append(padded_line)
    return padded_data

train_loader.setItem(addPadding(train_loader.getItem(),max_size,windows))
train_loader.setLabel(addPadding(train_loader.getLabel(),max_size,windows))
val_loader.setItem(addPadding(val_loader.getItem(),max_size,windows))
val_loader.setLabel(addPadding(val_loader.getLabel(),max_size,windows))
max_size = max(len(seq) for seq in train_loader.getItem())

# -------------------------------- DATALOADER -------------------------------- #

train_int = [[target2index[target] for target in words] for words in train_loader.getLabel()]
train_words_embedded = encoder(train_loader.getItem(), word2index, word_embedding)
train_inputs = torch.Tensor(train_words_embedded).float()
train_targets = torch.Tensor(train_int).long()
train_dataset = TensorDataset(train_inputs, train_targets)
train_load = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_int = [[target2index[target] for target in words] for words in val_loader.getLabel()]
val_word_embedded = encoder(val_loader.getItem(),word2index,word_embedding)
val_input = torch.Tensor(val_word_embedded).float()
val_targets = torch.Tensor(val_int).long()
val_dataset = TensorDataset(val_input,val_targets)
val_load = DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=True)
# ------------------------------- RNN CREATION ------------------------------- #

class RNN(nn.Module):
    def __init__(self,hidden_dim,layer_dim,input_dim,output_dim,dropout_rate):
        super(RNN,self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        
        # RNN
        self.gru = nn.GRU(input_size=input_dim,hidden_size=hidden_dim,dropout=dropout_rate)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    
    def forward(self,x):
        x,_ = self.gru(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

rnn = RNN(hidden_dim,layer_dim,max_size,nb_class,dropout)        
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# --------------------------------- TRAINING --------------------------------- #

def iterate(loader, isEval = False):
    sum_loss = 0
    correct = 0
    total = 0

    all_predicted = []
    all_actual = []

    for sentence, label in loader:

        optimizer.zero_grad()   
        outputs = rnn(sentence)
        loss = criterion(outputs.view(-1, 22),label.view(-1))
        if isEval == False:    
            loss.backward()
            optimizer.step()

        sum_loss += loss.item()
        _, predicted = outputs.max(2)
        correct += (predicted.view(-1) == label.view(-1)).sum().item()
        total += label.view(-1).size(0)

        all_predicted.extend(predicted.view(-1).cpu().numpy())
        all_actual.extend(label.view(-1).cpu().numpy())


    average_loss = sum_loss / len(loader)
    sum_loss += loss.item()
    accuracy = (correct/ total) * 100.0
    print(f"|-------------Epoch [{epoch + 1}/{epochs}]--------------|\n Loss: {average_loss:.4f},\n Accuracy: {accuracy:.2f}%")
    f1_scores = f1_score(all_actual, all_predicted, average=None)

    # for i in range(nb_class):
    #     print(f"F1 Score (Class {index2target[i]}): {f1_scores[i]*100:.2f}%")

    average_f1 = sum(f1_scores) / len(f1_scores)
    print(f" Average F1 Score: {average_f1*100:.2f}%")
    print("|----------------------------------------|\n\n")

for epoch in range(epochs):
    iterate(train_load)

# --------------------------------- EVALUATE --------------------------------- #

    if (epoch+1)%5==0:
        with torch.no_grad():
            print("|------------EVALUATION STEP-------------|")
            iterate(val_load,True)