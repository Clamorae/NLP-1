import torch
import string

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from sklearn.metrics import f1_score

# --------------------------------- CONSTANT --------------------------------- #
PATH = "./NLP/NLP-1/"
BATCH_SIZE = 64

word_emb_dim = 46
epochs = 20
learning_rate = 0.01
windows = 5
layer_dim = 4
hidden_dim = 4
dropout = 0.1

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
                if separate[0] not in string.punctuation:
                    sentence.append(separate[0])
                    label.append(separate[1].split('\n')[0])
            else:
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
max_size = max(len(seq) for seq in train_loader.getItem())

# -------------------------------- DATALOADER -------------------------------- #

train_int = [[target2index[target] for target in words] for words in train_loader.getLabel()]
train_words_embedded = encoder(train_loader.getItem(), word2index, word_embedding)
train_inputs = torch.Tensor(train_words_embedded).float()
train_targets = torch.Tensor(train_int).long()
train_dataset = TensorDataset(train_inputs, train_targets)
train_load = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

for epoch in range(epochs):
    sum_loss = 0
    correct = 0
    total = 0

    class_wise_true_positives = [0] * nb_class
    class_wise_false_positives = [0] * nb_class
    class_wise_false_negatives = [0] * nb_class

    all_predicted = []
    all_actual = []

    for sentence, label in train_load:

        optimizer.zero_grad()   
        outputs = rnn(sentence)
        loss = criterion(outputs.view(-1, 22),label.view(-1))
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        _, predicted = outputs.max(2)
        correct += (predicted.view(-1) == label.view(-1)).sum().item()
        total += label.view(-1).size(0)

        all_predicted.extend(predicted.view(-1).cpu().numpy())
        all_actual.extend(label.view(-1).cpu().numpy())


    average_loss = sum_loss / len(train_load) #modify
    sum_loss += loss.item()  # Accumulate the loss for this batch
    accuracy = (correct/ total) * 100.0
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
    f1_scores = f1_score(all_actual, all_predicted, average=None)

    for i in range(nb_class):
        print(f"F1 Score (Class {index2target[i]}): {f1_scores[i]*100:.2f}%")

    average_f1 = sum(f1_scores) / len(f1_scores)
    print(f"Average F1 Score: {average_f1*100:.2f}%")
    print("|------------------------------------------------------------|")
