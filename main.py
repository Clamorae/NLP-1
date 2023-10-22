import time
import torch

import random as rand
import torch.nn as nn
import class_helper as ch
import func_helper as fh
import torch.optim as optim

from torch.utils.data import DataLoader
from gensim.models import Word2Vec

#TODO improve splitting

#Constant and Hyperparameters
path = "./NLP/NLP-1/"
EPOCHS = 20
num_layers = 4
dim_model = 128
dff = 512
num_heads = 8
label_size = 22
batch_size = 64
dropout_rate = 0.1
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#create an array with all the sentences in the dataset
sentences = []
label = []

with open(path+'train.txt','r') as f:
    lines = f.readlines()
current_word = []
current_label = []
for line in lines:
    if line=='\n': 
        sentences.append(current_word)
        label.append(current_label)
        current_word = []
        current_label = []
    else:
        separate = line.split('\t')
        current_word.append(separate[0])
        current_label.append(separate[1].split('\n')[0])

# split the dataset using the 80/20
train_sentences = []
train_label = []
val_sentences = []
val_label = []
for i in range(len(sentences)):
    x = rand.randint(1,100)
    if x>80:
        val_sentences.append(sentences[i])
        val_label.append(label[i])
    else:
        train_sentences.append(sentences[i])
        train_label.append(label[i])

# Try to apply this to the test value
with open(path+'example.txt','r') as f:
    lines = f.readlines()
test_sentences = []

current_word = []
for line in lines:
    if line=='\n': 
        test_sentences.append(current_word)
        current_word = []
    else:
        separate = line.split('\t')
        current_word.append(separate[0])


#use word2vec to create a model and a vocabulary
word_emb_dim = 128
model_path = "./NLP/NLP-1/word2vec.model"
model = Word2Vec(sentences=train_sentences, vector_size=128, window=5, min_count=1, workers=4,sg=1)
model.save(model_path)
model = Word2Vec.load(model_path)

#using this model create an embeding of the training dataset
embedding = model.wv[list(model.wv.key_to_index)]
embedding = [[0.5] * word_emb_dim] + [[0.0] * word_emb_dim] + embedding.tolist()
voc = ["<PAD>","<UNK>"] + list(model.wv.key_to_index.keys())
voc_size = len(voc)

#create a dictionnary of all the words
idx2word = voc
word2idx = dict((j,i) for i,j in enumerate(voc))
idx2tag = ['<PAD>', 'O', 'B-company','I-company', 'B-facility', 'I-facility','B-geo-loc', 'B-movie', 'B-musicartist', 'B-other', 'B-person', 'B-product', 'B-sportsteam', 'B-tvshow','I-geo-loc', 'I-movie', 'I-musicartist', 'I-other', 'I-person', 'I-product', 'I-sportsteam', 'I-tvshow']
tag2idx = dict((j,i) for i,j in enumerate(idx2tag))

#transform each vect in numerical value
train_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in train_sentences]
train_label_as_int = [[tag2idx[tag] for tag in sentence] for sentence in train_label]
val_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in val_sentences]
val_label_as_int = [[tag2idx[tag] for tag in sentence] for sentence in val_label]
test_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in test_sentences]

# Create datasets and dataloaders
train_dataset = ch.Dataset(train_words_as_int, train_label_as_int)
val_dataset = ch.Dataset(val_words_as_int, val_label_as_int)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fh.padCollate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=fh.padCollate)

model = ch.NLPModel(voc_size, label_size, embedding_dim=128, hidden_dim=128, num_layers=num_layers, dropout=dropout_rate).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Start Training")
for epoch in range(EPOCHS):
    model.train()
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.view(-1, label_size), labels.view(-1))
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            # Calculate validation metrics
print("end of training")

# Save the trained model
torch.save(model.state_dict(), 'nlp_model.pth')

model = ch.NLPModel(voc_size, label_size, embedding_dim=128, hidden_dim=128, num_layers=num_layers, dropout=dropout_rate).to(device)
model.load_state_dict(torch.load('nlp_model.pth'))
model.eval()
test_dataset = ch.TestDataset(test_words_as_int)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=fh.padCollate)

correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    predicted_labels = []
    for data in test_loader:
        data = data.to(device)
        outputs = model(data)
        predictions = torch.argmax(outputs, dim=2)  # Predicted labels
        batch_labels = []  # Store labels for a batch
        for class_id in predictions:
            label = [idx2tag[c_id.item()] for c_id in class_id] # Convert class ID to label
            batch_labels.append(label)
        predicted_labels.append(batch_labels)

        # Calculate accuracy
        # correct_predictions += (predictions == ground_truth_labels).sum().item()
        # total_predictions += data.size(0) * data.size(1)  # Total number of tokens
print(predicted_labels)
# accuracy = correct_predictions / total_predictions
# print(f'Test Accuracy: {accuracy * 100:.2f}%')