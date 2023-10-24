
import os
import node
import data
import torch
import compute

import torch.nn as nn

from gensim.models import Word2Vec

#CONSTANT==================================================================
PATH = "./NLP/NLP-1/"
WORD_EMB_DIM = 128
EPOCHS = 20
BATCH_SIZE = 128

classes = 22
hidden_size = 128
learning_rate = 0.01
print_freq = 5
save_freq = print_freq
layers = 128
emb_size = 128
weight_decay = 0.0001
clip = 0.25

#========DATA CREATION=============================================
sentences, labels = data.data_loader(PATH)

split = int(80*len(sentences)/100)

train_sentences = sentences[:split]
train_label = labels[:split]
val_sentences = sentences[split+1:]
val_label = labels[split+1:]
test_sentences = data.test_loader(PATH)


#========CREATE EMBEDDING==========================================
embedding_model = Word2Vec(sentences=train_sentences, vector_size=emb_size, window=5, min_count=1, workers=8,sg = 1)
embedding_model.save(PATH + "word_embedding_model")
embedding_model = Word2Vec.load(PATH + 'word_embedding_model')

word_embedding = embedding_model.wv[list(embedding_model.wv.key_to_index.keys())].tolist()
word_embedding = [[0.5] * WORD_EMB_DIM] + [[0.] * WORD_EMB_DIM] + word_embedding


#========CREATE CORRESPONDING VOC=================================
vocab = ["<PAD>","<UNK>"] + list(embedding_model.wv.key_to_index.keys())
word2idx = dict((j,i) for i,j in enumerate(vocab))
train_loader = data.TextClassDataLoader(train_sentences, train_label,word2idx ,BATCH_SIZE)
val_loader = data.TextClassDataLoader(val_sentences, val_label,word2idx ,BATCH_SIZE)
vocab_size = len(vocab)

#CREATE MODEL=====================================================
print("===> creating rnn model ...")
model = node.RNN(vocab_size=vocab_size, embed_size=emb_size, num_output=classes, rnn_model='LSTM',
            use_last=( True),
            hidden_size=hidden_size, embedding_tensor=word_embedding, num_layers=layers, batch_first=True)
print(model)

# optimizer and loss
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()
print(optimizer)
print(criterion)

# training and testing
for epoch in range(1, EPOCHS+1):

    compute.adjust_learning_rate(learning_rate, optimizer, epoch)
    compute.train(train_loader, model, criterion, optimizer, epoch, clip, print_freq)
    compute.test(val_loader, model, criterion,print_freq)

    # save current model
    if epoch % save_freq == 0:
        name_model = 'rnn_{}.pkl'.format(epoch)
        path_save_model = os.path.join('gen', name_model)