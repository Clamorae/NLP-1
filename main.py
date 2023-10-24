import data
import compute
import node

import torch

import torch.nn as nn

from gensim.models import Word2Vec
from torch.utils.data import DataLoader

#=========CONSTANT=================================================
PATH = "./NLP/NLP-1/"
WORD_EMB_DIM = 128
BATCH_SIZE = 64
EPOCHS = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
label_size = 22
dropout_rate = 0.1
learning_rate = 0.01

#========DATA CREATION=============================================
sentences, labels = data.data_loader(PATH)

split = int(80*len(sentences)/100)

train_sentences = sentences[:split]
train_label = labels[:split]
val_sentences = sentences[split+1:]
val_label = labels[split+1:]
test_sentences = data.test_loader(PATH)


#========CREATE EMBEDDING==========================================
embedding_model = Word2Vec(sentences=train_sentences, vector_size=128, window=5, min_count=1, workers=8,sg = 1)
embedding_model.save(PATH + "word_embedding_model")
embedding_model = Word2Vec.load(PATH + 'word_embedding_model')

word_embedding = embedding_model.wv[list(embedding_model.wv.key_to_index.keys())].tolist()
word_embedding = [[0.5] * WORD_EMB_DIM] + [[0.] * WORD_EMB_DIM] + word_embedding


#========CREATE CORRESPONDING VOC=================================
vocab = ["<PAD>","<UNK>"] + list(embedding_model.wv.key_to_index.keys())
vocab_size = len(vocab)

idx2word = vocab
word2idx = dict((j,i) for i,j in enumerate(vocab))

idx2tag = ['<PAD>', 'O', 'B-company','I-company', 'B-facility', 'I-facility','B-geo-loc', 'B-movie', 'B-musicartist', 'B-other', 'B-person', 'B-product', 'B-sportsteam', 'B-tvshow','I-geo-loc', 'I-movie', 'I-musicartist', 'I-other', 'I-person', 'I-product', 'I-sportsteam', 'I-tvshow']
tag2idx = dict((j,i) for i,j in enumerate(idx2tag))

train_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in train_sentences]
train_targets_as_int = [[tag2idx[tag] for tag in sentence] for sentence in train_label]

eval_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in val_sentences]
eval_targets_as_int = [[tag2idx[tag] for tag in sentence] for sentence in val_label]

test_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in test_sentences]


#========FORMAT DATASET AND DATALOADER=============================
train_dataset = data.Dataset(train_words_as_int,train_targets_as_int)
train_loader = DataLoader(train_dataset,BATCH_SIZE,shuffle = True,collate_fn = compute.PadCollate)
eval_dataset = data.Dataset(eval_words_as_int,eval_targets_as_int)
eval_loader = DataLoader(eval_dataset,BATCH_SIZE, shuffle = True, collate_fn = compute.PadCollate)
test_dataset = data.Dataset(test_words_as_int)
test_loader = DataLoader(test_dataset, BATCH_SIZE,shuffle = False, collate_fn = compute.PadCollate)


#========TRAINING===================================================
loss_object = nn.CrossEntropyLoss(ignore_index = 0)
transformer = node.TaggingTransformer(num_layers, d_model, num_heads, dff, vocab_size, label_size, pe_input=vocab_size, word_emb=word_embedding, rate=dropout_rate).to(device)
optimizer = torch.optim.Adam(transformer.parameters(),lr = learning_rate)

for epoch in range(EPOCHS):  # loop over the dataset multiple times

  running_loss = 0.0

  transformer.train()

  for (batch, (inp, tar)) in enumerate(train_loader):
    inp = inp.to(device)
    tar = tar.to(device)
    enc_padding_mask = data.create_padding_mask(inp).to(device)

    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    predictions = transformer(inp, enc_padding_mask).transpose(1, 2)
    loss = loss_object(predictions, tar)

    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if batch % 50 == 0 and batch != 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(
          epoch + 1, batch, running_loss / 50))
      running_loss = 0.0

  if (epoch + 1) % 5 == 0:
    torch.save(transformer.state_dict(), PATH + '/checkpoint_{}'.format(epoch + 1))
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, PATH+ '/checkpoint_{}'.format(epoch + 1)))

  compute.evaluate(eval_loader, transformer, loss_object)

print('Finished Training')

#============PREDICTIONS===========================================
output_id = compute.predict(test_loader, transformer)
output_tag = [idx2tag[token] for token in output_id if token != 0]
labels = [item for sublist in output_tag for item in sublist]
labels = [item for sublist in labels for item in sublist]
with open(PATH+"test-submit.txt","r") as f:
    lines = f.readlines()
with open(PATH+"test_result.txt","w") as f:
    for line,label in zip(lines,labels):
        f.write(f"{line[:-1]} = {label}\n")