import helper
import time
import torch
import computational_func as cf

import random as rand
import torch.nn as nn
from torch.utils.data import DataLoader
from gensim.models import Word2Vec

#Constant (Hyperparameters)
path = "./NLP/NLP-1/"
EPOCHS = 20
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
label_size = 22
dropout_rate = 0.1
learning_rate = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#create an array with the dataset
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


#use word2vec to create a model and a vocabulary
word_emb_dim = 128
model_path = "./NLP/NLP-1/word2vec.model"
model = Word2Vec(sentences=train_sentences, vector_size=128, window=5, min_count=1, workers=4,sg=1)
model.save(model_path)
model = Word2Vec.load(model_path)

#using this model create an embeding of the training dataset
embedding = model.wv[list(model.wv.key_to_index)]
embedding = [[0.5] * word_emb_dim] + [[0.0] * word_emb_dim] + embedding.tolist()
#embedding = [[0.5] * word_emb_dim] + [[0.0] * word_emb_dim] + embedding
voc = ["<PAD>","<UNK>"] + list(model.wv.key_to_index.keys())
voc_size = len(voc)

#create a dictionnary of all the world
idx2word = voc
word2idx = dict((j,i) for i,j in enumerate(voc))
idx2tag = ['<PAD>', 'O', 'B-company','I-company', 'B-facility', 'I-facility','B-geo-loc', 'B-movie', 'B-musicartist', 'B-other', 'B-person', 'B-product', 'B-sportsteam', 'B-tvshow','I-geo-loc', 'I-movie', 'I-musicartist', 'I-other', 'I-person', 'I-product', 'I-sportsteam', 'I-tvshow']
tag2idx = dict((j,i) for i,j in enumerate(idx2tag))

#transform each vect in numerical value
train_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in train_sentences]
train_label_as_int = [[tag2idx[tag] for tag in sentence] for sentence in train_label]
val_words_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in val_sentences]
val_label_as_int = [[tag2idx[tag] for tag in sentence] for sentence in val_label]

#create the whole dataset using the previous value and the TODO
batch_size = 64
train_dataset = helper.CoNLLDataset(train_words_as_int,train_label_as_int)
val_dataset = helper.CoNLLDataset(val_words_as_int,val_label_as_int)
train_loader = DataLoader(train_dataset,batch_size,shuffle=True,collate_fn=cf.PadCollate)
val_loader = DataLoader(val_dataset,batch_size,shuffle=True,collate_fn=cf.PadCollate)


#NEED TO COMPLETE



loss_object = nn.CrossEntropyLoss(ignore_index = 0)

transformer = helper.TaggingTransformer(num_layers, d_model, num_heads, dff,
                          voc_size, label_size,
                          pe_input=voc_size,
                          word_emb=embedding,
                          rate=dropout_rate).to(device)

optimizer = torch.optim.Adam(transformer.parameters(),lr = learning_rate)

for epoch in range(EPOCHS):  # loop over the dataset multiple times
  start = time.time()

  running_loss = 0.0

  transformer.train()

  for (batch, (inp, tar)) in enumerate(train_loader):
    inp = inp.to(device)
    tar = tar.to(device)
    enc_padding_mask = cf.create_padding_mask(inp).to(device)

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
    torch.save(transformer.state_dict(), path + 'checkpoint_{}'.format(epoch + 1))
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, path + 'checkpoint_{}'.format(epoch + 1)))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  cf.evaluate(val_loader, transformer, loss_object)

print('Finished Training')

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

test_word_as_int = [[word2idx['<UNK>'] if word not in word2idx else word2idx[word] for word in sentence] for sentence in test_sentences]
test_dataset = helper.CoNLLDataset(test_word_as_int)
test_loader = DataLoader(test_dataset, batch_size,shuffle = False, collate_fn = cf.PadCollate)

output_id = cf.predict(test_loader, transformer)
output_tag = [idx2tag[token] for token in output_id if token != 0]
print(output_tag[:20])

output_position = 0
with open(path + 'test_result.txt', 'w') as f:
    for line in test_sentences:
        for word in line:
            f.write('{} {}\n'.format(word, output_tag[output_position]))
            output_position += 1
        f.write('\n')
