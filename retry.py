import helper
import computational_func as cf

import random as rand
from torch.utils.data import DataLoader
from gensim.models import Word2Vec

with open('./NLP/NLP-1/train.txt','r') as f:
    lines = f.readlines()

#create an array with the dataset
sentences = []
label = []

with open('./NLP/NLP-1/example.txt','w') as f:
    current_word = []
    current_label = []
    for line in lines:
        if line=='\n': 
            f.write(line)
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
path = "./NLP/NLP-1/word2vec.model"
model = Word2Vec(sentences=train_sentences, vector_size=128, window=5, min_count=1, workers=4,sg=1)
model.save(path)
model = Word2Vec.load(path)

#using this model create an embeding of the training dataset
embedding = model.wv[list(model.wv.key_to_index)]
embedding = [[0.5] * word_emb_dim] + [[0.] * word_emb_dim] + embedding
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
train_loader = DataLoader(train_dataset,batch_size,shuffle=True,collate_fn=helper.PadCollate)
val_loader = DataLoader(val_dataset,batch_size,shuffle=True,collate_fn=helper.PadCollate)

