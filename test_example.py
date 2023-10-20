import torch
import torch.nn as nn
import os
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.models import Word2Vec

#TODO - separate dataset
#TODO - better emb model
#TODO - CNN or RNN
#TODO - train and test model

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

path = "./NLP/NLP-1/word2vec.model"
if os.isfile(path):
    model = Word2Vec.load(path)
else:
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save("./NLP/NLP-1/word2vec.model")


emb_dim = 128
word_embedding = model.wv[list(model.wv.key_to_index.keys())].tolist()
word_embedding = [[0.5] * emb_dim] + [[0.] * emb_dim] + word_embedding

vocabulary = ["<PAD>","<UNK>"] + list(model.wv.key_to_index.keys())
voc_size = len(vocabulary)

idx2word = vocabulary
word2idx = dict((j,i) for i,j in enumerate(vocabulary))

idx2tag = ['<PAD>', 'O', 'B-company','I-company', 'B-facility', 'I-facility','B-geo-loc', 'B-movie', 'B-musicartist', 'B-other', 'B-person', 'B-product', 'B-sportsteam', 'B-tvshow','I-geo-loc', 'I-movie', 'I-musicartist', 'I-other', 'I-person', 'I-product', 'I-sportsteam', 'I-tvshow']
tag2idx = dict((j,i) for i,j in enumerate(idx2tag))



#========================== Pytorch part ================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = 64

