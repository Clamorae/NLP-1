import os
import random as rand
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
vald_label = []
for i in len(sentences):
    x = rand.randint(1,100)
    if x>80:
        val_sentences.append(sentences[i])
        vald_label.append(label[i])
    else:
        train_sentences.append(sentences[i])
        train_label.append(label[i])


#use word2vec to create a model and a vocabulary
word_emb_dim = 128
path = "./NLP/NLP-1/word2vec.model"
model = Word2Vec(sentences=train_sentences, vector_size=128, window=5, min_count=1, workers=4,sg=1)
model.save(path)
model = Word2Vec.load(path)

