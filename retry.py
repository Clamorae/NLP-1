import os
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

path = "./NLP/NLP-1/word2vec.model"
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4,sg=1)
model.save(path)
model = Word2Vec.load(path)

vocabulary = list(model.wv.key_to_index)

