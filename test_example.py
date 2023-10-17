import torch
import torch.nn as nn

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
            #f.write(line.replace('\n','\tO\n'))
print(sentences)
print(label)