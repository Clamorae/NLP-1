import torch
import numpy as np
import math
from sklearn.metrics import accuracy_score,f1_score

def padCollate(batch):
    if type(batch[0]) is tuple: #train, evaluate
        x = [t for t, _ in batch]
        y = [t for _, t in batch]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True)
        return x, y
    else: #test
        x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
        return x
    
def compute_metrics(predictions, ground_truth_labels):
    flat_predictions = [p for batch in predictions for p in batch]
    flat_ground_truth_labels = [l for batch in ground_truth_labels for l in batch]
    acc =  accuracy_score(flat_ground_truth_labels, flat_predictions)
    f1 = f1_score(flat_ground_truth_labels,flat_predictions,average='micro')
    return acc,f1

def output(labels,path):
    labels = [item for sublist in labels for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    with open(path+"test-submit.txt","r") as f:
        lines = f.readlines()
    with open(path+"test_result.txt","w") as f:
        for line,label in zip(lines,labels):
            f.write(f"{line[:-1]} = {label}\n")

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

#   return tf.cast(pos_encoding, dtype=tf.float32)
  return pos_encoding