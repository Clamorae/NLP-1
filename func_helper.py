import torch
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
