'''
    In this script, we will be doing the following
        1. Loading the data
        2. Tokenize the data
        3. Constructing the vocabulary
        4. Create embeddings using GloVe 100 dimensional 
        5. Creating the IMDBdataset class
        6. Creating train and validation datasets
'''

## Libraries 
import torch
from torch.utils.data import  DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from utils.preprocess_utils import *

glove_path = "/kaggle/input/glove6b100dtxt/glove.6B.100d.txt"

def preprocess_dataset(dataset):
    text, labels = create_datatset(dataset)
    tokenised_texts = [preprocess_text(t) for t in text]
    counter = Counter([tok for sent in tokenised_texts for tok in sent])
    vocab = ["<PAD>", "<UNK>"] + [w for w, c in counter.items() if c >= 2]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    print(f"Length of the vocabulary: {len(vocab)}") # 63518
    glove = process_glove(path=glove_path)
    
    embedding_dim = 100
    embedding_matrix = np.random.normal(0, 1, (len(vocab), embedding_dim))

    print(f"The dimensions of embedding matrix are : {embedding_matrix.shape}") # (63518, 100)

    for word, idx in word2idx.items():
        if word in glove:
            embedding_matrix[idx] = glove[word]

    indexed_seqs = [tokens_to_indices(seq, word2idx) for seq in tokenised_texts]
    padded_seqs = pad_sequence(indexed_seqs, batch_first=True, padding_value=word2idx["<PAD>"])
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(padded_seqs, labels_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    print(f"Training size = {train_size} and Testing size = {test_size}")

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    return train_loader, test_loader, vocab, word2idx, idx2word, embedding_matrix



