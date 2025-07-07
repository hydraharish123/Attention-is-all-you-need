import re
import numpy as np
import random
import torch
import nltk
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize
nltk.download("punkt")
random.seed(42)

# 1. Loading dataset
def create_datatset(dataset):
    all_texts = dataset['train']['text'] + dataset['test']['text']
    all_labels = dataset['train']['label'] + dataset['test']['label']
    combined = list(zip(all_texts, all_labels))
    sampled = random.sample(combined, k=30000)
    sampled_texts, sampled_labels = zip(*sampled)
    return sampled_texts, sampled_labels

# 2. IMDBdataset class
class IMBDdataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
        
# 3. remove all special character and punctuation from the text and tokenize them
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return word_tokenize(text)

# 4. Extract glove embedding vectors
def process_glove(path):
    glove = {}
    with open(path, "r", encoding="utf-8") as f :
        for line in f:
            word_embedding = line.strip().split()
            word = word_embedding[0]
            embedding = np.array(word_embedding[1:], dtype=np.float32)
            glove[word] = embedding
    
    return glove

# 5. Convert tokens to indices
def tokens_to_indices(tokens, word2idx):
    return torch.tensor([word2idx.get(t, word2idx["<UNK>"]) for t in tokens])