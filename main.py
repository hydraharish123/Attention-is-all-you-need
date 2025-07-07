from datasets import load_dataset
import torch
import torch.optim as optim
from preprocess import *
from attention.BahdanauAttention import *
from attention.LuongConcatAttention import *
from attention.LuongDotAttention import *
from attention.LuongGeneralAttention import *
from models.BiLSTM_Model import *
from models.BiRNN_Model import *
from models.LSTM_Model import *
from models.RNN_Model import *
from train import *

dataset = load_dataset("imdb")
train_loader, test_loader, vocab, word2idx, idx2word, embedding_matrix = preprocess_dataset(dataset)

vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 128
bi_hidden_dim = hidden_dim * 2
output_dim = 2
embedding_matrix = embedding_matrix  # torch.Tensor [vocab_size, 100]

# Attention instances for RNN / LSTM
bahdanau = BahdanauAttention(hidden_dim, hidden_dim, attention_dim=64)
luong_dot = LuongDotAttention()
luong_general = LuongGeneralAttention(hidden_dim)
luong_concat = LuongConcatAttention(hidden_dim, attention_dim=64)

# Attention instances for BiRNN / BiLSTM
bahdanau_bi = BahdanauAttention(bi_hidden_dim, bi_hidden_dim, attention_dim=64)
luong_dot_bi = LuongDotAttention()
luong_general_bi = LuongGeneralAttention(bi_hidden_dim)
luong_concat_bi = LuongConcatAttention(bi_hidden_dim, attention_dim=64)

model_variants = {
    # RNN
    'RNN_NoAttention': RNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=None),
    'RNN_Bahdanau': RNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=bahdanau),
    'RNN_Dot': RNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_dot),
    'RNN_General': RNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_general),
    'RNN_Concat': RNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_concat),

    # LSTM
    'LSTM_NoAttention': LSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=None),
    'LSTM_Bahdanau': LSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=bahdanau),
    'LSTM_Dot': LSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_dot),
    'LSTM_General': LSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_general),
    'LSTM_Concat': LSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_concat),

    # BiRNN
    'BiRNN_NoAttention': BiRNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=None),
    'BiRNN_Bahdanau': BiRNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=bahdanau_bi),
    'BiRNN_Dot': BiRNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_dot_bi),
    'BiRNN_General': BiRNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_general_bi),
    'BiRNN_Concat': BiRNN_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_concat_bi),

    # BiLSTM
    'BiLSTM_NoAttention': BiLSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=None),
    'BiLSTM_Bahdanau': BiLSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=bahdanau_bi),
    'BiLSTM_Dot': BiLSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_dot_bi),
    'BiLSTM_General': BiLSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_general_bi),
    'BiLSTM_Concat': BiLSTM_Model(vocab_size, embed_dim, hidden_dim, output_dim, embedding_matrix, attention=luong_concat_bi),
}


for model_name, model in model_variants.items():
    name, attention_type = model_name.split("_")
    print(f"------------{name}, {attention_type}------------")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=5, model_name=name, attention_name=attention_type)