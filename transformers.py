pip install torch nltk tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt')

# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Tokenizer and Vocabulary creation
def tokenize(text):
    return text.lower().split()

def build_vocab(sentences, max_size=10000):
    word_counts = {}
    for sentence in sentences:
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_size-2]
    
    word2idx = {word: idx + 2 for idx, (word, _) in enumerate(sorted_words)}
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len=30):
        self.src_data = self.load_file(src_file)
        self.tgt_data = self.load_file(tgt_file)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def load_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [tokenize(line.strip()) for line in file]

    def __len__(self):
        return len(self.src_data)

    def pad_sequence(self, seq, vocab):
        seq = [vocab.get(word, vocab['<UNK>']) for word in seq]
        return seq[:self.max_len] + [vocab['<PAD>']] * (self.max_len - len(seq))

    def __getitem__(self, idx):
        src_seq = self.pad_sequence(self.src_data[idx], self.src_vocab)
        tgt_seq = self.pad_sequence(self.tgt_data[idx], self.tgt_vocab)
        return torch.tensor(src_seq), torch.tensor(tgt_seq)

# Load train, val, test data and create vocab
train_src = tokenize(open('/kaggle/input/english-to-french/train.en').read())
train_tgt = tokenize(open('/kaggle/input/english-to-french/train.fr').read())

src_vocab, src_idx2word = build_vocab(train_src)
tgt_vocab, tgt_idx2word = build_vocab(train_tgt)

train_dataset = TranslationDataset('/kaggle/input/english-to-french/train.en', '/kaggle/input/english-to-french/train.fr', src_vocab, tgt_vocab)
val_dataset = TranslationDataset('/kaggle/input/english-to-french/dev.en', '/kaggle/input/english-to-french/dev.fr', src_vocab, tgt_vocab)
test_dataset = TranslationDataset('/kaggle/input/english-to-french/test.en', '/kaggle/input/english-to-french/test.fr', src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        # Ensure that the positional encoding is on the same device as input tensor x
        return x + self.pe[:x.size(0), :].to(x.device)

    def scaled_dot_product_attention(query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)
        return output, attn
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        query = split_heads(self.query_linear(query))
        key = split_heads(self.key_linear(key))
        value = split_heads(self.value_linear(value))
        
        attn_output, _ = scaled_dot_product_attention(query, key, value, mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(attn_output)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(torch.nn.functional.relu(self.linear1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + attn_output)
        enc_attn_output = self.enc_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + enc_attn_output)
        ff_output = self.ff(x)
        return self.norm3(x + ff_output)
    
    
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
    
    def forward(self, src, src_mask):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
    
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.embedding(tgt)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=100, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.fc_out(self.dropout(dec_output))
        return self.fc_out(dec_output)
def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    # Padding mask for the source
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    # Padding mask for the target (excluding <PAD> tokens) and autoregressive masking
    tgt_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(2)
    nopeak_mask = torch.triu(torch.ones((1, tgt.size(1), tgt.size(1)), device=tgt.device), diagonal=1).bool()
    
    tgt_mask = tgt_mask & ~nopeak_mask  # Combine padding and no-peak masks for the target
    return src_mask, tgt_mask


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)

# Instantiate and train model
num_epochs=6

def train_model(model, train_loader, val_loader, num_epochs=6):
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Training loop
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask, tgt_mask = create_masks(src, tgt_input, src_vocab['<PAD>'], tgt_vocab['<PAD>'])
            
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)

            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss*10 / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask, tgt_mask = create_masks(src, tgt_input, src_vocab['<PAD>'], tgt_vocab['<PAD>'])

                output = model(src, tgt_input, src_mask, tgt_mask)
                val_loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss*10 / len(val_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

        scheduler.step(avg_val_loss)
        
model = Transformer(len(src_vocab), len(tgt_vocab)).to(device)
model.apply(init_weights)
train_model(model, train_loader, val_loader)
model_save_path = "transformer.pt"
torch.save(model.state_dict(), model_save_path)


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# Evaluation function for BLEU score
def evaluate_bleu(model, data_loader, idx2word, tgt_vocab):
    model.eval()
    smooth_fn = SmoothingFunction().method4
    total_bleu_score = 0
    num_sentences = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            src_mask, tgt_mask = create_masks(src, tgt_input, src_vocab['<PAD>'], tgt_vocab['<PAD>'])

            output = model(src, tgt_input, src_mask, tgt_mask)
            output_indices = output.argmax(dim=-1).cpu().numpy()

            for i in range(output_indices.shape[0]):
                # Convert output indices to words
                pred_sentence = [idx2word[idx] for idx in output_indices[i] if idx != tgt_vocab['<PAD>']]
                target_sentence = [idx2word[idx.item()] for idx in tgt[i, 1:] if idx != tgt_vocab['<PAD>']]

                # Calculate BLEU score for this sentence
                bleu_score = sentence_bleu([target_sentence], pred_sentence, smoothing_function=smooth_fn)
                total_bleu_score += bleu_score
                num_sentences += 1

    avg_bleu_score = total_bleu_score / num_sentences
    print(f'Test BLEU score: {avg_bleu_score*100:.4f}')
    return avg_bleu_score

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Hyperparameter configurations for tuning
hyperparameter_configs = [
    {'num_layers': 4, 'num_heads': 4, 'd_model': 256, 'dropout': 0.1},
    {'num_layers': 6, 'num_heads': 8, 'd_model': 512, 'dropout': 0.3},
    {'num_layers': 8, 'num_heads': 8, 'd_model': 512, 'dropout': 0.4}
]

def train_and_evaluate(hyperparams):
    # Extract hyperparameters
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    d_model = hyperparams['d_model']
    dropout = hyperparams['dropout']

    # Initialize the model with the current hyperparameters
    model = Transformer(len(src_vocab), len(tgt_vocab), d_model=d_model, num_heads=num_heads, num_layers=num_layers, dropout=dropout).to(device)
    model.apply(init_weights)
    
    # Define criterion and optimizer for this model
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    train_losses, val_losses = [], []
    for epoch in range(3):
        model.train()
        total_train_loss = 0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask, tgt_mask = create_masks(src, tgt_input, src_vocab['<PAD>'], tgt_vocab['<PAD>'])
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask, tgt_mask = create_masks(src, tgt_input, src_vocab['<PAD>'], tgt_vocab['<PAD>'])
                output = model(src, tgt_input, src_mask, tgt_mask)
                val_loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss*10 / len(val_loader)
        val_losses.append(avg_val_loss)

    # Evaluate BLEU score on test data
    test_bleu_score = evaluate_bleu(model, test_loader, tgt_idx2word, tgt_vocab)
    return train_losses, val_losses, test_bleu_score

# Store results
results = []

# Train and evaluate for each hyperparameter configuration
for i, config in enumerate(hyperparameter_configs):
    print(f"Testing Configuration {i + 1}: {config}")
    train_losses, val_losses, test_bleu_score = train_and_evaluate(config)
    results.append({'config': config, 'train_losses': train_losses, 'val_losses': val_losses, 'test_bleu_score': test_bleu_score})

# Plot training and validation loss graphs
for i, result in enumerate(results):
    config = result['config']
    plt.figure(figsize=(10, 5))
    plt.plot(result['train_losses'], label='Train Loss')
    plt.plot(result['val_losses'], label='Validation Loss')
    plt.title(f"Training and Validation Losses (Config {i + 1}) - Layers: {config['num_layers']}, Heads: {config['num_heads']}, d_model: {config['d_model']}, Dropout: {config['dropout']}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Display test BLEU scores for each configuration
for i, result in enumerate(results):
    config = result['config']
    print(f"Configuration {i + 1} - Layers: {config['num_layers']}, Heads: {config['num_heads']}, d_model: {config['d_model']}, Dropout: {config['dropout']}")
    print(f"Test BLEU Score: {result['test_bleu_score']:.4f}")

def evaluate_and_save_bleu_scores(model, data_loader, idx2word, tgt_vocab, output_file='testbleu.txt'):
    model.eval()
    smooth_fn = SmoothingFunction().method4
    sentence_bleu_scores = []

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt_input = tgt[:, :-1].to(device)
            src_mask, tgt_mask = create_masks(src, tgt_input, src_vocab['<PAD>'], tgt_vocab['<PAD>'])

            output = model(src, tgt_input, src_mask, tgt_mask)
            output_indices = output.argmax(dim=-1).cpu().numpy()

            for i in range(output_indices.shape[0]):
                # Convert output indices to words
                pred_sentence = [idx2word[idx] for idx in output_indices[i] if idx != tgt_vocab['<PAD>']]
                target_sentence = [idx2word[idx.item()] for idx in tgt[i, 1:] if idx != tgt_vocab['<PAD>']]

                # Calculate BLEU score for this sentence
                bleu_score = sentence_bleu([target_sentence], pred_sentence, smoothing_function=smooth_fn)
                sentence_bleu_scores.append((pred_sentence, bleu_score))

    # Write the BLEU scores to the file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, score in sentence_bleu_scores:
            sentence_str = ' '.join(sentence)
            f.write(f"{sentence_str} {score:.4f}\n")

# Example usage:
# Assuming the model has already been trained using one of the hyperparameter configurations
# Replace 'results[0]' with the correct model configuration if needed
best_model_config = hyperparameter_configs[0]
model = Transformer(len(src_vocab), len(tgt_vocab), d_model=best_model_config['d_model'], num_heads=best_model_config['num_heads'], num_layers=best_model_config['num_layers'], dropout=best_model_config['dropout']).to(device)
model.apply(init_weights)

# Assuming the model has been trained, now we evaluate and save the BLEU scores
evaluate_and_save_bleu_scores(model, test_loader, tgt_idx2word, tgt_vocab)
