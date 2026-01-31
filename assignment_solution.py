"""
NLP 202 Assignment 1: Sentiment Analysis with Minibatching
Author: [Your Name]

This script implements:
1. Logistic Regression model with minibatching
2. LSTM model with minibatching
3. Model correctness testing
4. Hyperparameter tuning (batch size, learning rate)
5. Evaluation and analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import spacy
from collections import Counter
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import time
import json

# Set random seed for reproducibility
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Constants
PAD_IDX = 0
UNK_IDX = 1

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def tokenize(text, nlp):
    """Tokenize text using spaCy."""
    return [token.text.lower() for token in nlp(text)]

def build_vocab(texts, nlp, max_vocab_size=25_000):
    """Build vocabulary from texts."""
    counter = Counter(token for text in tqdm(texts, desc="Building vocab") for token in tokenize(text, nlp))
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_vocab_size))}
    vocab["<pad>"] = PAD_IDX
    vocab["<unk>"] = UNK_IDX
    return vocab

def numericalize(texts, vocab, nlp):
    """Convert texts to sequences of indices."""
    return [[vocab.get(token, UNK_IDX) for token in tokenize(text, nlp)] for text in tqdm(texts, desc="Numericalizing")]

def load_imdb_data(data_dir):
    """Load IMDB dataset from directory."""
    texts, labels = [], []
    for label_type in ["pos", "neg"]:
        folder = f"{data_dir}/{label_type}"
        for file in os.listdir(folder):
            if file.endswith('.txt'):
                with open(f"{folder}/{file}", "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(1 if label_type == "pos" else 0)
    return texts, labels

# ============================================================================
# DATASET AND DATALOADER
# ============================================================================

class IMDBDataset(Dataset):
    """IMDB Dataset for PyTorch."""
    
    def __init__(self, texts, labels, vocab, nlp):
        self.texts = numericalize(texts, vocab, nlp)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    """Custom collate function for padding sequences."""
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(labels)
    return padded_texts, labels, lengths

# ============================================================================
# MODELS
# ============================================================================

class LogisticRegressionModel(nn.Module):
    """Logistic Regression with embedding layer for sentiment analysis."""
    
    def __init__(self, vocab_size, embed_dim, padding_idx=PAD_IDX):
        super(LogisticRegressionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, lengths=None):
        """
        Forward pass.
        Args:
            x: [batch_size, seq_len] - padded input sequences
            lengths: [batch_size] - original lengths before padding
        """
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Average pooling - need to mask out padding
        if lengths is not None:
            # Create mask for padding
            mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2).float()  # [batch_size, seq_len, 1]
            
            # Apply mask and compute average
            masked_embedded = embedded * mask
            pooled = masked_embedded.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            pooled = embedded.mean(dim=1)
        
        return self.fc(pooled).squeeze(1)

class LSTMModel(nn.Module):
    """LSTM model for sentiment analysis."""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.5, padding_idx=PAD_IDX):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        """
        Forward pass with packing/unpacking.
        Args:
            x: [batch_size, seq_len] - padded input sequences
            lengths: [batch_size] - original lengths before padding
        """
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Pack the padded sequence
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), 
                                               batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack the sequence
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        # output: [batch_size, seq_len, hidden_dim]
        
        # Average pooling over the sequence (excluding padding)
        mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2).float()  # [batch_size, seq_len, 1]
        
        masked_output = output * mask
        pooled = masked_output.sum(dim=1) / lengths.unsqueeze(1).float()
        
        # Apply dropout and final linear layer
        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(1)

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    for texts, labels, lengths in tqdm(dataloader, desc="Training"):
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    epoch_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            
            predictions = model(texts, lengths)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            all_preds.extend(torch.round(torch.sigmoid(predictions)).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss / len(dataloader), accuracy, all_preds, all_labels

def train_model(model, train_loader, valid_loader, optimizer, criterion, device, n_epochs=10):
    """Train model for multiple epochs."""
    train_losses, valid_losses, valid_accs = [], [], []
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.4f}, "
              f"Valid Loss = {valid_loss:.4f}, Valid Acc = {valid_acc:.4f}")
    
    return train_losses, valid_losses, valid_accs

# ============================================================================
# MODEL CORRECTNESS TESTING
# ============================================================================

def test_model_correctness(model_class, vocab_size, embed_dim, device, test_samples, test_labels, 
                          hidden_dim=None, num_layers=None, dropout=0.5):
    """
    Test if minibatched model produces same results as single-instance processing.
    """
    print("\n" + "="*60)
    print("MODEL CORRECTNESS TESTING")
    print("="*60)
    
    # Create model and set to eval mode with fixed seed
    torch.manual_seed(SEED)
    if model_class == LSTMModel:
        model = model_class(vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
    else:
        model = model_class(vocab_size, embed_dim).to(device)
    model.eval()
    
    # Single instance processing
    single_losses = []
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    with torch.no_grad():
        for i, (text, label) in enumerate(zip(test_samples, test_labels)):
            text_tensor = torch.tensor(text, dtype=torch.long).unsqueeze(0).to(device)
            label_tensor = torch.tensor([label], dtype=torch.float).to(device)
            length = torch.tensor([len(text)]).to(device)
            
            pred = model(text_tensor, length)
            loss = criterion(pred, label_tensor)
            single_losses.append(loss.item())
    
    # Batched processing
    torch.manual_seed(SEED)
    if model_class == LSTMModel:
        model = model_class(vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
    else:
        model = model_class(vocab_size, embed_dim).to(device)
    model.eval()
    
    # Create batch
    texts_tensors = [torch.tensor(text, dtype=torch.long) for text in test_samples]
    labels_tensor = torch.tensor(test_labels, dtype=torch.float).to(device)
    lengths = torch.tensor([len(text) for text in test_samples]).to(device)
    padded_texts = pad_sequence(texts_tensors, batch_first=True, padding_value=PAD_IDX).to(device)
    
    with torch.no_grad():
        batch_preds = model(padded_texts, lengths)
        batch_losses = criterion(batch_preds, labels_tensor)
    
    # Compare results
    print(f"\nComparing {len(test_samples)} test samples:")
    print(f"{'Instance':<10} {'Single Loss':<15} {'Batch Loss':<15} {'Difference':<15}")
    print("-" * 60)
    
    max_diff = 0
    for i, (single, batch) in enumerate(zip(single_losses, batch_losses.cpu().numpy())):
        diff = abs(single - batch)
        max_diff = max(max_diff, diff)
        print(f"{i:<10} {single:<15.10f} {batch:<15.10f} {diff:<15.10e}")
    
    print(f"\nMaximum difference: {max_diff:.10e}")
    
    if max_diff < 1e-6:
        print("✓ PASSED: Minibatched and single-instance outputs are essentially identical!")
        return True
    else:
        print("✗ WARNING: Outputs differ. Check masking implementation.")
        return False

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_batch_size(model_class, train_dataset, valid_dataset, vocab_size, embed_dim, device,
                    batch_sizes, learning_rate=1e-3, n_epochs=5, hidden_dim=128, 
                    num_layers=1, dropout=0.5):
    """
    Tune batch size and record training time and accuracy.
    """
    results = {
        'batch_sizes': batch_sizes,
        'train_times': [],
        'valid_accs': [],
        'train_losses': [],
        'valid_losses': []
    }
    
    criterion = nn.BCEWithLogitsLoss()
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Training with batch_size={batch_size}")
        print(f"{'='*60}")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        # Initialize model
        if model_class == LSTMModel:
            model = model_class(vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
        else:
            model = model_class(vocab_size, embed_dim).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train and time
        start_time = time.time()
        train_losses, valid_losses, valid_accs = train_model(
            model, train_loader, valid_loader, optimizer, criterion, device, n_epochs
        )
        train_time = time.time() - start_time
        
        # Store results
        results['train_times'].append(train_time)
        results['valid_accs'].append(max(valid_accs))
        results['train_losses'].append(train_losses[-1])
        results['valid_losses'].append(valid_losses[-1])
        
        print(f"\nBatch size {batch_size}: Time={train_time:.2f}s, Best Valid Acc={max(valid_accs):.4f}")
    
    return results

def tune_learning_rate(model_class, train_dataset, valid_dataset, vocab_size, embed_dim, device,
                       learning_rates, batch_size=32, n_epochs=10, hidden_dim=128, 
                       num_layers=1, dropout=0.5):
    """
    Tune learning rate for a fixed batch size.
    """
    results = {
        'learning_rates': learning_rates,
        'valid_accs': [],
        'train_losses': [],
        'valid_losses': []
    }
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    for lr in learning_rates:
        print(f"\n{'='*60}")
        print(f"Training with learning_rate={lr}")
        print(f"{'='*60}")
        
        # Initialize model
        if model_class == LSTMModel:
            model = model_class(vocab_size, embed_dim, hidden_dim, num_layers, dropout).to(device)
        else:
            model = model_class(vocab_size, embed_dim).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train
        train_losses, valid_losses, valid_accs = train_model(
            model, train_loader, valid_loader, optimizer, criterion, device, n_epochs
        )
        
        # Store results
        results['valid_accs'].append(max(valid_accs))
        results['train_losses'].append(train_losses[-1])
        results['valid_losses'].append(valid_losses[-1])
        
        print(f"\nLearning rate {lr}: Best Valid Acc={max(valid_accs):.4f}")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_batch_size_results(results, model_name, save_dir='plots'):
    """Plot batch size tuning results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Training time vs batch size
    plt.figure(figsize=(10, 6))
    plt.plot(results['batch_sizes'], results['train_times'], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title(f'{model_name}: Training Time vs Batch Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name.lower()}_batch_size_time.png', dpi=300)
    plt.close()
    
    # Plot 2: Accuracy vs batch size
    plt.figure(figsize=(10, 6))
    plt.plot(results['batch_sizes'], results['valid_accs'], marker='o', linewidth=2, markersize=8, color='green')
    plt.xlabel('Batch Size', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title(f'{model_name}: Validation Accuracy vs Batch Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name.lower()}_batch_size_accuracy.png', dpi=300)
    plt.close()

def plot_learning_rate_results(results, model_name, save_dir='plots'):
    """Plot learning rate tuning results."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(results['learning_rates'], results['valid_accs'], marker='o', linewidth=2, markersize=8, color='red')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title(f'{model_name}: Validation Accuracy vs Learning Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{model_name.lower()}_learning_rate_accuracy.png', dpi=300)
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load spaCy
    print("\nLoading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    # Load data
    print("\nLoading IMDB dataset...")
    if not os.path.exists("./aclImdb"):
        print("Error: IMDB dataset not found. Please run the data download cells first.")
        print("Run: !wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
        print("Then: !tar -xzf aclImdb_v1.tar.gz")
        return
    
    train_texts, train_labels = load_imdb_data("./aclImdb/train")
    test_texts, test_labels = load_imdb_data("./aclImdb/test")
    
    # Split train into train/valid
    split_idx = 20000
    train_texts, valid_texts = train_texts[:split_idx], train_texts[split_idx:]
    train_labels, valid_labels = train_labels[:split_idx], train_labels[split_idx:]
    
    print(f"Train size: {len(train_texts)}")
    print(f"Valid size: {len(valid_texts)}")
    print(f"Test size: {len(test_texts)}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = build_vocab(train_texts, nlp, max_vocab_size=25000)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, nlp)
    valid_dataset = IMDBDataset(valid_texts, valid_labels, vocab, nlp)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, nlp)
    
    # ========================================================================
    # PART 1: LOGISTIC REGRESSION
    # ========================================================================
    print("\n" + "="*70)
    print("PART 1: LOGISTIC REGRESSION MODEL")
    print("="*70)
    
    # Model correctness testing
    print("\n1. Testing model correctness...")
    test_samples = [train_dataset.texts[i] for i in range(5)]
    test_labels_sample = [train_labels[i] for i in range(5)]
    
    test_model_correctness(
        LogisticRegressionModel, vocab_size, embed_dim=100, 
        device=device, test_samples=test_samples, test_labels=test_labels_sample
    )
    
    # Batch size tuning
    print("\n2. Tuning batch size...")
    batch_sizes = [16, 32, 64, 128]
    lr_batch_results = tune_batch_size(
        LogisticRegressionModel, train_dataset, valid_dataset, 
        vocab_size, embed_dim=100, device=device, batch_sizes=batch_sizes,
        learning_rate=1e-3, n_epochs=5
    )
    
    plot_batch_size_results(lr_batch_results, 'Logistic_Regression')
    
    # Learning rate tuning
    print("\n3. Tuning learning rate...")
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    lr_lr_results = tune_learning_rate(
        LogisticRegressionModel, train_dataset, valid_dataset,
        vocab_size, embed_dim=100, device=device, learning_rates=learning_rates,
        batch_size=32, n_epochs=10
    )
    
    plot_learning_rate_results(lr_lr_results, 'Logistic_Regression')
    
    # Train best model
    print("\n4. Training best Logistic Regression model...")
    best_lr_batch_size = 64
    best_lr_learning_rate = 1e-3
    
    train_loader = DataLoader(train_dataset, batch_size=best_lr_batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=best_lr_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=best_lr_batch_size, shuffle=False, collate_fn=collate_fn)
    
    best_lr_model = LogisticRegressionModel(vocab_size, embed_dim=100).to(device)
    optimizer = torch.optim.Adam(best_lr_model.parameters(), lr=best_lr_learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    train_model(best_lr_model, train_loader, valid_loader, optimizer, criterion, device, n_epochs=15)
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels_eval = evaluate(best_lr_model, test_loader, criterion, device)
    valid_loss, valid_acc, valid_preds, valid_labels_eval = evaluate(best_lr_model, valid_loader, criterion, device)
    
    print(f"\nBest Logistic Regression Model:")
    print(f"Valid Accuracy: {valid_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save predictions
    os.makedirs('outputs', exist_ok=True)
    np.save('outputs/lr_valid_predictions.npy', np.array(valid_preds))
    np.save('outputs/lr_test_predictions.npy', np.array(test_preds))
    
    # ========================================================================
    # PART 2: LSTM MODEL
    # ========================================================================
    print("\n" + "="*70)
    print("PART 2: LSTM MODEL")
    print("="*70)
    
    # Model correctness testing
    print("\n1. Testing model correctness...")
    test_model_correctness(
        LSTMModel, vocab_size, embed_dim=100, device=device,
        test_samples=test_samples, test_labels=test_labels_sample,
        hidden_dim=128, num_layers=1, dropout=0.5
    )
    
    # Batch size tuning
    print("\n2. Tuning batch size...")
    lstm_batch_results = tune_batch_size(
        LSTMModel, train_dataset, valid_dataset,
        vocab_size, embed_dim=100, device=device, batch_sizes=batch_sizes,
        learning_rate=1e-3, n_epochs=5, hidden_dim=128, num_layers=1, dropout=0.5
    )
    
    plot_batch_size_results(lstm_batch_results, 'LSTM')
    
    # Learning rate tuning
    print("\n3. Tuning learning rate...")
    lstm_lr_results = tune_learning_rate(
        LSTMModel, train_dataset, valid_dataset,
        vocab_size, embed_dim=100, device=device, learning_rates=learning_rates,
        batch_size=32, n_epochs=10, hidden_dim=128, num_layers=1, dropout=0.5
    )
    
    plot_learning_rate_results(lstm_lr_results, 'LSTM')
    
    # Train best model
    print("\n4. Training best LSTM model...")
    best_lstm_batch_size = 64
    best_lstm_learning_rate = 1e-3
    
    train_loader = DataLoader(train_dataset, batch_size=best_lstm_batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=best_lstm_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=best_lstm_batch_size, shuffle=False, collate_fn=collate_fn)
    
    best_lstm_model = LSTMModel(vocab_size, embed_dim=100, hidden_dim=128, num_layers=1, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(best_lstm_model.parameters(), lr=best_lstm_learning_rate)
    
    train_model(best_lstm_model, train_loader, valid_loader, optimizer, criterion, device, n_epochs=15)
    
    # Evaluate on test set
    test_loss, test_acc, test_preds, test_labels_eval = evaluate(best_lstm_model, test_loader, criterion, device)
    valid_loss, valid_acc, valid_preds, valid_labels_eval = evaluate(best_lstm_model, valid_loader, criterion, device)
    
    print(f"\nBest LSTM Model:")
    print(f"Valid Accuracy: {valid_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save predictions
    np.save('outputs/lstm_valid_predictions.npy', np.array(valid_preds))
    np.save('outputs/lstm_test_predictions.npy', np.array(test_preds))
    
    # Save models
    torch.save(best_lr_model.state_dict(), 'outputs/best_lr_model.pt')
    torch.save(best_lstm_model.state_dict(), 'outputs/best_lstm_model.pt')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("- Plots: ./plots/")
    print("- Predictions: ./outputs/")
    print("- Models: ./outputs/")

if __name__ == "__main__":
    main()
