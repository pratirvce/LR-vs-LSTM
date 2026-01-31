"""
Error Analysis Script
Analyzes predictions and provides insights into model errors.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import spacy
import torch
from torch.utils.data import DataLoader
import os

def load_predictions(model_name='lr'):
    """Load predictions and ground truth labels."""
    valid_preds = np.load(f'outputs/{model_name}_valid_predictions.npy')
    
    # Load ground truth from dataset
    from assignment_solution import load_imdb_data
    train_texts, train_labels = load_imdb_data("./aclImdb/train")
    valid_labels = np.array(train_labels[20000:])
    
    return valid_preds, valid_labels

def analyze_errors(predictions, labels, texts=None):
    """Analyze prediction errors."""
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate metrics
    correct = predictions == labels
    accuracy = correct.sum() / len(labels)
    
    # Error types
    false_positives = ((predictions == 1) & (labels == 0)).sum()
    false_negatives = ((predictions == 0) & (labels == 1)).sum()
    true_positives = ((predictions == 1) & (labels == 1)).sum()
    true_negatives = ((predictions == 0) & (labels == 0)).sum()
    
    print("="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"Actual   Neg   {true_negatives:5d}  {false_positives:5d}")
    print(f"         Pos   {false_negatives:5d}  {true_positives:5d}")
    
    print(f"\nError Breakdown:")
    print(f"False Positives (predicted positive, actually negative): {false_positives}")
    print(f"False Negatives (predicted negative, actually positive): {false_negatives}")
    print(f"Total Errors: {false_positives + false_negatives}")
    
    # Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'errors': ~correct
    }

def analyze_error_examples(texts, predictions, labels, nlp, n_examples=10):
    """Analyze specific error examples."""
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    errors = predictions != labels
    error_indices = np.where(errors)[0]
    
    print(f"\n{'='*60}")
    print(f"EXAMPLE ERRORS (showing {min(n_examples, len(error_indices))} examples)")
    print(f"{'='*60}\n")
    
    # False positives
    false_pos_indices = np.where((predictions == 1) & (labels == 0))[0]
    print(f"FALSE POSITIVES (Model predicted positive, actually negative):")
    print("-"*60)
    for i, idx in enumerate(false_pos_indices[:n_examples//2]):
        text = texts[idx]
        words = [token.text.lower() for token in nlp(text)]
        print(f"\nExample {i+1}:")
        print(f"Text preview: {' '.join(words[:50])}...")
        print(f"Length: {len(words)} words")
    
    # False negatives
    false_neg_indices = np.where((predictions == 0) & (labels == 1))[0]
    print(f"\n\nFALSE NEGATIVES (Model predicted negative, actually positive):")
    print("-"*60)
    for i, idx in enumerate(false_neg_indices[:n_examples//2]):
        text = texts[idx]
        words = [token.text.lower() for token in nlp(text)]
        print(f"\nExample {i+1}:")
        print(f"Text preview: {' '.join(words[:50])}...")
        print(f"Length: {len(words)} words")

def analyze_length_distribution(texts, predictions, labels, nlp):
    """Analyze error rates by review length."""
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Get lengths
    lengths = []
    for text in texts:
        words = [token.text.lower() for token in nlp(text)]
        lengths.append(len(words))
    lengths = np.array(lengths)
    
    # Bin by length
    bins = [0, 100, 200, 300, 400, 500, float('inf')]
    bin_labels = ['0-100', '100-200', '200-300', '300-400', '400-500', '500+']
    
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS BY REVIEW LENGTH")
    print(f"{'='*60}\n")
    
    for i in range(len(bins)-1):
        mask = (lengths >= bins[i]) & (lengths < bins[i+1])
        if mask.sum() == 0:
            continue
        
        bin_preds = predictions[mask]
        bin_labels_data = labels[mask]
        bin_accuracy = (bin_preds == bin_labels_data).sum() / len(bin_preds)
        
        print(f"Length {bin_labels[i]}: {mask.sum():4d} reviews, Accuracy: {bin_accuracy:.4f}")

def plot_error_analysis(lr_results, lstm_results, save_dir='plots'):
    """Create comparison plots for error analysis."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Confusion matrix comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    models = [('Logistic Regression', lr_results), ('LSTM', lstm_results)]
    
    for idx, (name, results) in enumerate(models):
        ax = axes[idx]
        
        conf_matrix = np.array([
            [results['true_negatives'], results['false_positives']],
            [results['false_negatives'], results['true_positives']]
        ])
        
        im = ax.imshow(conf_matrix, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{name}\nAccuracy: {results["accuracy"]:.4f}', fontsize=12)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, conf_matrix[i, j],
                             ha="center", va="center", color="black", fontsize=14)
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300)
    plt.close()
    
    # Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    lr_vals = [lr_results[m] for m in metrics]
    lstm_vals = [lstm_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, lr_vals, width, label='Logistic Regression', alpha=0.8)
    ax.bar(x + width/2, lstm_vals, width, label='LSTM', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/metrics_comparison.png', dpi=300)
    plt.close()

def main():
    """Run error analysis."""
    
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    
    print("Loading data and predictions...")
    from assignment_solution import load_imdb_data
    train_texts, train_labels = load_imdb_data("./aclImdb/train")
    valid_texts = train_texts[20000:]
    valid_labels = np.array(train_labels[20000:])
    
    # Analyze Logistic Regression
    print("\n" + "="*70)
    print("LOGISTIC REGRESSION ERROR ANALYSIS")
    print("="*70)
    lr_preds, lr_labels = load_predictions('lr')
    lr_results = analyze_errors(lr_preds, lr_labels)
    analyze_error_examples(valid_texts, lr_preds, lr_labels, nlp, n_examples=10)
    analyze_length_distribution(valid_texts, lr_preds, lr_labels, nlp)
    
    # Analyze LSTM
    print("\n" + "="*70)
    print("LSTM ERROR ANALYSIS")
    print("="*70)
    lstm_preds, lstm_labels = load_predictions('lstm')
    lstm_results = analyze_errors(lstm_preds, lstm_labels)
    analyze_error_examples(valid_texts, lstm_preds, lstm_labels, nlp, n_examples=10)
    analyze_length_distribution(valid_texts, lstm_preds, lstm_labels, nlp)
    
    # Create comparison plots
    print("\n\nGenerating comparison plots...")
    plot_error_analysis(lr_results, lstm_results)
    
    print("\n" + "="*70)
    print("ERROR ANALYSIS COMPLETE!")
    print("="*70)
    print("\nPlots saved to: ./plots/")

if __name__ == "__main__":
    main()
