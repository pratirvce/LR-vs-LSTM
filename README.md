# Logistic Regression vs LSTM for Sentiment Analysis

NLP 202 Assignment 1 - Comparative study of Logistic Regression and LSTM models on IMDB movie reviews dataset.

## 🎯 Project Overview

This project implements and compares two neural network architectures for binary sentiment classification:
- **Logistic Regression** with embeddings
- **LSTM (Long Short-Term Memory)** with embeddings

Both models use minibatching, proper padding/masking, and comprehensive hyperparameter tuning.

## 📊 Key Results

| Model | Test Accuracy | Val Accuracy | Best Batch Size | Best LR |
|-------|--------------|--------------|-----------------|---------|
| **Logistic Regression** | **88.14%** | 87.00% | 64 | 1e-3 |
| **LSTM** | 83.95% | 76.62% | 64 | 1e-3 |

**Key Finding:** The simpler Logistic Regression model outperformed LSTM by 4.19% on test accuracy!

## 📁 Repository Structure

```
├── assignment_solution.py          # Main implementation
├── error_analysis.py               # Error analysis and visualization
├── README.md                       # Detailed documentation
├── requirements.txt                # Dependencies
├── outputs/                        # Model predictions
│   ├── lr_valid_predictions.npy
│   ├── lr_test_predictions.npy
│   ├── lstm_valid_predictions.npy
│   └── lstm_test_predictions.npy
└── overleaf_report/               # LaTeX report
    ├── main.tex
    └── plots/                      # 8 visualization plots
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/pratirvce/LR-vs-LSTM.git
cd LR-vs-LSTM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Download IMDB Dataset
```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

### 4. Run Training
```bash
python assignment_solution.py
```

### 5. Run Error Analysis
```bash
python error_analysis.py
```

## 🔧 Implementation Features

### Data Processing
- **Tokenization:** spaCy `en_core_web_sm`
- **Vocabulary:** Top 25,000 words
- **Dataset:** IMDB 50K reviews (20K train, 5K val, 25K test)

### Logistic Regression Model
- Embedding layer (100-dim)
- Average pooling with masking
- Linear output layer
- BCEWithLogitsLoss

### LSTM Model
- Embedding layer (100-dim)
- LSTM layer (128 hidden units)
- pack_padded_sequence / pad_packed_sequence
- Average pooling with masking
- Dropout (0.5)
- Linear output layer

### Technical Highlights
✅ Proper padding using `pad_sequence()`  
✅ Masking to exclude padding from computations  
✅ LSTM packing/unpacking for efficiency  
✅ Model correctness verification  
✅ Comprehensive hyperparameter tuning  
✅ Batch sizes tested: [16, 32, 64, 128]  
✅ Learning rates tested: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

## 📈 Experimental Results

### Hyperparameter Tuning Results

**Batch Size Impact:**
- Larger batch sizes → Faster training
- Optimal: 64 for both models
- Best trade-off between speed and accuracy

**Learning Rate Impact:**
- 1e-3 optimal for both models
- Too low (1e-4): Slow convergence
- Too high (1e-2): Unstable training

### Model Comparison

**Logistic Regression:**
- ✅ Higher accuracy (88.14%)
- ✅ Faster training (~68s/epoch)
- ✅ Simpler architecture
- ✅ Better generalization

**LSTM:**
- 📊 Lower accuracy (83.95%)
- ⏱️ Slower training (~168s/epoch)
- 🔄 More complex architecture
- ⚠️ Potential overfitting on validation set

## 📊 Visualizations

The project includes 8 comprehensive plots:
1. Training time vs batch size (both models)
2. Accuracy vs batch size (both models)
3. Accuracy vs learning rate (both models)
4. Confusion matrices comparison
5. Metrics comparison bar chart

All plots are available in `overleaf_report/plots/`

## 🔍 Error Analysis

### Key Observations

**Logistic Regression Errors:**
- False negatives: Sarcasm, subtle negativity
- False positives: Mixed reviews, plot spoilers

**LSTM Errors:**
- More false negatives overall
- Struggles with: Long reviews, complex narratives
- Better at: Capturing sequential patterns (when correct)

See `error_analysis.py` for detailed examples and analysis.

## 📝 Report

A comprehensive LaTeX report is available in `overleaf_report/main.tex` covering:
- Model architectures
- Correctness verification
- Hyperparameter tuning experiments
- Performance comparison
- Error analysis
- Conclusions

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **spaCy** - NLP tokenization
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **scikit-learn** - Metrics
- **tqdm** - Progress bars

## 📄 License

This project was created as coursework for NLP 202.

## 👤 Author

Pratibha Revankar  
GitHub: [@pratirvce](https://github.com/pratirvce)

## 🙏 Acknowledgments

- IMDB dataset: [Maas et al., 2011](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)
- Course: NLP 202 - Natural Language Processing
- Assignment focus: Understanding minibatching in PyTorch

## 📚 References

1. Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL.
2. PyTorch Documentation: https://pytorch.org/docs/
3. spaCy Documentation: https://spacy.io/

---

⭐ If you find this project useful, please consider giving it a star!
