# LSTM Sentiment Analysis (PyTorch)

A complete, production-friendly **sentiment analysis** pipeline built around a stacked **LSTM**.  
This project includes robust tokenization, vocabulary building, padding/masking, class-imbalance handling, and a clean training loop with early stopping and LR scheduling.

---

## 🔍 Highlights
- **Custom tokenization & vocab** (torchtext/nltk) with OOV handling
- **Embedding layer** (random or pretrained vectors if provided)
- **Stacked LSTM** (optionally bidirectional) + dropout regularization
- **Packed sequences** for efficient variable-length batching
- **Class weights / focal loss** option for imbalance
- **Thorough evaluation**: Accuracy, Precision/Recall/F1, ROC/PR curves, confusion matrix

---

## 🏗️ Model
- Embedding(d_model=EMB_DIM)
- LSTM: 2–3 layers, hidden size = H, dropout = 0.3–0.5
- Bidirectional (optional)
- Classifier head: Linear → Softmax

- ### Loss: CrossEntropy (or focal)
- ### Optimizer: Adam (lr=1e-3 default)
- ### Scheduler: ReduceLROnPlateau

## 🧾 requirements
```text
torch==2.4.1
torchtext==0.19.1
numpy==2.1.3
pandas==2.2.3
matplotlib==3.9.3
seaborn==0.13.2
scikit-learn==1.5.2
tqdm==4.66.5
nltk==3.9.1
```

## 📌 Insights
- Bidirectional LSTM improves recall on minority classes
- Packed sequences + masking stabilize training
- Moderate dropout (0.3–0.5) and LR scheduling prevent overfitting

## 📦 Weights
If you prefer to reuse my trained model, weights are attached to the repo.
