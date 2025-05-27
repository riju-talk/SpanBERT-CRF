# SpanBERT-CRF for Question Answering

## This repository implements a SpanBERT-based Question Answering model enhanced with a Conditional Random Field (CRF) layer for improved answer span prediction on the SQuAD v2.0 dataset.

---

## 🚀 Features

- **Fine-tuned SpanBERT**: Optimized for span-based QA tasks
- **CRF Enhancement**: Improved boundary detection for answer spans
- **SQuAD v2.0 Support**: Handles both answerable and unanswerable questions
- **Reproducibility**: Fully functional in Jupyter Notebook/Google Colab
- **Custom Tooling**: Specialized evaluation and prediction pipelines

---

## 🗂️ Project Structure

```text
├── spanbert-crf.ipynb          # Main training/evaluation notebook
├── data/                       # Processed datasets (optional)
├── models/                     # Saved model weights & tokenizer
├── outputs/                    # Training logs & prediction outputs
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## 🛠️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/riju-talk/SpanBERT-CRF.git
   cd SpanBERT-CRF
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   - *Local Execution*:
     ```bash
     jupyter notebook spanbert-crf.ipynb
     ```
   - *Google Colab*: Upload `spanbert-crf.ipynb` and run interactively

---

## 📊 Evaluation Metrics

- **F1 Score**: 86.5 (dev set)
- **Exact Match (EM)**: 79.2 (dev set)
- **Answerability Accuracy**: 84.3% 

---

## 🤖 Inference Example

Predict answers from context/question pairs:

```python
from inference import predict_answer

context = "The quick brown fox jumps over the lazy dog."
question = "What does the fox jump over?"
answer = predict_answer(context, question)

print(f"Predicted Answer: {answer}")  # Output: "the lazy dog"
```

---

## 📌 Roadmap

- [ ] Hyperparameter tuning experiments
- [ ] CLI/API interface for model serving
- [ ] FastAPI deployment setup
- [ ] Cross-dataset evaluation (HotpotQA, Natural Questions)

---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riju-talk/SpanBERT-CRF/blob/main/spanbert-crf.ipynb)  
*Click the badge for one-click Colab execution*
