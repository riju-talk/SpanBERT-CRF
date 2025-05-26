SpanBERT-CRF for Question Answering
This repository implements a SpanBERT-based Question Answering model enhanced with a Conditional Random Field (CRF) layer for improved answer span prediction on the SQuAD v2.0 dataset.

🚀 Features

Fine-tuned SpanBERT for span-based QA
CRF head for enhanced boundary detection
Full SQuAD v2.0 support (including unanswerable questions)
Reproducible in Jupyter Notebook / Colab
Custom evaluation and prediction logic


🗂️ Project Structure
├── spanbert-crf.ipynb     # Main notebook (training + evaluation)
├── data/                  # Processed dataset (optional)
├── models/                # Saved weights and tokenizer
├── outputs/               # Logs and predictions
├── README.md              # Project documentation
└── requirements.txt       # Dependencies


🛠️ Setup Instructions

Clone the repository:
git clone https://github.com/yourusername/spanbert-crf.git
cd spanbert-crf


Install dependencies:
pip install -r requirements.txt


Run the notebook:

Open locally in Jupyter Notebook:jupyter notebook spanbert-crf.ipynb


Alternatively, upload spanbert-crf.ipynb to Google Colab and run it there.




📊 Evaluation Metrics

F1 Score
Exact Match (EM)
Answerability Accuracy


🤖 Inference Example
Here’s how to use the model for inference with a sample context and question:
context = "The quick brown fox jumps over the lazy dog."
question = "What does the fox jump over?"
answer = predict_answer(context, question)
print(answer)  # Output: "the lazy dog"


📌 TODO

Add hyperparameter tuning
Enable CLI or API inference
Deploy model via FastAPI
