# 🏦 Consumer Complaint Classifier

### Deep Learning NLP System for Financial Complaint Classification

### Consumer Complaint Analysis & Explainable NLP System

🔗 **Live Demo:** https://consumer-complaint-classifier.streamlit.app/

---

# 🏦 Consumer Complaint Classifier: Deep Learning NLP System

   

### 🔗 [https://consumer-complaint-classifier.streamlit.app/](#)

---

## 📌 Overview

The **Consumer Complaint Classifier** is an end-to-end Natural Language Processing (NLP) deep learning system designed for the financial sector. It automatically ingests raw text from consumer complaints and categorizes them into specific financial departments such as:

* Credit Cards
* Mortgages
* Student Loans
* Bank Accounts
* Debt Collection
* Personal Loans

using advanced neural network architectures.

Beyond simple classification, this system prioritizes **compliance and transparency** by integrating an Explainable AI (XAI) module that highlights the exact words driving the model's decisions, along with an automated PDF reporting engine for compliance auditing.

---

## 🚀 Key Features

## ✅ Deep Learning Architectures

The project evaluates and compares multiple deep learning models:

* **BiLSTM (Bidirectional LSTM)**
* **LSTM**
* **CNN for Text Classification**

This helps identify the best-performing architecture for NLP sequence classification.

---

## ✅ Semantic Word Embeddings

Uses pre-trained:

* **GloVe (Global Vectors for Word Representation)**

These embeddings provide semantic understanding of:

* English language structure
* Financial terminology
* Contextual word relationships

---

## ✅ Explainable AI (XAI)

Implements a custom:

* **Occlusion Sensitivity-based Token Attention Visualizer**

The system dynamically highlights the exact words responsible for the prediction.

Example:

> "My credit card payment was charged twice"

The model may highlight:

* *credit card*
* *charged twice*

as the most influential tokens.

---

## ✅ Automated PDF Reporting

The application generates downloadable:

* Prediction Reports
* Confidence Scores
* Attention Visualization Reports
* Model Performance Summaries

using:

* `reportlab`
* `Pillow (PIL)`

---

## ✅ Production-Ready Streamlit Dashboard

Features a modern multi-page UI built using:

* Streamlit
* Custom CSS Styling
* Responsive Layouts
* Interactive Visualizations

---

## 🧠 System Architecture

## 1️⃣ Data Ingestion & Transformation

Raw complaint data undergoes:

* Text cleaning
* Lowercasing
* Stopword removal
* Tokenization
* Sequence padding
* Embedding mapping using GloVe 100d vectors

This converts textual data into numerical tensors suitable for deep learning.

---

## 2️⃣ Model Training

Neural networks are trained using:

* TensorFlow
* Keras

Optimization Objective:

* **Categorical Crossentropy**

Additional training techniques:

* Early Stopping
* Dropout Regularization
* Validation Monitoring

---

## 3️⃣ Evaluation Pipeline

The evaluation system generates:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Reports

Results are saved as:

* JSON files
* PNG visualizations
* Trained artifacts

---

## 4️⃣ Real-Time Inference

The Streamlit frontend:

1. Accepts user complaint text
2. Loads trained `.h5` models
3. Runs prediction inference
4. Generates XAI explanations
5. Displays confidence scores

---

## 5️⃣ PDF Report Generation

Prediction outputs are converted into professional PDF reports containing:

* Complaint Text
* Predicted Category
* Confidence Probability
* Attention Visualization
* Timestamp
* Model Used

---

## 💻 Tech Stack

| Category                   | Technologies              |
| -------------------------- | ------------------------- |
| Programming Language       | Python 3.10+              |
| Deep Learning              | TensorFlow, Keras         |
| NLP                        | GloVe, Keras Tokenizer    |
| Frontend                   | Streamlit                 |
| Data Processing            | Pandas, NumPy             |
| Machine Learning Utilities | Scikit-learn              |
| Visualization              | Matplotlib, Seaborn       |
| Reporting                  | ReportLab, Pillow         |
| Deployment                 | Streamlit Community Cloud |

---

## 📂 Project Workflow

```text
User Complaint
       ↓
Text Preprocessing
       ↓
Tokenization & Padding
       ↓
GloVe Embedding Mapping
       ↓
Deep Learning Model
       ↓
Prediction + Confidence Score
       ↓
XAI Token Highlighting
       ↓
PDF Report Generation
       ↓
Streamlit Dashboard Output
```

---

## 🛠️ Local Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/YourUsername/Consumer-Complaint-Classifier.git
cd Consumer-Complaint-Classifier
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Download GloVe Embeddings

Download:

* `glove.6B.100d.txt`

from Stanford NLP and place it inside the root project directory.

Reason:

* Large file size
* Excluded via `.gitignore`

---

## 5️⃣ Run the Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

---

## 📁 Repository Structure

```text
📦 Consumer-Complaint-Classifier
 ┣ 📂 artifacts/
 ┃ ┣ 📜 lstm_model.h5
 ┃ ┣ 📜 bilstm_model.h5
 ┃ ┣ 📜 cnn_model.h5
 ┃ ┣ 📜 tokenizer.pkl
 ┃ ┗ 📜 evaluation_report.json
 ┃ ┗ 📜 glove_matrix.npy
 ┃ ┗ 📜 label_encoder.json
 ┃
 ┣ 📂 src/
 ┃ ┣ 📂 components/
 ┃ ┃ ┣ 📜 data_ingestion.py
 ┃ ┃ ┣ 📜 data_transformation.py
 ┃ ┃ ┣ 📜 model_trainer.py
 ┃ ┃ ┣ 📜 model_evaluation.py
 ┃ ┃ ┣ 📜 attention_extractor.py
 ┃ ┃ ┗ 📜 report_generator.py
 ┃ ┃
 ┃ ┣ 📂 pipeline/
 ┃ ┃ ┣ 📜 train_pipeline.py
 ┃ ┃ ┗ 📜 predict_pipeline.py
 ┃ ┃ ┗ 📜 report_pipeline.py
 ┃ ┃
 ┃ ┣ 📜 logger.py
 ┃ ┣ 📜 exception.py
 ┃ ┗ 📜 utils.py
 ┃
 ┣ 📂 streamlit_app/
 ┃ ┣ 📂 components/
 ┃ ┃ ┣ 📜 attention_heatmap.py
 ┃ ┃ ┗ 📜 confidence_bar.py
 ┃ ┃ ┗ 📜 ui_tweaks.py
 ┃ ┃
 ┃ ┣ 📂 pages/
 ┃ ┃ ┣ 📜 1_classify.py
 ┃ ┃ ┣ 📜 2_results.py
 ┃ ┃ ┗ 📜 3_model_Comparison.py
 ┃ ┃
 ┃ ┗ 📜 app.py
 ┃
 ┣ 📜 requirements.txt
 ┣ 📜 setup.py
 ┣ 📜 README.md
 ┗ 📜 .gitignore
```

---

## 📊 Model Evaluation Metrics

The project compares multiple models using:

* Accuracy
* Precision
* Recall
* F1-Score

Typical Observations:

| Model  | Strength                   |
| ------ | -------------------------- |
| CNN    | Faster Training            |
| LSTM   | Good Sequential Learning   |
| BiLSTM | Best Context Understanding |

BiLSTM generally performs best due to:

* Bidirectional contextual learning
* Better semantic understanding
* Improved sequence retention

---

## 🔍 Explainable AI (XAI)

Traditional deep learning models behave like black boxes.

To improve transparency, this project includes:

## Occlusion Sensitivity Analysis

The process:

1. Removes one word at a time
2. Re-runs prediction
3. Measures confidence drop
4. Highlights important tokens

This helps:

* Compliance teams
* Auditors
* Analysts
* End users

understand *why* a prediction was made.

---

## 📄 PDF Reporting System

Generated reports include:

* Complaint Text
* Predicted Label
* Prediction Confidence
* Model Used
* Attention Highlights
* Timestamp
* Visualization Charts

Useful for:

* Compliance Documentation
* Audit Trails
* Enterprise Reporting
* Customer Support Analysis

---

## 🌐 Deployment

The application is deployed using:

* Streamlit Community Cloud

### Deployment Notes

To avoid compatibility issues:

* TensorFlow version is strictly pinned
* Keras version is pinned
* Model artifacts are pushed to GitHub for inference only

---

## ⚠️ Important Note About TensorFlow Installation

If you encounter:

```bash
ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0
```

Check your Python version.

TensorFlow 2.15 does NOT currently support:

```text
Python 3.14
```

Recommended Python versions:

* Python 3.10
* Python 3.11

You can install Python 3.10 and create a new virtual environment.

---

## 🔮 Future Improvements

Possible future enhancements:

* Transformer-based Models (BERT)
* HuggingFace Integration
* Real-time API Deployment
* Docker Containerization
* CI/CD Pipelines
* Cloud Deployment (AWS/GCP/Azure)
* Multi-language Complaint Support
* Voice Complaint Classification

---

## 🤝 Contribution

Contributions are welcome.

You can contribute by:

* Improving model performance
* Enhancing UI/UX
* Adding new NLP techniques
* Improving explainability modules
* Optimizing deployment

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

## Nishit Kumar

B.Tech Student | Data Science & NLP Enthusiast

### Skills

* Machine Learning
* Deep Learning
* NLP
* Streamlit
* TensorFlow
* Python
* Data Science

---
