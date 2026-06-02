# Veritas AI - Fake News Intelligence Platform

## Overview

Veritas AI is a production-style Fake News Detection platform built using DistilBERT, PyTorch, Hugging Face Transformers, and Streamlit.

The system analyzes news articles, headlines, and URLs to classify content as Real News or Fake News while providing explainable AI insights, confidence scores, credibility assessment, sentiment analysis, and interactive visualizations.

---

## Features

### AI-Powered News Detection
- DistilBERT Transformer Model
- Real vs Fake classification
- Confidence scoring
- Probability distribution analysis

### Explainable AI
- Influential word attribution
- Model explanation panel
- Highlighted text review

### Trust Analytics
- Trust Score Gauge
- Risk Assessment
- Credibility Score

### News URL Analysis
- Extract article content from URLs
- Automated content analysis
- End-to-end prediction pipeline

### Batch News Analysis
- CSV Upload Support
- Analyze multiple headlines simultaneously
- Exportable results
- Analytics dashboard

### Interactive Visualizations
- Confidence Metrics
- Probability Charts
- Sentiment Analysis
- Prediction History

### Reporting
- PDF Report Generation
- Recruiter-friendly dashboard
- Enterprise-style interface

---

## Tech Stack

### Machine Learning
- DistilBERT
- PyTorch
- Hugging Face Transformers

### Backend
- Python

### Frontend
- Streamlit

### Data Processing
- Pandas
- NumPy

### Visualization
- Plotly

---

## Dataset

Dataset used:

Fake and Real News Dataset

Total Articles: 44,898

Training Samples: 35,918

Testing Samples: 8,980

---

## Model Performance

| Metric | Score |
|----------|----------|
| Accuracy | 99.97% |
| Precision | 99.98% |
| Recall | 99.95% |
| F1 Score | 99.96% |

---

## Project Architecture

User Input
↓
Tokenizer
↓
DistilBERT
↓
Classification Layer
↓
Prediction
↓
Explainable AI Layer
↓
Dashboard Analytics

---

## Installation

Clone repository

```bash
git clone https://github.com/kashishcrafts/Veritas-AI-Fake-News-Detector.git
```

Move into project folder

```bash
cd Veritas-AI-Fake-News-Detector
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
streamlit run app.py
```

---

## Future Improvements

- Real-time News Monitoring
- Multi-language Detection
- Deepfake Detection Integration
- News Source Ranking System
- Fact Verification API
- Advanced Explainable AI

---

## Author

Kashish Kalim Shaikh

Computer Engineering Student

AI • Machine Learning • Data Engineering • Cybersecurity

---

## License

MIT License
