# 💳 FinTech Fraud Detection System — End-to-End MLOps Project

A production-style fraud detection system that simulates how modern fintech companies detect suspicious transactions in real time using behavioral, velocity, and anomaly features.

This project demonstrates the complete MLOps lifecycle: data ingestion → feature engineering → model training → API serving → UI → containerization → CI/CD → monitoring.

---

## 🚀 Key Highlights

* Realistic **FinTech fraud detection pipeline**
* Behavioral + velocity feature engineering
* Imbalanced data handling with proper metrics
* FastAPI inference service
* Streamlit dashboard connected to backend
* Docker multi-service architecture
* CI pipeline with GitHub Actions
* Prediction logging for monitoring & retraining

---

## 🧠 Problem Statement

Financial institutions must detect fraudulent transactions with **high recall** while minimizing false positives.

This system predicts fraud probability for each transaction and assigns a **risk level** to support real-time decision making.

---

## 📊 Dataset

Credit Card Fraud Detection dataset (ULB)

Due to GitHub size limits, data is not included.

Download from Kaggle and place as:

```
data/raw.csv
```

---

## 🏗️ Architecture

```
Raw Data
   ↓
Ingestion Pipeline
   ↓
Feature Engineering (behavior + velocity)
   ↓
Model Training & Evaluation
   ↓
Artifact Registry
   ↓
FastAPI Fraud Detection Service
   ↓
Streamlit Dashboard (UI)
   ↓
Logging & Monitoring
```

---

## ⚙️ Feature Engineering (FinTech Twist)

The system creates realistic fraud signals:

* Transaction velocity features
* Behavioral spending baseline
* Night-time anomaly indicator
* Deviation from recent spending pattern
* Log-scaled transaction amount

These features mimic production fraud systems.

---

## 🤖 Model

* Logistic Regression with class imbalance handling
* Stratified split
* ROC-AUC, Precision, Recall evaluation

Artifacts saved:

```
artifacts/model.pkl
artifacts/scaler.pkl
artifacts/feature_names.pkl
```

---

## 🌐 API (FastAPI)

Endpoint:

```
POST /predict
```

Returns:

```
{
  fraud_probability: float,
  risk_level: LOW | MEDIUM | HIGH
}
```

Includes schema enforcement and logging.

---

## 🖥️ Streamlit Dashboard

* Real-time transaction simulation
* Fraud probability visualization
* Risk classification display
* Connected to backend API

Run:

```
streamlit run streamlit_app.py
```

---

## 🐳 Docker (Multi-Service)

Backend API and UI are containerized.

Run full system:

```
docker compose up --build
```

---

## 🔄 CI/CD

GitHub Actions automatically:

* installs dependencies
* builds containers
* validates project structure

---

## 📈 Monitoring

Prediction logs stored for:

* fraud rate tracking
* anomaly detection
* retraining triggers

---

## 🧪 Local Setup

Install dependencies:

```
pip install -r requirements.txt
```

Run pipeline:

```
python src/ingestion.py
python src/features.py
python src/train.py
```

Run API:

```
uvicorn app.main:app --reload
```

Run dashboard:

```
streamlit run streamlit_app.py
```

---

## 🔮 Future Improvements

* XGBoost / ensemble fraud model
* Real-time streaming ingestion (Kafka)
* Feature store integration
* Drift detection dashboard
* Automated retraining pipeline

---

## 👨‍💻 Author

Mahammed Rafi
AI & ML Engineering Student
GitHub: https://github.com/Rafff-ml
