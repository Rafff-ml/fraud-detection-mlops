💳 FinTech Fraud Detection MLOps System

An end-to-end production-ready FinTech fraud detection system built with FastAPI, Streamlit, Docker, CI/CD, and cloud deployment.

This project simulates a real payment risk engine used by fintech companies to detect suspicious transactions in real time.

---

🚀 Live Demo

🌐 Streamlit App

👉 https://fraud-detection-mlops.streamlit.app/

⚡ Live Prediction API

👉 https://fraud-detection-mlops-iiib.onrender.com/docs

---

⭐ Key Highlights

✅ Real-time fraud prediction API
✅ FinTech feature engineering (velocity, deviation, night risk)
✅ Streamlit interactive dashboard
✅ Dockerized backend + frontend
✅ GitHub Actions CI pipeline
✅ Production cloud deployment
✅ Clean modular MLOps architecture

---

🧠 FinTech Features Used

* Transaction velocity (last 10 transactions)
* Average spend behaviour
* Amount deviation detection
* Log amount transformation
* Night transaction risk
* Behaviour drift signal

---

🏗️ Tech Stack

ML: Scikit-learn, Pandas, Numpy
Backend: FastAPI, Uvicorn
Frontend: Streamlit
MLOps: Docker, GitHub Actions
Deployment: Render + Streamlit Cloud

---

📂 Project Structure

app/            → FastAPI inference service  
src/            → training + pipelines  
artifacts/      → trained model + feature objects  
streamlit_app.py → UI dashboard  
Dockerfile      → API container  
Dockerfile.ui   → Streamlit container  
.github/        → CI pipeline  

---

🔥 Local Run

Backend

uvicorn app.main:app --reload

Frontend

streamlit run streamlit_app.py

---

🐳 Docker Run

docker-compose up --build

---

🎯 Future Improvements

* Monitoring dashboard
* Data drift detection
* Auto retraining pipeline
* Kafka streaming simulation
* Payment graph fraud detection

---

👨‍💻 Author

Rafff — AI/ML Engineer (FinTech + MLOps Focus)
