# 🏏 T20 Score Predictor (Machine Learning)

A machine learning application that predicts the final score of a T20 cricket innings using real match conditions.

The model uses ball-by-ball match data and match state features to estimate the final total.

---

## 🚀 Features

• Predict final T20 score using current match situation  
• Uses Machine Learning (Random Forest Regressor)  
• Interactive Streamlit web interface  
• Built using real ball-by-ball cricket data  

---

## ⚙️ Input Parameters

The model uses the following match state features:

- Current Score
- Overs Completed
- Wickets Out
- Runs in Last 5 Overs
- Balls Remaining
- Current Run Rate

---

## 🧠 Machine Learning Model

Model used:

Random Forest Regressor

Feature engineering includes:

- Current score progression
- Balls remaining
- Current run rate
- Momentum from last overs

---

## 🖥️ Streamlit Application

The project includes a Streamlit interface where users can enter live match data and predict the final score.

Run locally:
streamlit run app.py


---

## 📊 Dataset

The model was trained on **1,100,000+ ball-by-ball T20 deliveries**.

Due to GitHub file size limits, the dataset and trained model are not included in this repository.

---

## 🛠️ Tech Stack

Python  
Pandas  
NumPy  
Scikit-Learn  
Streamlit  

---

## 👨‍💻 Author

Shreyans Rath
