# Diabetes Risk Screening — Machine Learning Project

This project predicts whether a person is **at risk for diabetes** using demographic and health-related features from the Kaggle Diabetes Dataset. The goal is to support early screening and help clinics identify individuals who may need further testing.

---

## 📌 Project Overview

We built a machine learning model that classifies patients as:

- **0 — Not at risk**
- **1 — At risk**

Dataset used:  
https://www.kaggle.com/datasets/mathchi/diabetes-data-set

---

## 📊 Dataset Description

The dataset includes:

- 768 patient records  
- 8 input features  
- 1 output label (Outcome)

Features include:

- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  

---

## 🧠 Models Used

We trained multiple binary classification models:

- **Logistic Regression** — baseline model  
- **Decision Tree** — rule-based model  
- **Random Forest** — ensemble model  
- **KNN** — distance-based model  

We evaluated accuracy, precision, recall, and confusion matrices.

---

## ⚙️ Training & Testing

- 80/20 train-test split  
- StandardScaler for normalization  
- Missing values handled  
- Cross-validation performed  
- Bias checks across demographic features (where applicable)

---

## 🖥️ User Interaction

The app provides a clean interface:

1. User enters values such as glucose, BMI, age, pregnancies, etc.  
2. User clicks **Predict**  
3. The model outputs **“At Risk”** or **“Not At Risk”**  
4. Users can adjust inputs and run multiple predictions

---

## 📁 Repository Structure
├── README.md
├── requirements.txt
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── eda_and_visualization.ipynb    
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── model.pkl                       
├── app.py                              
└── .gitignore

---

## 👥 Team Members

- Anish  
- Marlene  
- Ingrid  
- Christian  
- Aaryav  
- Alessandra  

---

## 📚 Citations

CDC National Diabetes Statistics Report  
https://www.cdc.gov/diabetes/php/data-research/index.html

Dataset  
https://www.kaggle.com/datasets/mathchi/diabetes-data-set
