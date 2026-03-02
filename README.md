# Diabetes_predict
An interactive CLI tool for predicting diabetes risk and managing patient records For my class assessment 

# 🩺 Interactive Diabetes Risk Assessment & Management System

**Author:** Rupam Roy  
**Environment:** Kali Linux (Raspberry Pi 5) / Python 3.x

## 📌 Project Overview
This project is a standalone, terminal-based clinical application that combines **Machine Learning** with an **Electronic Health Record (EHR)** management system. It utilizes the Pima Indians Diabetes Dataset to predict the probability of a patient developing diabetes using a Gaussian Naive Bayes classification algorithm. 

Beyond simple predictions, this tool features a highly robust, interactive Command Line Interface (CLI) that allows users to securely enter, validate, store, and query patient data in real-time.

## ✨ Key Features
* **Machine Learning Engine:** Implements GaussianNB from Scikit-Learn with a StandardScaler pipeline for highly accurate, probability-based diagnostic predictions.
* **Persistent Database (CRUD):** Automatically logs all patient assessments to a local patient_logs.csv database.
* **Intelligent Querying:** Features a dynamic, case-insensitive search engine. Users can query the database by ID, full/partial Name, Age, or specific medical biometrics.
* **Advanced Input Validation:** Custom mathematical date parsing, strict demographic clamping, and fault-tolerant session handling.
* **Gender-Aware Logic:** Dynamically adapts medical inputs based on patient gender.
* **Dynamic Tabular UI:** Uses Pandas to generate clean, readable data tables directly in the terminal interface.

## 🛠️ Technology Stack
* **Language:** Python 3
* **Libraries:** pandas, numpy, scikit-learn, os, re, time, sys
* **Algorithm:** Gaussian Naive Bayes

## 🚀 How to Run

1. Clone the repository:
git clone https://github.com/RupamR0y/Diabetes_predict.git
cd Diabetes_predict

2. Set up a virtual environment:
python3 -m venv ca2_test_env
source ca2_test_env/bin/activate

3. Install dependencies:
pip install pandas numpy scikit-learn

4. Execute the application:
python3 diabetes_predict.py

## ⚠️ Disclaimer
This software is developed for educational and academic purposes only. The predictive model is trained on the historical Pima Indians Diabetes dataset. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
