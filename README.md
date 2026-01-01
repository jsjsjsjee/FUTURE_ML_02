# 📊 Telecom Customer Churn Prediction System

An end-to-end **Machine Learning project** that predicts **customer churn probability** and provides **actionable business insights** using an interactive **Streamlit dashboard**.

This project was developed as part of a **Machine Learning Internship at Future Interns**.

---

## 🚀 Project Overview

Customer churn is a major challenge in the telecom industry, leading to revenue loss and reduced customer lifetime value.  
This project aims to **identify customers who are likely to churn** by analyzing customer behavior, service usage, and billing patterns.

The system not only predicts churn but also explains **why customers churn** through feature importance analysis and evaluation metrics.

---

## 🎯 Key Features

✔ Model that predicts **churn probability per customer**  
✔ **Feature importance chart** identifying key churn drivers  
✔ **Confusion matrix and evaluation metrics** (Accuracy, Precision, Recall, F1-score)  
✔ **Dashboard / PDF-ready business insights**  
✔ **Interactive Streamlit UI** to demo the system  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Web App:** Streamlit  
- **Model:** Random Forest Classifier  

---

## 📂 Project Structure

Task-2/
│── app.py # Streamlit application
│── train_model.py # Model training script
│── preprocess.py # Data preprocessing logic
│── telco.csv # Telecom customer dataset
│── churn_model.pkl # Trained ML model
│── features.pkl # Feature names
│── metrics.pkl # Evaluation metrics
│── confusion_matrix.pkl # Confusion matrix
│── feature_importance.pkl # Feature importance values
│── requirements.txt
│── README.md

yaml
Copy code

---

## 📊 Dataset Description

The dataset contains detailed telecom customer information including:

- Demographics (Age, Gender, Senior Citizen)
- Account information (Tenure, Contract, Payment Method)
- Service usage (Internet, Streaming, Data usage)
- Financial metrics (Monthly Charges, Total Charges, Revenue)
- Customer churn details

### 🎯 Target Variable
- **Churn Label**  
  - `Yes` → Customer churned  
  - `No` → Customer retained  

---

## ⚙️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
2️⃣ Train the Model
bash
Copy code
python train_model.py
This will generate:

churn_model.pkl

features.pkl

metrics.pkl

confusion_matrix.pkl

feature_importance.pkl

3️⃣ Run the Streamlit App
bash
Copy code
streamlit run app.py
The application will open in your browser and allow real-time churn prediction.

📈 Model Evaluation Metrics
The model is evaluated using standard classification metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

These metrics help assess the reliability and effectiveness of the churn prediction model.

🔥 Feature Importance Analysis
The feature importance chart highlights the factors that most influence churn, such as:

Contract type

Monthly charges

Customer tenure

Payment method

This helps businesses understand why customers leave.

💡 Business Insights
Customers on month-to-month contracts have a higher churn risk

High monthly charges increase churn probability

New customers (low tenure) are more likely to churn

Payment methods also influence churn behavior

📌 Recommendations
Encourage customers to switch to long-term contracts

Provide loyalty benefits to high-risk customers

Personalize offers based on customer usage patterns

Improve customer satisfaction and support services

🚀 Future Enhancements
Power BI dashboard integration

Automated PDF report generation

Real-time churn prediction API

Advanced models (XGBoost, Neural Networks)

👤 Author
Rushik
Machine Learning Intern – Future Interns

⭐ If you found this project useful, feel free to star the repository!