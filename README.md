# Stroke Prediction Project

## 📌 Overview

This project focuses on predicting stroke risk using **machine learning** and **data analysis** techniques. The dataset includes various health and demographic factors that influence the likelihood of a stroke. The dataset consists of **5110 rows**, each providing relevant patient information.


## 🔍 Objectives

- Analyze the relationship between various **attributes** and **stroke occurrence**.
- **Preprocess data**, including handling missing values and feature engineering.
- Train **machine learning models** (Logistic Regression & Random Forest) to predict stroke risk.
- Evaluate models using metrics such as **accuracy, precision, recall, and F1-score**.
- **Visualize** important insights and prediction results.

## 🛠 Technologies Used

- **Python** 🐍
- **Pandas, NumPy** (Data handling)
- **Matplotlib, Seaborn** (Visualization)
- **Scikit-Learn** (Machine Learning)
- **Jupyter Notebook** (Interactive Analysis)

## 📊 Dataset

The dataset used in this project is **healthcare-dataset-stroke-data.csv**.
The dataset contains the following attributes:

- **ID** (Removed during preprocessing)
- **Gender**
- **Age**
- **Hypertension**
- **Heart Disease**
- **Marital Status**
- **Work Type**
- **Residence Type**
- **Glucose Level**
- **BMI (Body Mass Index)**
- **Smoking Status**
- **Stroke (Target Variable)**

### 🏗 Data Preprocessing

- **Handled missing values** (BMI column had **201 missing values**, filled with median).
- **Removed unnecessary columns** (e.g., ID).
- **Applied Label Encoding** for categorical variables.
- **Addressed data imbalance** using **SMOTE** to oversample the minority class.
- **Split the dataset** into **75% training** and **25% testing**.

## 🚀 Model Implementation

### **1️⃣ Logistic Regression**

- A supervised learning method used for classifying data into **binary outcomes**.
- Uses a **sigmoid function** to estimate the probability of stroke occurrence.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

lr = LogisticRegression(solver='newton-cg', penalty='l2')
lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)
lr_pred_proba = lr.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, lr_pred_proba))
```

### **2️⃣ Random Forest Classifier**

- An ensemble method using multiple **decision trees** trained with the **bagging method**.
- Helps in capturing **complex, non-linear relationships**.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', n_estimators=50, random_state=1)
rf.fit(x_train, y_train)

rf_pred = rf.predict(x_test)
rf_pred_proba = rf.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, rf_pred_proba))
```

## 🏆 Results & Findings

- The dataset is **imbalanced**, with only **4.9%** of cases labeled as stroke.
- **Logistic Regression Accuracy**: **76%**
- **Random Forest Accuracy**: **89%**
- **Random Forest outperformed Logistic Regression**, as it better captures non-linear relationships in the data.

## 📢 Future Improvements

- Implement **deep learning models** (e.g., Neural Networks) for better accuracy.
- Integrate **real-world datasets** to improve generalization.
- Develop a **web-based application** for stroke risk assessment.

## 📌 Contributors

- **Khulud Alshammari** - *Project Lead & AI Specialist*
