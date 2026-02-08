# Task 11: SVM – Breast Cancer Classification

## 📌 Project Overview
This project implements **Support Vector Machine (SVM)** based classification to predict whether a breast tumor is **malignant or benign** using the **Breast Cancer Wisconsin dataset** from `scikit-learn`.

The task focuses on understanding **kernel-based classification**, **feature scaling**, and **hyperparameter tuning** using **GridSearchCV**, along with proper model evaluation using **ROC–AUC** metrics.

---

## 🎯 Objectives
- Load and analyze the breast cancer dataset
- Apply feature normalization using `StandardScaler`
- Train and compare:
  - Linear SVM
  - RBF Kernel SVM
- Tune hyperparameters (`C`, `gamma`) using GridSearchCV
- Evaluate model using:
  - Confusion Matrix
  - Classification Report
  - ROC Curve & AUC Score
- Save trained model and evaluation outputs for reuse

---

## 🧰 Tools & Technologies
- Python  
- Scikit-learn  
- Matplotlib  
- Joblib  
- Google Colab  

---

## 📂 Dataset
- **Primary Dataset:** Breast Cancer Wisconsin Dataset  
- **Source:** `sklearn.datasets.load_breast_cancer()`

**Target Classes:**
- `0` → Malignant  
- `1` → Benign  

---

## ⚙️ Project Workflow
1. Load dataset and inspect feature distribution  
2. Split data into training and testing sets  
3. Apply feature scaling using `StandardScaler`  
4. Train baseline **Linear SVM**  
5. Train **RBF SVM** with hyperparameter tuning  
6. Select best model using **ROC-AUC score**  
7. Evaluate performance using classification metrics  
8. Plot and save ROC Curve  
9. Save final trained model pipeline  

---

## 📈 Model Evaluation
- **Metrics Used:**
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC Curve
  - AUC Score

All evaluation outputs are automatically saved, eliminating the need for screenshots.

---

## 💾 Saved Outputs
After execution, the following files are generated:

outputs/
├── roc_curve.png
├── classification_report.txt
├── results_summary.txt
└── svm_breast_cancer_pipeline.pkl
