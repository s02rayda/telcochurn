# 📊 Telecom Customer Churn Prediction & Analysis

A comprehensive, end-to-end data science project that predicts customer churn using machine learning and presents actionable insights through an interactive web dashboard.

![Streamlit Dashboard Screenshot](https://i.imgur.com/YOUR_SCREENSHOT_URL.png) 
*<(Note: Replace this link with a screenshot of your running Streamlit app for maximum impact!)*

---

## 🎯 Business Problem

Customer churn is a major challenge for telecom companies, leading to significant revenue loss. This project aims to solve this by building a predictive model that can identify customers at high risk of leaving. By flagging these customers proactively, the business can implement targeted retention strategies to improve customer loyalty and protect its revenue base.

---

## ✨ Key Features & Highlights

This project demonstrates a complete machine learning workflow from data to deployment:

-   **End-to-End Data Pipeline:** Comprehensive data cleaning, exploratory analysis, and feature engineering using Pandas and Scikit-learn.
-   **Comparative Model Analysis:** Rigorous training and evaluation of multiple models, including **XGBoost**, **LightGBM**, and **CatBoost**, to select the optimal solution.
-   **Advanced Techniques:** Successfully addressed severe class imbalance using **SMOTE** and optimized model performance through **hyperparameter tuning** (`RandomizedSearchCV`).
-   **Deep Model Interpretability:** Leveraged **SHAP** (SHapley Additive exPlanations) to look inside the "black box" model and understand the key factors driving its predictions.
-   **Interactive Dashboard:** Deployed the final model and insights into a user-friendly web application built with **Streamlit**, transforming complex data into an actionable business tool.

---

## 🚀 Actionable Insights from the Model

The final model revealed several critical drivers of customer churn:

1.  **Contract is King:** Customers on **Month-to-Month contracts** are the highest churn risk. Locking customers into longer-term contracts is the most effective retention tool.
2.  **Early-Life Risk:** **Low customer tenure** is a major predictor of churn. Retention efforts should be heavily focused on the first few months of a customer's lifecycle.
3.  **Price Sensitivity:** **High Monthly Charges** consistently correlate with an increased likelihood to churn.
4.  **Service-Specific Issues:** Customers with **Fiber Optic** internet are surprisingly more likely to leave, indicating a need to investigate service satisfaction or pricing for this premium product.

---

## 📈 Final Model Performance

After a thorough comparison, the **tuned XGBoost Classifier** was selected as the champion model.

| Metric (for Churners) | Precision | Recall | F1-Score |
| :-------------------- | :-------: | :----: | :------: |
| **Final Model**       |   0.53    | **0.77** | **0.63** |

-   **ROC-AUC Score:** **0.83**

The model successfully identifies **77% of all customers who are at risk of churning**, providing a powerful and reliable tool for the business.

---

## 🛠️ Tech Stack

-   **Core Libraries:** Python, Pandas, Scikit-learn, NumPy
-   **Modeling:** XGBoost, LightGBM, CatBoost, Imbalanced-learn
-   **Visualization:** Matplotlib, Seaborn, SHAP
-   **Deployment:** Streamlit, Joblib

---


## ⚙️ Setup & How to Run

Follow these steps to set up the environment and run the interactive dashboard on your local machine.

### 1. Prerequisites

-   Python 3.8 or higher
-   [Git](https://git-scm.com/downloads)

### 2. Clone the Repository

First, clone this repository to your local machine using Git:

```bash
git clone https://github.com/s02rayda/telcochurn.git
cd telcochurn


