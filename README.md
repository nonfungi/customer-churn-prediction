# Customer Churn Prediction Project

## Overview

This project focuses on predicting customer churn for a telecommunications company. The primary goal is to demonstrate a comprehensive data preparation and preprocessing workflow using Python libraries such as **Pandas**, **NumPy**, and **Scikit-learn**. By analyzing customer data, we aim to build a machine learning model that can identify customers who are likely to cancel their subscriptions.

This project emphasizes the critical steps of the data science lifecycle before model training, including:
- Exploratory Data Analysis (EDA)
- Data Cleaning and Imputation
- Feature Engineering and Transformation
- Feature Scaling and Encoding

## Project Structure

```
customer-churn-prediction/
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01-eda.ipynb
│   └── 02-data-preprocessing.ipynb
├── data/
│   ├── raw/
│   │   └── telco-customer-churn.csv
│   └── processed/
│       └── cleaned_churn_data.csv
└── src/
    ├── data_preprocessing.py
    └── train_model.py
```

## Dataset

The dataset used in this project is the "Telco Customer Churn" dataset, which contains information about customers, their subscribed services, and whether they churned or not.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [Your-GitHub-Repo-Link]
    cd customer-churn-prediction
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Explore the Jupyter Notebooks:**
    Open the `notebooks/` directory to follow the step-by-step process of data exploration and preprocessing.

## Workflow

1.  **Exploratory Data Analysis (01-eda.ipynb):** Initial analysis to understand data distributions, identify missing values, and discover relationships between features.
2.  **Data Preprocessing (02-data-preprocessing.ipynb):** Cleaning the data, handling missing values, encoding categorical features, and scaling numerical features.
3.  **Model Training (src/train_model.py):** (Future Step) Implementing and training a classification model on the processed data.

---
*This project is created for learning and demonstration purposes.*