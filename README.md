# Customer Churn Prediction with a Focus on Data Preprocessing

This project is an end-to-end machine learning pipeline focused on predicting customer churn for a telecommunications company. A significant emphasis is placed on the crucial stages of **data cleaning**, **exploratory data analysis (EDA)**, **feature preprocessing**, and **model interpretation**. The goal is to build a reliable classification model and, more importantly, to understand the key drivers behind customer churn.

---

## 📊 Project Workflow

The project follows a structured workflow, with each phase documented in the `notebooks/` directory.

### 1. Exploratory Data Analysis (EDA)
*(See `notebooks/01-eda.ipynb`)*

* **Initial Analysis:** The dataset was loaded and examined using `pandas`. Initial checks with `.info()`, `.describe()`, and `.isnull().sum()` were performed to understand the data's structure, types, and basic statistics.
* **Key Finding:** A critical discovery was that the `TotalCharges` column, which should be numerical, was an `object` data type. This was due to hidden empty strings for customers with zero tenure, which were not caught by `.isnull()`.

---

### 2. Data Cleaning and Preprocessing
*(See `notebooks/02-data-preprocessing.ipynb`)*

This phase involved a series of transformations to prepare the raw data for machine learning models.

* **Handling Missing Values:**
    * The empty strings in `TotalCharges` were converted to `NaN` (Not a Number) using `pd.to_numeric` with `errors='coerce'`.
    * These `NaN` values (11 in total) were then imputed using the **median** of the column to maintain the data distribution without being skewed by outliers.

* **Encoding Categorical Features:** Machine learning models require numerical input, so all categorical (text-based) features were converted to numbers.
    * **Label Encoding (for Binary Features):** Columns with two distinct values (e.g., 'Yes'/'No', 'Male'/'Female') were mapped directly to `1` and `0`.
    * **One-Hot Encoding (for Multi-class Features):** For columns with more than two categories (e.g., `Contract`, `InternetService`), One-Hot Encoding was applied using `pd.get_dummies()`. This technique creates new binary columns for each category to prevent the model from assuming a false ordinal relationship between them. The `drop_first=True` argument was used to avoid multicollinearity.

---

### 3. Model Training and Evaluation
*(See `notebooks/03-model-training-and-evaluation.ipynb`)*

* **Data Splitting:** The fully preprocessed dataset was split into an 80% training set and a 20% testing set using `train_test_split` from Scikit-learn. This ensures the model is evaluated on data it has never seen before.

* **Baseline Model: Logistic Regression:**
    * A `LogisticRegression` model was chosen as the initial baseline due to its simplicity and high interpretability.
    * The model was trained on the training data using the `.fit()` method.

* **Evaluation:**
    * The trained model was used to make predictions on the test set.
    * **Accuracy:** The model achieved an overall accuracy of approximately **81.6%**.
    * **Confusion Matrix:** A confusion matrix was generated to analyze the model's performance in detail. It revealed that the model is very good at identifying customers who **do not churn** but is weaker at correctly identifying customers who **do churn** (higher number of False Negatives).

---

### 4. Model Interpretation
*(See `notebooks/03-model-training-and-evaluation.ipynb`)*

* **Feature Importance:** The coefficients (`model.coef_`) of the trained Logistic Regression model were extracted to understand the importance and influence of each feature.
    * **Positive Coefficients:** Indicate features that increase the probability of churn (e.g., Month-to-month contracts, Fiber optic internet).
    * **Negative Coefficients:** Indicate features that decrease the probability of churn, signaling customer loyalty (e.g., long tenure, Two-year contracts).
* A visualization was created to display the most impactful features, providing valuable business insights into the key drivers of churn.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nonfungi/customer-churn-prediction.git](https://github.com/nonfungi/customer-churn-prediction.git)
    cd customer-churn-prediction
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Explore the Jupyter Notebooks** in the `notebooks/` directory to follow the project step-by-step.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook