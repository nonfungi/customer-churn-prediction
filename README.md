# Customer Churn Prediction with a Focus on Data Preprocessing

This project is an end-to-end machine learning pipeline focused on predicting customer churn for a telecommunications company. A significant emphasis is placed on the crucial stages of **data cleaning**, **exploratory data analysis (EDA)**, **feature preprocessing**, and **model comparison**. The goal is to build a reliable classification model and understand the key drivers behind customer churn.

---

## 🏆 Final Results and Conclusion

The final selected model for this project is the **`LogisticRegression` model**.

This project serves as a practical example of the principle that model complexity does not guarantee better performance. While a more advanced `RandomForestClassifier` was also trained, the simpler, more interpretable baseline model proved to be more effective for the specific business goal of minimizing missed churners (False Negatives).

* **Winning Model Accuracy:** ~81.6%
* **Key Business Insights:** The model interpretation revealed that key drivers of churn include having a **month-to-month contract**, **fiber optic internet service**, and using **electronic check payments**. Conversely, factors like **long tenure** and subscribing to services like **tech support** and **online security** are strong indicators of customer loyalty.

---

## 📊 Project Workflow

The project follows a structured workflow, with each phase documented in the `notebooks/` directory.

### 1. Exploratory Data Analysis (EDA)
*(See `notebooks/01-eda.ipynb`)*

The dataset was loaded and examined to understand its structure, types, and basic statistics. A critical discovery was that the `TotalCharges` column was an `object` data type due to hidden empty strings, which were not caught by standard null checks.

### 2. Data Cleaning and Preprocessing
*(See `notebooks/02-data-preprocessing.ipynb`)*

* **Handling Missing Values:** Empty strings in `TotalCharges` were converted to `NaN` and then imputed using the column's **median**.
* **Encoding Categorical Features:** All text-based features were converted to a numerical format.
    * **Label Encoding:** Binary features ('Yes'/'No') were mapped to `1`/`0`.
    * **One-Hot Encoding:** Multi-class features (e.g., `Contract`) were converted into binary columns using `pd.get_dummies()` to prevent the model from assuming a false ordinal relationship.

### 3. Model Training and Comparison
*(See `notebooks/03-model-training-and-evaluation.ipynb`)*

* **Data Splitting:** The dataset was split into an 80% training set and a 20% testing set.

* **Baseline Model (Logistic Regression):** A `LogisticRegression` model was trained as an interpretable baseline. It achieved an accuracy of **~81.6%**. Its confusion matrix showed it was strong at identifying non-churners but weaker at finding true churners (158 False Negatives).

* **Advanced Model (Random Forest):** To improve upon the baseline, a more complex `RandomForestClassifier` was also trained. It achieved an accuracy of **~79.9%**.

* **Model Comparison:** A direct comparison revealed a critical insight: despite being a more complex algorithm, the Random Forest model performed worse. It was not only less accurate overall but was also significantly less effective at identifying true churners, resulting in more False Negatives (200 vs. 158).

### 4. Model Interpretation
*(See `notebooks/03-model-training-and-evaluation.ipynb`)*

Based on the comparison, the **Logistic Regression** model was selected as the final model. Its coefficients (`model.coef_`) were analyzed to understand the key drivers of churn, providing the business insights mentioned in the conclusion.

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