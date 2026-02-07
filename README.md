#  Telco Customer Churn Prediction Analysis

This project focuses on predicting customer churn for a telecommunications company using **Logistic Regression**. The primary goal is to identify high-risk customers and understand the key drivers behind their decision to leave.



##  Project Motivation
In the telecom industry, acquiring a new customer is significantly more expensive than retaining an existing one. This project implements a machine learning pipeline that not only predicts churn but also optimizes for **Recall**, ensuring that the business identifies as many potential "churners" as possible to take proactive retention actions.

## Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `seaborn`, `matplotlib`
* **Machine Learning:** `scikit-learn`

## Workflow & Methodology

### 1. Data Cleaning & Preprocessing
* Handled missing values in `TotalCharges` (converted from object to numeric).
* Removed irrelevant features like `customerID`.
* Applied **One-Hot Encoding** to categorical variables.
* Implemented **Feature Scaling** using `StandardScaler` to handle the varying scales of tenure and charges.

### 2. Exploratory Data Analysis (EDA)
Key insights discovered:
* Customers with **Month-to-month contracts** have a significantly higher churn rate.
* New customers (low tenure) are more likely to leave.
* High **Monthly Charges** correlate with increased churn.



### 3. Model Development & Optimization
I implemented two versions of the Logistic Regression model:
* **Baseline Model:** A standard implementation using default parameters.
* **Improved Model:** Optimized using:
    * `class_weight="balanced"`: To handle the imbalance between "Churn" and "No Churn" classes.
    * `L1 Regularization (Lasso)`: To perform automatic feature selection and prevent overfitting.

##  Performance Comparison

| Metric | Baseline Model | Improved Model (Balanced) |
| :--- | :--- | :--- |
| **Recall (Churn Class)** | Lower | **Significantly Higher** |
| **F1-Score** | Standard | **Better Balanced for Business** |

*Note: While overall accuracy might decrease slightly in the improved model, the **Recall** is prioritized to minimize missing customers who are likely to leave.*

##  Key Churn Drivers (Feature Importance)
Based on the model coefficients:
* **Positive Impact (Increases Churn):** Fiber Optic internet, Month-to-month contracts, Paperless billing.
* **Negative Impact (Increases Retention):** Two-year contracts, High Tenure, DSL service.



##  Repository Structure
* `churn_analysis.py`: Main Python script containing the full pipeline.
* `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset used for analysis.
* `README.md`: Project documentation.


##  Install the required dependencies:

Bash
pip install pandas matplotlib seaborn scikit-learn
