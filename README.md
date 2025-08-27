# LoanApprovalPrediction

# Loan Status Prediction

This project aims to build a **Loan Status Prediction System** using machine learning models.  
We compare **Logistic Regression** and **Decision Tree Classifier** to evaluate which model performs better in predicting loan approval status.

---

## 📌 Project Overview

Loan approval is a common real-world problem faced by financial institutions.  
By applying machine learning, we can automate and improve the decision-making process.

In this project:

- We preprocess loan data
- Train ML models
- Evaluate and compare results
- Visualize performance of each model

---

## 📂 Dataset

- The dataset contains loan-related information with the target variable **Loan Status** (`True` = Approved, `False` = Rejected).
- Distribution of target variable:

![Loan Status Distribution](images/loan_status_distribution.png)

---

## ⚙️ Tech Stack

- **Python 3**
- **Pandas, NumPy** – Data processing
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Machine Learning models

---

## 🔄 Data Preprocessing

1. Handling missing values
2. Encoding categorical variables
3. Feature scaling (for Logistic Regression)
4. Train-Test Split

---

## 🧠 Models Used

### 1. Logistic Regression

- Accuracy: **84.02%**
- Suitable for binary classification problems
- Works well with linearly separable data

### 2. Decision Tree Classifier

- Accuracy: **(your result here)%**
- Captures non-linear relationships
- May overfit on small datasets

---

## 📊 Model Comparison

The accuracies of the two models are compared below:

![Model Comparison](images/model_comparison.png)

- **Logistic Regression** performed better on this dataset.
- Decision Tree captured patterns but showed signs of overfitting.

---

## ✅ Results

- Logistic Regression gave **higher accuracy** and more stable results.
- Decision Tree can be tuned further using **max_depth, min_samples_split, pruning** etc.

---

## 🚀 Future Improvements

- Try **Random Forest / XGBoost** for better performance
- Perform **hyperparameter tuning** (GridSearchCV / RandomizedSearchCV)
- Handle **class imbalance** using SMOTE or class weights
- Deploy model using **Flask / Streamlit**

---

## 📎 How to Run

```bash
# Clone the repository
git clone https://github.com/tashfeen786/loan-status-prediction.git
cd loan-status-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Loan_Prediction.ipynb

📬 Author

👤 Tashfeen Aziz

Data Analyst | Python Developer | ML/DL Enthusiast

GitHub: tashfeen786
```
