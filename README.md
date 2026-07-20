# -Decision-Tree-Implementation-Classification-Regression-
A Machine Learning project demonstrating the implementation of **Decision Tree Classification** and **Decision Tree Regression** using **Scikit-learn**. This project covers data preprocessing, model training, hyperparameter tuning, model evaluation, prediction, and tree visualization.
Decision Tree is a supervised machine learning algorithm that can solve both:

- **Classification** problems (predicting categories or classes)
- **Regression** problems (predicting continuous numerical values)

It builds a tree-like structure by recursively splitting the data based on the feature that best reduces impurity (classification) or prediction error (regression).

---

# 📂 Project Structure

```text
Decision-Tree/
│
├── data/
│   ├── classification_dataset.csv
│   └── regression_dataset.csv
│
├── notebooks/
│   ├── DecisionTree_Classification.ipynb
│   └── DecisionTree_Regression.ipynb
│
├── images/
│   └── decision_tree.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Features

- Data Loading
- Data Preprocessing
- Train-Test Split
- Decision Tree Classification
- Decision Tree Regression
- Hyperparameter Tuning using GridSearchCV
- Model Evaluation
- Prediction
- Decision Tree Visualization

---

# 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Decision-Tree.git
```

Move into the project directory

```bash
cd Decision-Tree
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

# 📚 Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

---

# ⚙️ Workflow

## Step 1

Load the dataset.

---

## Step 2

Perform data preprocessing.

- Handle missing values
- Encode categorical variables
- Feature selection
- Separate features and target

---

## Step 3

Split the dataset into training and testing sets.

---

## Step 4

Train the Decision Tree model.

### Classification

```python
model = DecisionTreeClassifier()
```

### Regression

```python
model = DecisionTreeRegressor()
```

Train the model

```python
model.fit(X_train, y_train)
```

---

# 🔧 Hyperparameter Tuning

Example parameter grid

```python
parameters = {
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
```

For regression

```python
parameters = {
    "criterion": [
        "squared_error",
        "friedman_mse",
        "absolute_error",
        "poisson"
    ],
    "splitter": ["best", "random"],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
```

Use GridSearchCV

```python
grid = GridSearchCV(model, parameters, cv=5)
grid.fit(X_train, y_train)
```

---

# 📊 Model Evaluation

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

Example

```python
accuracy_score(y_test, y_pred)

confusion_matrix(y_test, y_pred)

classification_report(y_test, y_pred)
```

---

## Regression Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Example

```python
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)
```

---

# 🌳 Decision Tree Visualization

```python
plt.figure(figsize=(20,10))

plot_tree(
    model,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.show()
```

---

# 🎯 Prediction

```python
prediction = model.predict(X_test)
```

---

# ⚡ Decision Tree Classification

### Used For

- Spam Detection
- Disease Classification
- Customer Churn Prediction
- Loan Approval
- Email Classification
- Fraud Detection

### Common Criteria

- Gini Index
- Entropy
- Log Loss

---

# 📈 Decision Tree Regression

### Used For

- House Price Prediction
- Salary Prediction
- Sales Forecasting
- Stock Price Prediction
- Temperature Prediction
- Demand Forecasting
