# credit_balance_regession
# Financial Data Modeling: Credit Balance Prediction
By Benjamin Cabrera & Alexander Ohye

Predicting credit card balances using Linear, Ridge, and Lasso regression models with cross-validation and feature selection (ISLR Credit dataset).

## Introduction

This project explores how demographic and financial data can be used to **predict credit card balances** using machine learning regression techniques.  

The dataset contains variables such as income, credit limit, credit rating, education, and marital/student status.  

The goal is to apply the regression approaches Linear, Ridge, and Lasso to model the relationship between predictors and credit card balances, evaluate their performance, and identify the most influential factors.  

A data frame with 400 observations on the following variables:

- ID: Identification  
- Income: Income in $1,000  
- Limit: Credit limit  
- Rating: Credit rating  
- Cards: Number of credit cards  
- Age: Age in years  
- Education: Number of years of education  
- Gender: Male/Female  
- Student: Yes/No  
- Married: Yes/No  
- Balance: Average credit card balance ($)

**The outcome variable to predict is *Balance*.**

---

## Methods

- Data cleaning and encoding of categorical variables  
- Train/test split and K-Fold cross-validation  
- Linear Regression, Ridge Regression, and Lasso Regression  
- Hyperparameter tuning via GridSearchCV  
- Evaluation metrics: RMSE and R²  

Libraries used:
`pandas`, `numpy`, `matplotlib`, `scikit-learn`

---

## Key Results

**Ridge Regression**
- Best alpha: 1  
- CV MSE: 9944.63  
- Test RMSE: 113.30  
- Test error ≈ 21.77%

**Lasso Regression**
- α = 10 → R² = 0.939  
- α = 1 → R² = 0.949 (best)  
- Zeroed features: Education, Gender_Female, Married_Yes  
- Key predictors: Limit, Rating, Income, Student_Yes, Cards, Age  

**Ridge equation (standardized features, rounded):**  
Balance ≈ 508.39 − 258.28·Income + 310.90·Limit + 278.17·Rating + 18.64·Cards − 13.73·Age − 1.49·Education + 0.14·Gender_Female + 120.48·Student_Yes + 0.70·Married_Yes  

---

## Interpretation

Higher credit limits and ratings are associated with larger balances, while higher income correlates with lower balances when controlling for other variables.  
Student status is a strong positive predictor of balance, suggesting higher utilization rates among students.

---

## Conclusion

Regularization significantly improves predictive accuracy and reduces overfitting when modeling financial behavior.  
While ridge regression stabilized coefficients, lasso achieved the best performance (R² ≈ 0.95) and effectively removed less relevant predictors.  
This project demonstrates how regularized regression can both enhance model reliability and highlight the most influential financial variables affecting credit balance.

---

## How to Run

1. Clone this repository.  
2. Place `Credit_ISLR.csv` in the project root.  
3. Open `credit-balance-regression.ipynb` in Jupyter or Google Colab.  
4. Run all cells to reproduce results.

---

## Skills Demonstrated

- Data preprocessing and encoding  
- Linear, Ridge, and Lasso regression modeling  
- Cross-validation and hyperparameter tuning  
- Model interpretation and feature selection  
- Python machine learning workflow (scikit-learn)
