
#  Loan Risk Prediction Model

##  Overview

This project aims to build a robust and production-ready machine learning pipeline to **predict loan applicant risk** (`Risk_Flag`). The pipeline includes data loading, exploratory data analysis (EDA), preprocessing, feature engineering, model training, evaluation, and fairness analysis.

> **Goal**: Classify applicants as either:
> - `1` → **Risky** (likely to default)
> - `0` → **Not Risky** (likely to repay)

This solution is designed for use by financial institutions to support risk-based decision-making.

---

##  Dataset Description

The dataset is divided into **training** and **test** sets (both in JSON format). Each entry represents an individual loan applicant, characterized by various demographic and financial attributes.

### Key Features

| Feature             | Description                                 |
|---------------------|---------------------------------------------|
| Id                  | Unique identifier for each applicant        |
| Income              | Income level                                |
| Age                 | Applicant's age                             |
| Experience          | Professional experience (years)             |
| Married/Single      | Marital status                              |
| House_Ownership     | Home ownership status                       |
| Car_Ownership       | Car ownership status                        |
| Profession          | Applicant’s occupation                      |
| CITY                | City of residence                           |
| STATE               | State of residence                          |
| CURRENT_JOB_YRS     | Years at current job                        |
| CURRENT_HOUSE_YRS   | Years at current residence                  |
| Risk_Flag           | **Target Variable** (1 = Risky, 0 = Not Risky) |

---

##  Project Workflow

### 1.  Data Loading
- Handled using `pandas` for flexibility and portability.
- JSON parsing and validation ensured consistent schema.

### 2.  Exploratory Data Analysis (EDA)
- Visualizations: Histograms, boxplots, correlation heatmaps.
- Identified feature distributions and class imbalance.
- Categorical variables examined via chi-squared tests.

### 3.  Data Cleaning & Preprocessing
- Handled missing values and anomalies.
- Categorical encoding using **OneHotEncoder**.
- Scaled numerical features using **StandardScaler**.
- Addressed class imbalance using **SMOTE** and **ADASYN** techniques.

### 4.  Feature Engineering
- Derived features like:
  - Income-to-Age Ratio
  - Employment Bands
  - Housing Duration Bins
- Feature selection via feature importance and correlation analysis.

### 5.  Model Development
Models trained include:
- Logistic Regression
- Random Forest
- XGBoost
- Stacking Ensemble Model

Used `RandomizedSearchCV` and `StratifiedKFold` for hyperparameter tuning.

### 6.  Evaluation
Evaluated using the following metrics:
-  **Accuracy**
-  **F1-Score**
-  Recall
-  Precision
-  ROC AUC

> **Primary focus**: Accuracy and F1-Score

### 7.  Bias & Fairness
- Audited predictions across sensitive attributes.
- Strategies for bias mitigation discussed:
  - Equal opportunity checks
  - Fair sampling techniques

---

##  Installation & Requirements

```bash
pip install -r requirements.txt
```

**Core Libraries**:
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`
- `imblearn`

---

##  How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd loan-risk-prediction
   ```

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook Loan_risk_prediction_model.ipynb
   ```

3. Submit predictions generated from the trained model using the test dataset.

---

##  Results Summary

| Model               | Accuracy | F1-Score | ROC AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | XX.XX%   | XX.XX    | XX.XX   |
| Random Forest       | XX.XX%   | XX.XX    | XX.XX   |
| XGBoost             | XX.XX%   | XX.XX    | XX.XX   |
| **Stacking Ensemble** | **XX.XX%** | **XX.XX** | **XX.XX** |

> Replace "XX.XX" with your actual results after model training.

---

##  Highlights

-  End-to-end ML pipeline
-  Visual insights via seaborn/matplotlib
-  Ensemble modeling for optimal results
-  Cross-validation and hyperparameter tuning
-  Fairness and ethical assessment included

---

##  Final Thoughts

This project represents a real-world financial risk assessment scenario, balancing **technical depth**, **fairness**, and **model explainability**. Future extensions could include:
- SHAP/LIME explainability
- Deployment via API
- Continuous fairness auditing

---

##  Hackathon Readiness Checklist

- [x] Reusable code modules
- [x] Interpretable EDA and visual storytelling
- [x] Fairness and bias analysis
- [x] Ensemble modeling
- [x] Clean documentation

---

## License

This project is released under the MIT License.
