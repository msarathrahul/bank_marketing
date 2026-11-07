# Bank Marketing Prediction - Binary Classification

**Course Name:** Machine Learning

**Course Code:** AIMLZG565

**Assignment:** Assignment 2 

**Name:** Sarath Rahul Malla

**Student ID:** 2026AJ05622


---

## a. Problem Statement

Direct telemarketing campaigns remain one of the most proactive strategies used by banking institutions to sell long term term deposit products. However, indiscriminate cold calling is resource intensive and often yields low conversion rates while causing customer fatigue. 

The objective of this assignment is to develop and evaluate multiple supervised machine learning classification models to predict whether a banking client will subscribe to a term deposit (`y = "yes"` or `"no"`). By leveraging historical customer demographics, financial indicators, and past campaign interaction features, the bank can optimize campaign targeting, maximize deposit conversions, and minimize operational costs.

---

## b. Dataset Description

- **Dataset Title:** Bank Marketing (with social/economic context)
- **Primary Citation:** 
  > S. Moro, P. Cortez and P. Rita. *A Data Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014, [doi:10.1016/j.dss.2014.03.001](http://dx.doi.org/10.1016/j.dss.2014.03.001).
- **Source Repository:** UCI Machine Learning Repository (Dataset ID: 222) / Moro et al. (2014)
- **Total Dataset Size:** 45,211 instances
- **Test Dataset Size:** 9,043 instances (20% stratified test partition saved as `test_data.csv`)
- **Number of Features:** 16 input attributes (7 numerical, 9 categorical)
- **Target Variable:** `y` — Binary indicator whether the client subscribed to a term deposit (`"yes"` = 1, `"no"` = 0). Class imbalance ratio: ~88.3% "no" vs ~11.7% "yes".

### Attribute Information:

#### 1. Bank Client Data:
1. `age` (numeric): Age of the client.
2. `job` (categorical): Type of job (`admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown`).
3. `marital` (categorical): Marital status (`divorced`, `married`, `single`, `unknown`).
4. `education` (categorical): Education level (`primary`, `secondary`, `tertiary`, `unknown`).
5. `default` (categorical): Credit in default? (`no`, `yes`, `unknown`).
6. `balance` (numeric): Average yearly balance in euros.
7. `housing` (categorical): Has housing loan? (`no`, `yes`, `unknown`).
8. `loan` (categorical): Has personal loan? (`no`, `yes`, `unknown`).

#### 2. Related to the Last Contact of the Current Campaign:
9. `contact` (categorical): Communication type (`cellular`, `telephone`, `unknown`).
10. `day_of_week` (numeric / categorical): Last contact day of the month.
11. `month` (categorical): Last contact month of year (`jan`, `feb`, `mar`, ..., `nov`, `dec`).
12. `duration` (numeric): Last contact duration in seconds.

#### 3. Other Campaign Attributes:
13. `campaign` (numeric): Number of contacts performed during this campaign for this client.
14. `pdays` (numeric): Number of days that passed by after the client was last contacted from a previous campaign (-1 means not previously contacted).
15. `previous` (numeric): Number of contacts performed before this campaign for this client.
16. `poutcome` (categorical): Outcome of previous marketing campaign (`failure`, `other`, `success`, `unknown`).

#### 4. Output Target:
17. `y` (binary): Has client subscribed to a term deposit? (`yes`, `no`).

---

## c. GitHub Repository Link

- **GitHub Repository:** [https://github.com/msarathrahul/bank_marketing.git](https://github.com/msarathrahul/bank_marketing.git)

### Repository Structure:
```text
bank_marketing/
│-- app.py                  # Streamlit web application frontend
│-- requirements.txt        # Deployment dependencies
│-- README.md               # Assignment documentation & comparison report
│-- test_data.csv           # 20% Stratified test partition for model evaluation
│-- model/
    │-- train_models.py     # End to end model training & hyperparameter search
    │-- preprocessor.joblib # ColumnTransformer (Median imputer + Scaler + OHE)
    │-- model_logistic_regression.joblib
    │-- model_decision_tree.joblib
    │-- model_knn.joblib
    │-- model_naive_bayes.joblib
    │-- model_svm.joblib
    │-- metrics_summary.json# Serialized evaluation metrics & confusion matrices
```

---

## d. Models Used

Five classification models were implemented on the exact same dataset using an identical 80/20 stratified split and preprocessing pipeline:
1. **Logistic Regression** (L2 Regularized with balanced class weighting)
2. **Decision Tree Classifier** (Cost complexity pruning via max depth and min samples leaf)
3. **K Nearest Neighbors Classifier (KNN)** (Distance weighted voting)
4. **Naive Bayes Classifier** (Gaussian Naive Bayes with variance smoothing)
5. **Support Vector Machine (SVM)** (Radial Basis Function kernel with probability estimates)

### Model Comparison Table

Evaluation metrics calculated on the 20% stratified test dataset (`test_data.csv`, 9,043 instances):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8364 | 0.9056 | 0.4033 | 0.8299 | 0.5428 | 0.5016 |
| **Decision Tree** | 0.8403 | 0.8553 | 0.4070 | 0.7987 | 0.5392 | 0.4926 |
| **KNN** | **0.8943** | 0.8359 | **0.5746** | 0.3715 | 0.4512 | 0.4072 |
| **Naive Bayes** | 0.8765 | 0.8441 | 0.4737 | 0.5028 | 0.4878 | 0.4179 |
| **Support Vector Machine** | 0.8380 | **0.9230** | 0.4103 | **0.8800** | **0.5597** | **0.5282** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Strong baseline linear classifier with balanced class weights. Achieved high Recall (82.99%) and competitive AUC (0.9056), effectively minimizing false negatives under class imbalance. |
| **Decision Tree** | Accurately identifies hierarchical non linear interactions between duration, poutcome, and economic factors. Achieved 84.03% Accuracy and 0.5392 F1, but exhibits slightly lower AUC (0.8553) due to rigid axis aligned boundaries. |
| **KNN** | Produced highest Precision (57.46%) and highest overall Accuracy (89.43%), but suffered significantly in Recall (37.15%) due to metric sparsity in the 51 dimensional one hot encoded feature space. |
| **Naive Bayes** | Offers fastest training and inference with decent AUC (0.8441). However, the conditional feature independence assumption causes miscalibrated probabilities when correlated financial indicators are present, yielding a lower F1 score (0.4878). |
| **Support Vector Machine** | Top overall classifier. The RBF kernel mapped features into a non linear margin space, capturing subtle boundaries between deposit subscribers and non subscribers. Achieved the highest AUC (0.9230), highest Recall (88.00%), highest F1 Score (0.5597), and highest MCC (0.5282). |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** is the clear overall winner. It achieved the highest AUC (0.9230), highest F1 Score (0.5597), and highest MCC (0.5282). In bank telemarketing, detecting actual positive subscribers (high recall) while maintaining strong discriminative power (AUC) is critical, making SVM the optimal operational model. |

---

## How to Run the Streamlit Application Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/msarathrahul/bank_marketing.git
   cd bank_marketing
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
