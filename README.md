# Loan-Credit-Risk-Analysis-and-Prediction
This project, Loan Credit Risk Prediction, was created as part of the final project for the Project-Based Internship at Rakamin Academy x ID/X Partners.

## Problem Statement
Lending companies (multifinance) face significant challenges in accurately assessing the credit risk of each loan applicant. Two critical decisions often involve different risks:

- Approving loans for applicants who are unlikely to repay, which can result in increased bad debt and substantial financial losses.

- Rejecting loans for applicants who are actually creditworthy, leading to missed business opportunities and reduced profitability.

These challenges highlight the urgent need for a reliable and data-driven credit risk prediction system to support better decision-making and minimize both types of risk.

## Company Goals
1. Approve loans for applicants who are likely to repay on time.

2. Decline loans for applicants who have a high risk of not paying back.

## Objectives
1. Build a machine learning model to predict credit risk based on historical data from loan
 applicants.

2. Predict whether an applicant is a good or bad borrower.

3. Identify the factors that indicate a borrower is high risk.

## Dataset
This dataset contains credit loan information from a lending company from 2007 to 2014.

## Data Understanding
1. The dataset consists of 75 columns (features) and 466,285 rows (data entries).
2. The dataset has been checked for duplicates, and the result shows zero duplicate records.
3. A descriptive statistical analysis was performed on all numerical features to understand the data distribution, minimum and maximum values, mean, and standard deviation spread.
4. I grouped **loan_status** column into two main labels: "good loaner" for borrowers with a good repayment history, and "bad loaner" for those who are at high risk of default. This step is essential to convert the target variable into a binary format suitable for a machine learning classification model.

5. gambar chart loan status

After grouping loan statuses into good loaner and bad loaner, the chart shows that the majority of borrowers (around 88.4%) fall into the good loaner category, while only 11.6% are classified as bad loaner. This indicates that the dataset is imbalanced, which may affect the performance of classification models and should be addressed during modeling—possibly through techniques like oversampling or undersampling.
5. To gain a deeper understanding of the loan applicants, I created visualizations that answer three key questions: 

a. Who applies for credit?
image

b. Where the borrowers are domiciled?
image

c. Why they apply for loans?
image

These insights are crucial in identifying patterns and trends among borrowers, which can help improve the accuracy of credit risk predictions by revealing demographic and behavioral factors associated with loan applications.

## Exploratory Data Analysis

### 1. Categorical Feature

Several categorical columns such as emp_title (205.475), url (466.285), desc (124.435), title (63.098), and zip_code (888) contain a large number of unique values.

### 2. Missing Value
- There are 40 columns in the dataset that contain missing values.
- There are 21 columns with more than 50% missing values in the dataset. To keep the data clean and reliable, these columns were removed. This helps make the analysis more accurate and avoids problems caused by too much missing data.

### 3. Handling Missing Value
- Several columns such as Unnamed: 0, id, member_id, url, grade,emp_title, sub_grade, home_ownership, verification_status, purpose, pymnt_plan, title, zip_code, addr_state, policy_code, application_type, initial_list_status, funded_amnt_inv, total_pymnt, 
total_pymnt_inv were removed because they do not provide meaningful contributions to the core objective of predicting credit risk. These features are either identifiers, redundant, high-cardinality with low predictive power, or not relevant to model training.

- Several columns were removed due to their limited relevance or potential negative impact on the model:

    a. next_pymnt_d: Only applies to active loans, making it irrelevant for historical prediction.
    
    b. last_pymnt_d: Risk of data leakage as it reflects post-loan behavior.
    
    c. last_credit_pull_d: Shows minimal influence on the model's predictive capability.

    d. tot_coll_amt, tot_cur_bal, total_rev_hi_lim: These features contain excessive missing values, making reliable imputation impractical and potentially harmful to model accuracy.

- Several columns with missing values were retained because reasonable imputation is still possible:

    a. emp_length: Missing value can be filled with "Unknown" since it's a categorical variable.
    
    b. revol_util: Missing value can be filled with the median value as it represents a percentage of credit utilization.
    
    c. collections_12_mths_ex_med: Missing value can be filled with 0 based on the majority of the data.

### 4. Converting Column
- The columns containing date-related information (issue_d, earliest_cr_line, last_pymnt_d, next_pymnt_d, and last_credit_pull_d) were converted to datetime format.

- I transformed the loan_status column into a binary numerical format by creating a new column called good_bad_loaner, where "good loaner" is labeled as 0 and "bad loaner" as 1. I removed any rows with undefined status and converted the column into integer type.

### 5. Heatmap

image

The correlation matrix reveals several key relationships between features. The loan amount and funded amount have a perfect correlation (1.00), indicating they are nearly identical. Similarly, installment and loan amount show a strong correlation (0.95), suggesting that higher loans result in higher monthly payments. Additionally, out_prncp and out_prncp_inv also have a perfect correlation (1.00), as both represent the outstanding principal. Meanwhile, the good/bad loaner label shows weak correlations with other features, indicating that loan default risk is influenced by multiple factors beyond those captured in this dataset. These insights are crucial for refining feature selection in predictive modeling.

### 6. Data Distribution

image
Based on the histogram, the columns int_rate and dti exhibit a distribution that is close to normal. This suggests that these features may follow a typical bell-shaped curve, which could be useful for statistical analysis and modeling.

### 7. Univariate Analysis

image
Based on the analysis, we can see the following:

- The graph indicates a highly left skewed distribution, with a significantly tall peak at the "good" value. This suggests that the majority of loans in the dataset are categorized as good loans.
- There is a big imbalance between good and bad loans, which could make it harder to build an accurate risk prediction model.
- Since there are very few bad loans, the model might struggle to learn their patterns, making it less effective at predicting loan risks.

### 8. Bivariate Analysis

image
From the bivariate analysis, we can see the following: 
- The graph clearly shows that loans categorized as "bad" have a higher average interest rate (15.90%) compared to "good" loans which have an average interest rate of 13.74%.

## Data Preparation

### 1. Handling Outliers

image
Based on the boxplot above, several columns contain extreme outliers. To reduce these outliers, a transformation will be applied to minimize their impact.

### 2. Log Transformation

image
With the application of Log Transformation, the distribution of the annual_inc column appears more normal, making it ready to be used as a feature in the model. However, some columns, such as total_rec_late_fee, collections_12_mths_ex_med, delinq_2yrs, and inq_last_6mths, still have high peaks, likely due to many borrowers having no late payments or outstanding collections.

### 3. Scaling Data
I applied standardization to selected numerical features using StandardScaler, which transforms the data to have a mean of 0 and standard deviation of 1.
The following columns were scaled: loan_amnt,term, int_rate, installment, emp_length, annual_inc, dti, delinq_2yrs, inq_last_6mths, revol_util, total_acc, total_rec_late_fee, and collections_12_mths_ex_med.

### 4. Selection Feature
I selected a subset of features that are most relevant for analyzing and modeling loan credit risk. The selected columns include: loan_amnt, term, int_rate, installment, emp_length, annual_inc, dti, delinq_2yrs, inq_last_6mths, revol_util, total_acc, total_rec_late_fee, collections_12_mths_ex_med, and good_bad_loaner.

### 5. SMOTE

image
- Before SMOTE:
a. The "Good Loaner" class is highly dominant (95.4%).
b. The "Bad Loaner" class is very small (4.6%).

 - After SMOTE:
 a. Both classes are balanced (50% each).
 b. SMOTE successfully created new "Bad Loaner" data.

### 6. Spliting Data
I split the dataset into training and testing sets using an 80:20 ratio to ensure the model is trained on a majority of the data while retaining a portion for evaluation. After applying resampling, the training set consists of 336,254 samples, and the testing set contains 84,064 samples.

## Data Modeling
I used six different machine learning models to evaluate and compare their performance in classifying loan risk. These models include:

1. Logistic Regression
2. Random Forest
3. Decision Tree
4. K-Nearest Neighbors (KNN)
5. Gaussian Naive Bayes
6. Stochastic Gradient Descent (SGD)

This variety of models helps ensure a comprehensive understanding of which algorithm performs best for the loan credit risk classification task.

## Evaluation

image
The K-Nearest Neighbors (KNN) model performs the best compared to other models. It has the highest Score (88.78), ROC-AUC (96.18), and Cross-Validation Score (78.14). This means KNN is very good at making predictions and distinguishing between classes.

## Confusion Matrix

image
Based on the confusion matrix, K-Nearest Neighbors (KNN) outperforms other models with the highest accuracy in predicting both positive and negative classes. It records the highest True Positives (41,960) and high True Negatives (32,674), with only 72 False Negatives, which means it almost perfectly identifies high-risk (Bad) loans. 

## Conclusion
K-Nearest Neighbors (KNN) is the best model for loan classification because it has the highest accuracy (88.78%), meaning it correctly predicts most loans. It also has the highest AUC-ROC score (96.18%), showing that it can effectively distinguish between "good" and "bad" loans. The model is stable, with a cross-validation score of 78.14%, ensuring it performs well on different data samples. Although its precision (81.76%) is slightly lower than Naïve Bayes, KNN has an extremely high recall (99.83%). In credit risk assessment, recall is very important because it helps reduce the risk of misclassifying "bad" loans as "good."

## Business Solution
1. Build an Automated Credit Scoring System

Use machine learning models to automatically assess loan risk. This can speed up the loan approval process and reduce human error from manual evaluations.

2. Classifying Borrowers by Risk Level

Use prediction results to classify borrowers into risk categories, such as low, medium, and high. Each group can then receive different treatment, like higher interest rates for high-risk borrowers.

3. Optimize Collection Strategies

By identifying borrowers who are more likely to default, the company can focus its reminders or collection efforts on these groups early, reducing potential losses.


