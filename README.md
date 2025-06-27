# Banking_DM_DATASCIENCE
This GitHub repository will serve as a home for my personal projects, explorations, and contributions to the banking direct marketing data science community. I look forward to connecting with fellow data enthusiasts and collaborating on exciting challenges!

-----

# Predicting Customer Conversion for Bank Term Deposits

## 1\. Project Overview

This project develops a machine learning model to predict whether a client will subscribe to a term deposit during a direct marketing campaign. The model aims to improve the efficiency and return on investment (ROI) of marketing efforts by identifying clients with the highest propensity to convert, allowing the bank to target its resources more effectively. 

The project follows an end-to-end machine learning lifecycle, including:

  * In-depth Exploratory Data Analysis (EDA)
  * Robust Feature Engineering and Preprocessing
  * A "bake-off" between multiple model architectures
  * Critical diagnosis and correction of a data leakage issue
  * Final model selection, evaluation, and strategic recommendations

-----

## 2\. The Business Problem

The bank's marketing division invests significant resources into campaigns to attract term deposit subscriptions. To maximize the effectiveness of these campaigns, it is crucial to move beyond traditional selection methods and adopt a data-driven approach to identify clients who are most likely to subscribe. This model provides a "propensity score" for each client, enabling tailored marketing and reduced spend on low-propensity leads.

### 2.1\. The Dataset used

The dataset used for this analysis originates from a series of direct marketing campaigns conducted by a Portuguese banking institution. These campaigns, primarily based on telephonic marketing, aimed to increase subscriptions to term deposits, which are a crucial source of revenue for the bank. The data captures 17 client attributes and campaign interaction details to predict the final outcome: whether a client subscribed to a term deposit (y).

### Source and Citation

This dataset is a well-regarded benchmark in the data science community and is publicly available from the UCI Machine Learning Repository. Its use and analysis are detailed in the following academic paper, which should be cited in any further work:

•	Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, 62, 22-31

-----

## 3\. Exploratory Data Analysis (EDA) - Key Insights

Analysis of the client and campaign data revealed several key predictors of conversion:

**Class Imbalance:** The dataset is highly imbalanced, with non-subscribers outnumbering subscribers by approximately 8 to 1. This requires the use of robust evaluation metrics like ROC-AUC and techniques like class weighting. 
  * **Contact Duration:** Call duration is a powerful indicator of client engagement. Clients who subscribed had significantly longer conversations. *(Note: This feature was later removed from the final model to prevent data leakage, as duration is only known after a call is completed)*. 
  * **Previous Campaign Success:** Clients who had previously subscribed to an offer were exceptionally likely to convert again. 
  * **Seasonality:** Conversion rates peaked during the spring and autumn months (March, April, September, October, December). 
  * **Contact Method:** Campaigns conducted via 'cellular' contact were more successful than those using 'telephone'. 
  * **Number of Contacts:** The conversion rate was highest on the first contact and dropped sharply with subsequent contacts, indicating diminishing returns.
  * **Client Demographics:** 'Students' and 'retired' clients showed the highest conversion rates, while those without existing housing or personal loans were also more likely to subscribe, suggesting greater financial flexibility.

-----

## 4\. Feature Engineering

To capture more complex patterns, several new features were engineered from the original dataset:

| Feature Name | Calculation / Logic | Hypothesis |
| :--- | :--- | :--- |
| `duration_per_contact` | `duration / campaign` |Measures the average engagement level per interaction.  |
| `balance_to_age_ratio` | `balance / age` | A proxy for wealth accumulation relative to a client's life stage. |
| `total_liabilities` | Combines `housing` and `loan` | The type and number of loans may have a more nuanced effect than each loan individually. |
| `is_student_or_retired` | Binary flag for job type | Explicitly flags the two highest-converting demographic groups. |
| `is_high_season` | Binary flag for peak months | Simplifies the strong seasonality effect into a clear signal. |
| `was_previously_contacted` | Binary flag if `pdays != -1` | Separates clients with a prior relationship from those being contacted for the first time. |

-----

## 5\. A Critical Lesson: Data Leakage Diagnosis and Correction

A crucial phase of this project involved identifying and correcting a severe data leakage issue.

### 5.1. The Problem

Initial model evaluations showed perfect scores (ROC-AUC of 1.0) on both the training and test sets. This unrealistic performance is a classic symptom of data leakage, where the model is trained on information that would not be available at the time of prediction.

### 5.2. The Investigation

A diagnostic script revealed a critical flaw in the initial data setup: **100% of the test set rows were also present in the training set**. The `data_leakage_checker.py`  script was the one thatrevealed that 100% of the test set rows were also present in the training set, invalidating all initial results.

### 5.3. The Solution

The following corrective action was taken:

1.  **Discarded Splits:** All existing datasets were discarded. 
2.  **Stratified Resampling:** A new, clean 80/20 train/test split was created from the original, complete dataset. Crucially, this split was **stratified** by the target variable (`y`) to ensure the proportion of subscribers and non-subscribers was identical in both sets, which is essential for an imbalanced dataset.
     - The `correct_train_test_split.py` script was used to perform a new, methodologically sound 80/20 split on the original dataset.  
4.  **Verification:** A final check confirmed that the systemic data leakage was resolved, with only a negligible 0.07% overlap remaining (likely due to legitimate duplicates in the raw data).
     - The `data_leakage_checker.py` was rerun to verify

This rigorous debugging and correction process was fundamental to building a valid and trustworthy model.

-----

## 6\. Model Selection: A "Bake-Off"

After correcting the data leakage and removing the `duration` feature, a comparative analysis was performed using 5-fold cross-validation to select the best model architecture. Three candidates were evaluated:

1.  **Logistic Regression (with WOE transformation):** A highly interpretable linear model.
2.  **Random Forest:** An ensemble model known for its robustness.
3.  **Gradient Boosting:** An advanced sequential ensemble model.

### Cross-Validation Results (ROC-AUC)

**The Gradient Boosting model emerged as the champion**, achieving the highest mean ROC-AUC score during cross-validation.

-----

## 7\. Final Model Performance (Gradient Boosting)

The champion Gradient Boosting model was trained on the full, clean training set and evaluated on the unseen test set.

### Performance Metrics

| Dataset | ROC-AUC | PR-AUC | Precision | Recall | F1-Score | Gini |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 0.8040 | 0.4781 | 0.6946 | 0.2333 | 0.3493 | 0.6080 |
| **Test** | **0.7985** | **0.4566** | **0.6676** | **0.2278** | **0.3397** | **0.5970** |


### Key Findings:

  * **Strong Predictive Power:** The model achieved a **Gini coefficient of 0.5970** on the unseen test data, which is considered a strong result for a marketing propensity model.
  * **Excellent Generalization:** The minimal drop in performance between the train and test sets confirms that the **model is not overfitting** and will perform reliably on new data.

### Diagnostic Plots

  * **Calibration:** The model's probabilities are well-calibrated, meaning a predicted 40% probability of conversion corresponds to an actual 40% conversion rate for that group. 
  * **Threshold Analysis:** The threshold plot is a critical tool for business strategy. The marketing team can select a probability threshold to balance the trade-off between Precision (minimizing cost) and Recall (maximizing captured customers). For instance, a lower threshold (e.g., 0.20) would capture more potential customers at the cost of contacting more non-converters.

-----

## 8\. Strategic Recommendations & Conclusion

This project successfully delivered two production-ready models:

1.  **For Maximum Predictive Power: Gradient Boosting**

      * This model provides the highest accuracy (Gini of \~0.60) and should be used when the primary goal is to maximize campaign ROI by generating the most accurate list of high-propensity clients.

2.  **For Transparency and Explainability: Logistic Regression (WOE)**

      * While slightly less accurate (Gini of \~0.54), this model is fully transparent. Its scorecard-based nature allows for clear explanations of any decision, which is invaluable for regulatory compliance (Explainable AI) and building trust with stakeholders. 

The bank is now equipped with a powerful and flexible data-driven toolkit to significantly enhance its marketing effectiveness and achieve a higher return on investment.

-----

## 9\. How to Run This Project

1.  **Run the notebooks/scripts in order:**

      * `01_Exploratory_Data_Analysis.py`
      * `02_feature_engineering.py`
      * `03_woe_binning.py`
      * `04_model_comparison.py`
      * `05_final_model_evaluation_gb.py`/ `06_final_model_evaluation_lr.py`/ `07_final_model_evaluation_rf.py`
  
