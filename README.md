# 💳 Credit Card Fraud Detection using Machine Learning

## Project Overview

This project focuses on developing a robust machine learning model to accurately detect fraudulent credit card transactions, thereby helping financial institutions mitigate financial losses and enhance customer trust.

The core challenge lies in the highly **imbalanced nature** of the dataset, where fraudulent transactions are a rare event. The goal is to build a model that maximizes the detection of true fraud cases (high recall) while minimizing the flagging of legitimate transactions as fraudulent (high precision and low false positives).

## Table of Contents
1. [Project Objective](#project-objective)
2. [Data Dictionary](#data-dictionary)
3. [Setup and Installation](#setup-and-installation)
4. [Methodology](#methodology)
    - [Data Preprocessing](#data-preprocessing)
    - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
    - [Feature Engineering](#feature-engineering)
5. [Model Training and Results](#model-training-and-results)
    - [Model Comparison](#model-comparison)
6. [Conclusion and Recommendations](#conclusion-and-recommendations)

---

## 🎯 Project Objective

To build and deploy a machine learning model capable of accurately predicting fraudulent credit card transactions, with an emphasis on **maximizing fraud detection (Recall)** while **minimizing false positives** to avoid disrupting legitimate customer activities.

## 🗃️ Data Dictionary

The model uses the following features to classify a transaction as legitimate (0) or fraudulent (1):

| Column Name | Description |
|---|---|
| `distance_from_home` | Distance from the cardholder's home to the transaction location |
| `distance_from_last_transaction` | Distance from the previous transaction |
| `ratio_to_median_purchase_price` | Ratio of transaction value to the median purchase price |
| `repeat_retailer` | Whether the transaction was from a repeat retailer (1: Yes, 0: No) |
| `used_chip` | Whether a chip was used in the transaction (1: Yes, 0: No) |
| `used_pin_number` | Whether a PIN number was used (1: Yes, 0: No) |
| `online_order` | Whether the transaction was online (1: Yes, 0: No) |
| **`fraud`** | **Target variable:** Indicates if the transaction was fraudulent (1: Fraud, 0: Legitimate) |

---

## 🛠️ Setup and Installation

### Prerequisites
* Python 3.x

### Dependencies

This project requires the following libraries, as listed in `requirements.txt`:
```bash
pandas
numpy
scikit-learn
streamlit
joblib
matplotlib
seaborn
```

### Run the interactive demo

From the project directory, install the dependencies and launch the Streamlit interface:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app has four sections: **Project overview**, **Data exploration**, **Model evaluation**, and
**Try a transaction**. It reads `card_transdata.csv` for dataset counts and descriptive statistics,
shows the saved notebook charts and test-set model metrics, and loads `model.pkl` and `scaler.pkl`
for interactive predictions. The transaction analyzer accepts the seven source fields, derives the
three engineered features, and scales the resulting ten features before prediction.
The notebook also caps three numeric columns using IQR bounds computed from the full dataset, but
those bounds were not saved. The app cannot reproduce that capping step exactly. For a faithful
serving pipeline, save the training-time bounds (or fit capping and scaling inside a single
scikit-learn pipeline) and apply the same pipeline in the app.

## Data Preprocessing  

### Missing Data  
We checked for missing values using `df.isnull().sum()`. Missing data was imputed or removed based on its significance.

### Outlier Detection and Handling  
Outliers were detected using box plots for key columns:  
- `distance_from_home`  
- `distance_from_last_transaction`  
- `ratio_to_median_purchase_price`  

To handle outliers, we capped values using the Interquartile Range (IQR) method via a custom `cap_outliers` function. This ensured that extreme values did not skew the model.

---

## Exploratory Data Analysis (EDA)  
- **Correlation Heatmap:** Plotted to examine relationships between features.  
- **Class Distribution:** Checked the balance of fraudulent vs. legitimate transactions in the `fraud` column.  

---

## Feature Engineering  
1. **Distance Ratio:** Calculated the ratio of `distance_from_home` to `distance_from_last_transaction`.  
2. **Transaction Score:** Combined multiple features to create a transaction risk score.  
3. **Purchase Category:** Grouped transactions based on `ratio_to_median_purchase_price` into categories:  
   - `very_low`, `low`, `medium`, `high`, `very_high`  
4. **Encoding:** Encoded purchase categories using a custom function to make them model-ready.  

---

## Feature Selection, Train-Test Split & Scaling  
- Selected significant features based on correlation analysis.  
- Split the dataset into training and test sets using an 80-20 ratio.  
- Applied **Standard Scaler** for feature normalization.  

---

## Model Training  

### Logistic Regression  
- **Accuracy:** 91.65%  
- **Confusion Matrix:**  
  ```
  [[166976  15581]
   [  1124  16319]]
  ```  
- **Classification Report:**  
  | Metric        | 0 (Legit) | 1 (Fraud) |  
  |---------------|------------|-----------|  
  | Precision     | 0.99       | 0.51      |  
  | Recall        | 0.91       | 0.94      |  
  | F1-Score      | 0.95       | 0.66      |  

---

### Random Forest  
- **Accuracy:** 94.27%  
- **Confusion Matrix:**  
  ```
  [[171253  11304]
   [   153  17290]]
  ```  
- **Classification Report:**  
  | Metric        | 0 (Legit) | 1 (Fraud) |  
  |---------------|------------|-----------|  
  | Precision     | 1.00       | 0.60      |  
  | Recall        | 0.94       | 0.99      |  
  | F1-Score      | 0.97       | 0.75      |  

---

## Model Comparison  
We compared both models using accuracy, precision, recall, and F1-score metrics, summarized in a DataFrame for easy reference.
The Random Forest model was selected for deployment due to its significantly higher Recall (0.99) and overall Accuracy (94.27%) and F1-Score (0.75), making it better at identifying the minority class (fraud).


---

## Conclusion  
Our credit card fraud detection model demonstrated a significant ability to distinguish between fraudulent and legitimate transactions. By leveraging machine learning techniques such as Logistic Regression and Random Forest, we developed a robust solution capable of identifying fraudulent transactions with high accuracy.

### Key Insights:  
- **Fraud Detection Efficiency:** The model consistently identified fraudulent transactions with minimal false positives, reducing potential financial losses for credit card companies and enhancing customer trust.  
- **Impact on Fraud Prevention:** By accurately detecting fraudulent transactions, financial institutions can minimize disruptions to legitimate transactions, improving the overall user experience.  
- **High-Risk Factors:** Features like unusually high transaction distances and online orders showed strong correlations with fraud, offering valuable indicators for real-time monitoring systems.

### Recommendations:  
1. **Real-Time Monitoring:** Implementing this model in a real-time transaction monitoring system can help flag high-risk transactions promptly, reducing fraud-related losses.  
2. **Continuous Model Updating:** Regular updates with new transaction data will ensure the model adapts to emerging fraud patterns.  
3. **Customer Awareness:** Educating customers about common fraud triggers, such as unusual transaction patterns and PIN security, can further strengthen fraud prevention.  
4. **Integration with Multi-Layer Security:** This model should complement other security measures, such as two-factor authentication, to enhance fraud detection and prevention.

By integrating this machine learning-based solution into financial systems, organizations can strengthen their defenses against fraud, ensuring both security and a seamless customer experience.
