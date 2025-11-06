# Predicting Heart Disease Risk Using Machine Learning  
### *Turning Health Data into Life-Saving Predictions*  
**Team Matrix | Capstone Project**

*A machine learning project for predicting heart disease risk using patient health indicators.*

---

## Overview  
Heart disease remains a leading cause of mortality worldwide. Early detection through data-driven modeling can help clinicians identify high-risk patients and intervene early.  
This project presents a **data-driven machine learning approach** for predicting heart disease risk using non-invasive medical data as **age, cholesterol, resting blood pressure, maximum heart rate,** and **chest pain type**.  
By combining **exploratory data analysis (EDA)**, **predictive modeling**, and **clinical interpretation**, the team developed interpretable models that can assist healthcare professionals in identifying high-risk patients more accurately and efficiently.

---

## Objectives
- Perform **comprehensive EDA** to uncover the most influential factors associated with heart disease.  
- Train and compare multiple **classification algorithms** to predict disease likelihood.  
- Optimize the top models through **hyperparameter tuning** and **cross-validation**.  
- Translate results into **clinically interpretable insights** that align with medical understanding.  
- Save and package the final models for potential **deployment** in clinical decision systems.


The work involves:
- Data cleaning and preprocessing  
- Exploratory data analysis (EDA) to uncover clinical patterns  
- Model building, tuning, and evaluation  
- Clinical interpretation of predictive features  
- Deployment-ready model saving  

---

## Project Workflow  

### **1. Data Cleaning**
- Verified dataset integrity: no missing, or invalid entries.
- - Duplicate records were identified and removed to ensure each patient represented a unique observation.
- Feature names were renamed for clarity and consistency.
- Checked for biologically impossible values (e.g., zero cholesterol, extreme blood pressure).  
- Created two data versions:
  - `df_eda` – for human-readable visualization.  
  - `df` – encoded and ready for machine learning.  

***Outcome:*** A clinically valid, consistent, and analysis-ready dataset.

### **2. Exploratory Data Analysis (EDA)**
EDA explored the relationships between key clinical variables and heart disease outcomes.

**Highlights:**
- **Age & Sex:** Risk increases after age 40, with males showing higher prevalence.  
- **Chest Pain Type:** Strongest categorical predictor; non-angina highly associated with heart disease.  
- **Maximum Heart Rate (thalach):** Lower rates corresponded to higher disease likelihood.  
- **ST Depression (oldpeak):** Strong indicator of reduced blood flow (ischemia).  
- **Exercise-Induced Angina (exang):** Elevated risk among patients with positive angina responses.  

Visual tools such as boxplots, correlation matrices, and density plots provided interpretable and clinically aligned insights.

### **3. Model Development and Evaluation**
Four classification algorithms were initially trained and compared:
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **XGBoost Classifier**

After initial evaluation, **XGBoost was dropped** due to low performance, **unstable validation results** and a tendency to **overfit** during baseline testing. 
The remaining three models were selected for deeper optimization and interpretation:
-  **Logistic Regression (Primary Model)** – Best interpretability, recall, and clinical reliability.  
-  **Random Forest (Supporting Model)** – Strong secondary validation, confirming key feature stability.  
-  **Decision Tree** – Retained for comparison and interpretability benchmarking, though less generalizable.

### **4. Model Evaluation Process**
Each model was trained and tested using an 80/20 data split, with metrics including:
- Accuracy  
- Precision, Recall, and F1-score  
- ROC-AUC  
- Confusion Matrix  
- Learning Curves (to evaluate bias vs. variance)

| Model                       | Accuracy | Precision | Recall     | F1-Score   | ROC-AUC    |
| --------------------------- | -------- | --------- | ---------- | ---------- | ---------- |
| Logistic Regression (Tuned) | 0.8033   | 0.7838    | **0.8788** | **0.8286** | **0.8810** |
| Decision Tree (Tuned)       | 0.7213   | 0.7500    | 0.7273     | 0.7385     | 0.7960     |
| Random Forest (Tuned)       | 0.8033   | 0.8000    | 0.8485     | 0.8235     | 0.8734     |


**After baseline testing**, the top three models were tuned using **GridSearchCV** for optimal hyperparameter settings.  

### **5. Final Models and Clinical Insights**
After tuning:
- **Logistic Regression** achieved the best overall recall (0.8788) and ROC-AUC (0.8810).  
- **Random Forest** offered strong secondary performance with consistent generalization.  
- **Decision Tree** remained interpretable but showed weaker test generalization.

**Feature Importance / Coefficients revealed:**

| Feature | Clinical Meaning | Influence |
|----------|------------------|------------|
| Chest Pain Type | Angina-type chest pain and non-angina linked to restricted blood flow | Strong positive |
| Maximum Heart Rate | Indicator of cardiovascular efficiency | Negative correlation |
| ST Depression | ECG-based ischemia measure | Strong positive |
| Major Vessels Count | Extent of arterial blockage | Positive |
| Thalassemia | Genetic blood abnormality affecting oxygen flow | Moderate positive |

***Clinical relevance:*** These findings align with established cardiology insights, validating the models’ interpretability and medical reliability.

### **6. Model Saving and Deployment Readiness**
The final tuned models and preprocessing scaler were serialized using `joblib` for potential deployment.

---

## **Final Model Chosen:** Logistic Regression (Primary), Random Forest (Supporting)

---

## **Key Insights & Findings**

- Early detection through ML can significantly improve heart disease risk assessment.
- Logistic Regression balances interpretability and accuracy; vital for medical decisions.
- Random Forest validates and enhances feature reliability through ensemble averaging.
- Functional indicators (e.g., chest pain, ST depression, heart rate) outweigh demographics (e.g., gender, age) in predictive power.
- Minimizing false negatives ensures fewer missed diagnoses, aligning with clinical priorities.

---

## Tech Stack  
- **Language:** Python 3  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, joblib  
- **Environment:** Jupyter Notebook / Anaconda  
- **Version Control:** Git & GitHub  

---

## Conclusion

This project demonstrates how machine learning can support clinical decision-making by accurately predicting heart disease risk using non-invasive data.
The selected Logistic Regression model combines interpretability, reliability, and strong recall performance, making it well-suited for integration into real-world healthcare systems.
When supported by Random Forest, it provides a powerful balance between transparency and predictive strength; a foundation for future AI-assisted medical tools.

---

## Medium Article
Read our full story here: [How We Used Machine Learning to Predict Heart Disease](<your_medium_link>)

