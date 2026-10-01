# Ransomware Detection Using Generative Adaptive Learning

A final-year Computer Science and Engineering project focused on detecting ransomware using machine learning and Generative AI-based analysis.

## 📌 Project Overview

Ransomware is a type of malicious software that can restrict access to files or systems and demand payment from victims.

Traditional signature-based detection methods can struggle with previously unseen or evolving ransomware variants. This project explores a machine-learning-based approach for identifying suspicious executable files using extracted features.

The project also aims to integrate Generative AI to provide human-readable explanations of detected threats and support intelligent risk assessment.

---

## 🎯 Objectives

- Detect ransomware using machine-learning techniques.
- Analyze executable-file features for classification.
- Compare Decision Tree and Random Forest models.
- Evaluate the models using standard classification metrics.
- Provide meaningful explanations of detected threats using Generative AI.
- Develop a system that can support timely ransomware detection and alert generation.

---

## 🧠 Machine Learning Models

The current machine-learning module uses:

- Decision Tree Classifier
- Random Forest Classifier

The models are trained and evaluated using a ransomware detection dataset containing executable-file features.

### Current Features

The ML model currently uses 15 features:

- Machine
- DebugSize
- DebugRVA
- MajorImageVersion
- MajorOSVersion
- ExportRVA
- ExportSize
- IatVRA
- MajorLinkerVersion
- MinorLinkerVersion
- NumberOfSections
- SizeOfStackReserve
- DllCharacteristics
- ResourceSize
- BitcoinAddresses

`FileName` and `md5Hash` are treated as identifiers and are not used as model features.

---

## 📊 Dataset

The project currently uses the following Kaggle dataset:

**Ransomware Detection Data Set**

https://www.kaggle.com/datasets/amdj3dax/ransomware-detection-data-set

The dataset contains executable-file metadata/features used for binary classification.

### Dataset preprocessing

The current preprocessing pipeline:

1. Loads the dataset.
2. Removes identifier columns.
3. Checks missing values.
4. Removes duplicate records.
5. Separates features and target.
6. Splits the dataset into training and testing sets.
7. Uses stratified splitting to preserve class distribution.

### Current dataset statistics

| Item | Value |
|---|---:|
| Original records | 62,485 |
| Duplicate records removed | 30,229 |
| Final records | 32,256 |
| Number of ML features | 15 |
| Training records | 25,804 |
| Testing records | 6,452 |
| Missing values | 0 |

---

## 📈 Current Model Results

The models were evaluated on the held-out test set.

| Metric | Decision Tree | Random Forest |
|---|---:|---:|
| Accuracy | 98.7911% | 99.3490% |
| Precision | 97.9060% | 98.8345% |
| Recall | 98.4558% | 99.2045% |
| F1-score | 98.1801% | 99.0191% |
| False Positive Rate | 1.5442% | 0.7955% |
| Prediction Time | 0.01967 s | 0.04331 s |

These results are based on the current experimental setup and test dataset.

---

## 🗂️ Project Structure

```text
Ransomware-Detection/
│
├── data/
│   ├── data_file.csv
│   └── test_data.csv
│
├── models/
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
├── results/
│   └── model_comparison.csv
│
├── src/
│   ├── 01_data_analysis.py
│   ├── 02_preprocessing.py
│   ├── 03_train_models.py
│   ├── 04_evaluate_models.py
│   └── 05_predict.py
│
├── .gitignore
├── requirements.txt
└── README.md