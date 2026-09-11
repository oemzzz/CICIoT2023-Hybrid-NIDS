# CICIoT2023 Hybrid NIDS

A two-stage hybrid Network Intrusion Detection System (NIDS) developed to detect cyberattacks in IoT networks.

Built as my graduation thesis in Computer Engineering at Bilecik Seyh Edebali University, this project combines the feature-extraction power of deep learning with the classification capability of machine learning.

---

## What Does the Project Do?

* **Dataset:** A balanced 20% subset of the CICIoT2023 dataset (34 distinct classes).
* **Architecture:**
  1. **Stage 1:** A multi-scale **1D-CNN** that extracts 128-dimensional deep features from raw network traffic.
  2. **Stage 2:** These extracted features are combined with the raw data ($46 + 128 = 174$ dimensions) and fed into **XGBoost**, **Random Forest**, and **LightGBM** models.
* **Optimization:** Custom sample weighting (`sample_weight`) was applied to the training set to address class imbalance.

---

## Summary Results (Macro-Average)

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Base)** | 0.993 | 0.803 | 0.805 | 0.800 |
| **XGBoost (Hybrid)** | 0.993 | 0.791 | 0.803 | 0.794 |
| **Random Forest (Base)** | 0.992 | 0.891 | 0.751 | 0.779 |
| **Random Forest (Hybrid)** | 0.990 | 0.893 | 0.732 | 0.762 |
| **LightGBM (Base)** | 0.992 | 0.764 | 0.742 | 0.745 |
| **LightGBM (Hybrid)** | 0.993 | 0.780 | 0.742 | 0.751 |
| **Ensemble (Majority Vote)** | **0.993** | **0.843** | **0.761** | **0.784** |
