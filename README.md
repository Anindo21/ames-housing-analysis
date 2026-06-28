# Ames Housing Layout Analytics: Age Regression & Architectural Inference

This project leverages the classic **Ames Housing Dataset** to explore residential architecture through machine learning, completely independent of market valuations or price metrics. It features two distinct machine learning tasks:
1. **Regression:** Predicting a home's original construction year based on structural materials and utilities.
2. **Multi-Class Classification:** Reverse-engineering and classifying architectural typologies using purely spatial footprints and internal layouts.

---

## 🚀 Project Overview

### 1. Regression Focus: Predicting the Age of a House
* **Research Question:** How accurately can a machine learning model predict the original construction year of a residential property based solely on its current structural materials, layout configurations, and utility quality metrics?
* **The Angle:** Housing designs evolve over generations. Older homes typically feature segmented, closed-off layouts optimized for localized heating, while modern homes favor sweeping, open-concept configurations. This model attempts to map these physical, historical design shifts directly to a chronological timeline.
* **Practical Benefit:** * **Data Integrity:** Serves as an automated data-cleaning tool for municipal planners or real estate platforms to accurately fill in missing or corrupted historical city records.
  * **Value Detection:** Identifies "hidden gems"—older structures whose modernized interior metrics cause the model to flag them as brand-new builds.

### 2. Multi-Class Classification: Automated Architectural Typology Inference
* **Research Question:** Can machine learning algorithms accurately infer a residential property's architectural typology from its measurable spatial footprint, structural scale metrics, and internal layout configurations alone?
* **The Angle:** By removing price and artificial valuation tiers, this task treats machine learning as an automated architectural classifier. The objective is to discover the exact geometric thresholds (floor square footage, room distribution, story configurations) that cleanly separate single-story, multi-story, 1.5-story, and split-level typologies.

---

## 📊 Dataset
The project utilizes the **Ames Housing Dataset** available on [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset). 

*Note: The raw data file should be placed in the `data/raw/` directory and is excluded from source control.*

---

## 🛠️ Tech Stack & Methods
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, (Optional: LightGBM / XGBoost)
* **Key Techniques:**
  * Categorical encoding for building materials and utility metrics.
  * Feature engineering (e.g., aggregating room areas vs. total footprint).
  * Regression algorithms (e.g., Ridge/Lasso, Random Forest Regressor, Gradient Boosting).
  * Multi-class classification evaluation (Confusion matrices, F1-Score per typology tier).
