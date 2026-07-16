# Predicting Residential Construction Year and Classifying Architectural Typology Using the Ames Housing Dataset

---

## 1. Introduction

The Ames Housing Dataset, compiled by De Cock (2011), documents residential property sales in Ames, Iowa between 2006 and 2010. It contains 2,930 observations and 82 variables covering physical dimensions, material quality ratings, neighbourhood classifications, basement and garage attributes, and sale conditions. While this dataset is most commonly applied to sale price prediction, this study deliberately departs from that convention to explore a different kind of information encoded in a property's physical record: the traces left by its construction era and architectural form.

Two prediction tasks are investigated. The first is a regression problem: estimating the original year of construction from a property's structural characteristics and quality attributes, without access to any temporal metadata or market valuation. The motivation for this question rests on the observation that residential building practice changes measurably over time. Materials, room configurations, the standard inclusion of amenities such as central air conditioning, and the typical scale of living spaces all shifted systematically across the twentieth century. A model capable of recovering construction year from these physical signals would validate the chronological information latent in the data and could serve practical purposes — for example, filling missing or corrupted year fields in municipal property records, or flagging properties whose recorded year of construction appears inconsistent with their measured attributes.

The second task is a four-class classification problem: predicting a property's architectural typology — whether it is single-story, multi-story, one-and-a-half story, or split-level — from its spatial and structural measurements, excluding any temporal or market-based features. This tests whether the floor plan geometry of a house is sufficient to recover the categories that architects and planners use to describe residential form.

Both tasks are treated as independent prediction problems sharing a common preprocessing pipeline constructed to prevent leakage between training and test data.

---

## 2. Exploratory Data Analysis

### 2.1 Target Variable Distributions

**Year Built.** Construction years in the dataset span from 1872 to 2010, with a mean of approximately 1971 and a standard deviation of 30.2 years. The distribution is moderately left-skewed (skewness = −0.604, kurtosis = −0.502), reflecting the historical accumulation of housing stock: relatively few nineteenth-century properties survive in the sample, with the bulk of observations concentrated in the mid-twentieth century and later.

**House Style.** The original dataset records house style using eight specific codes. For modelling purposes, these are consolidated into four typology classes based on the primary structural form. Single-story properties (original code `1Story`) account for 1,481 observations (50.5% of the dataset). Multi-story properties (`2Story`, `2.5Fin`, `2.5Unf`) account for 905 (30.9%). One-and-a-half story properties (`1.5Fin`, `1.5Unf`) account for 333 (11.4%), and split-level properties (`SLvl`, `SFoyer`) account for 211 (7.2%). This distribution is noticeably imbalanced: single-story homes represent half the dataset, while split-level homes are the smallest class. As discussed in Sections 4 and 5, this imbalance influences both model behaviour and the choice of evaluation metric.

### 2.2 Predictor Relationships

Pearson correlations between numerical predictors and construction year were computed to identify features that carry a temporal signal. The strongest positive correlates include Garage Year Built, Overall Quality, and Ground Living Area, reflecting broader historical trends: rising construction standards, expanding dwelling sizes, and the gradual integration of previously optional amenities. Negative correlates include features associated with older architectural conventions, such as enclosed porch area and certain basement finishing configurations.

Scatter plots of the six most strongly correlated predictors confirm these relationships. The near-linear association between Garage Year Built and Year Built is particularly pronounced, as garages were typically constructed alongside the main dwelling. Overall Quality shows a step-pattern consistent with the introduction of higher-grade materials in later decades. Ground Living Area increases more gradually, reflecting the long-term trend towards larger homes.

For the classification task, the geometric and structural variables — room counts, floor areas, basement dimensions — were examined to assess their discriminating power across typology classes. Single-story and multi-story properties show the clearest separation in features such as second-floor area (near zero for single-story properties) and total living area. Split-level and one-and-a-half story properties overlap substantially in many features, which is consistent with their architectural ambiguity and, as discussed later, their relative difficulty to classify correctly.

---

## 3. Data Preprocessing

### 3.1 Initial Data Cleaning

Four columns were removed at the outset because each contained more than 70% missing values: `Alley`, `Pool QC`, `Fence`, and `Misc Feature`. Imputing columns with this level of missingness risks introducing more noise than signal, and the information they capture (alley access, pool quality, fence type, and miscellaneous features) is of low relevance to either target. After this step, the dataset retained 78 variables. Any rows missing either target variable were subsequently removed, leaving 2,930 usable observations.

### 3.2 Feature Engineering and Leakage Prevention

A single derived variable, `Year Remod/Add Delta`, was constructed as the difference between the recorded remodel year and the construction year. This captures how recently a property was renovated relative to when it was built, without retaining the raw remodel year — which correlates directly with the regression target — as an independent predictor. The raw remodel year was then dropped.

Several columns were excluded from the regression feature set on the grounds that they represent identifiers, sale circumstances, or potential leakage rather than physical property attributes. Row indices (`Order`, `PID`) carry no predictive content. Sale timing variables (`Yr Sold`, `Mo Sold`) describe when a transaction occurred rather than any characteristic of the property itself. Sale price (`SalePrice`) was excluded to keep both tasks independent of market valuation. `House Style`, the classification target, was copied to a separate series before being dropped from the shared feature frame, so that it remains available for the classification task without functioning as a predictor in the regression task.

### 3.3 Imputation

Missing values in the remaining features were handled inside the modelling pipeline to ensure that no information from the test set influenced the imputed values. Numerical features were imputed using the column median; categorical features were imputed using the most frequent value. Both strategies are applied within the training fold only.

### 3.4 Encoding and Scaling

Feature types were divided into five groups, each receiving an appropriate transformation based on the distribution and measurement level of the variables.

Sixteen numerical features with approximately symmetric distributions were standardised: a Yeo-Johnson power transform was applied to reduce any residual skew, followed by standardisation to zero mean and unit variance. Eight features with higher skewness were normalised instead, using the same power transform followed by min-max scaling to the [0, 1] interval.

Twenty-one categorical features with a natural ordering — quality and condition ratings such as `Exter Qual`, `Kitchen Qual`, `Heating QC`, and the range of basement and garage condition variables — were encoded using ordinal encoding, with explicit category lists specifying the order from worst to best. Nominal features with fewer than ten distinct values received one-hot encoding. High-cardinality features, including `Neighborhood` and `Exterior 1st`, were encoded using binary encoding to prevent the feature space from expanding excessively.

### 3.5 Pipeline Construction

All transformation steps were assembled into a single `ColumnTransformer` composed of five sub-pipelines, one per feature group. The transformer was embedded in a scikit-learn `Pipeline` alongside each estimator. This design ensures that imputers, scalers, and encoders are fitted only on training data. Fitting any of these steps on the full dataset before the train/test split would introduce leakage: the imputer would borrow test-set values into training medians, the scaler would align training bounds to the global distribution, and the power transform would learn the full dataset's skew rather than the training fold's. All such biases are eliminated by keeping the transformations strictly inside the pipeline.

### 3.6 Train/Test Split

The dataset was divided into a training set of 2,344 observations (80%) and a held-out test set of 586 observations (20%) using a fixed random seed for reproducibility. For the classification task, stratified sampling was applied to ensure that all four typology classes were proportionally represented in both folds, which is particularly important given the small size of the split-level class.

The classification feature set was constructed separately from the regression set. Features carrying temporal information — `Year Built`, `Garage Yr Blt`, and `Year Remod/Add Delta` — were excluded, since the classification task is defined specifically around geometric information. `MS SubClass` was also excluded because its numeric codes directly encode storey configuration, which would constitute leakage for a task aimed at recovering that configuration. The remaining 30 spatial and structural features were imputed with column medians and standardised to zero mean and unit variance within the pipeline.

Finally, the string class labels for the classification target (`Single-story`, `Multi-story`, `1.5-story`, `Split-level`) were encoded to integers prior to training. This step was necessary because XGBoost 3.x stores class indices internally as integers and raises an error if re-fitted with string labels after an initial fit.

---

## 4. Model Selection

### 4.1 Regression Models

Nine regression algorithms were trained and compared to provide a broad view of how different learning strategies handle the task.

**Linear Regression** establishes a parametric baseline that assumes a linear relationship between predictors and target. Its inclusion makes the contribution of non-linearity visible when comparing against more flexible methods. **Ridge Regression** extends this with an L2 penalty (α = 1.0), which regularises the coefficient estimates and performs better when predictors are collinear — a reasonable concern given that many quality and area features in this dataset are correlated. Note that Polynomial Regression was initially considered but was replaced: applying degree-2 expansion to the approximately 150 post-encoding features produces an ill-conditioned feature matrix of around 11,000 columns, causing ordinary least squares to diverge.

**Decision Tree** provides a non-parametric, fully non-linear baseline. It is included to illustrate the characteristic pattern of unregularised trees: low training error but moderate generalisation performance. **Random Forest** builds on this by aggregating predictions from an ensemble of trees fitted on bootstrap samples with random feature subsampling, substantially reducing variance. **Gradient Boosting** and **XGBoost** take a different ensemble approach, fitting trees sequentially to the residuals of previous iterations, generally achieving high accuracy at the cost of more training time. XGBoost additionally applies regularisation terms to the leaf weights.

**Support Vector Regression (SVR)** with a radial basis function kernel finds the widest ε-insensitive tube around the regression function. It is sensitive to feature scaling, which the pipeline addresses, but its performance depends on kernel bandwidth choices that are not tuned in the initial comparison. **K-Nearest Neighbours (KNN)** with k = 5 makes predictions from the average of the nearest training observations and is included as a non-parametric, instance-based reference. **Multi-Layer Perceptron (MLP)** represents a neural network approach; the maximum iteration count was set to 2,000 to ensure convergence.

The primary evaluation metric for regression is **Root Mean Squared Error (RMSE)** in years, which penalises large individual prediction errors and is directly interpretable in the units of the target. **Mean Absolute Error (MAE)** and the **coefficient of determination (R²)** are reported as supplementary metrics.

### 4.2 Classification Models

Eight classifiers were trained using the same pipeline structure. **Logistic Regression** serves as the linear baseline. **Decision Tree** provides a non-linear, interpretable alternative. **Random Forest**, **Gradient Boosting**, and **XGBoost** are the ensemble methods expected to perform best based on their track record in tabular classification. **Support Vector Classifier (SVC)** provides a margin-based perspective. **K-Nearest Neighbours** is the instance-based reference. **Multi-Layer Perceptron** represents the neural network approach.

Given the class imbalance — single-story properties constitute 50.5% of observations while split-level properties account for only 7.2% — accuracy alone is not a sufficient evaluation metric. A model that simply classifies every property as single-story would achieve roughly 50% accuracy without learning anything meaningful. For this reason, **macro-averaged F1 score** is used as the primary evaluation metric. This computes precision and recall separately for each class and averages them with equal weight, penalising poor performance on minority classes as heavily as poor performance on the majority class. Accuracy, macro precision, and macro recall are reported alongside F1 for completeness.

### 4.3 Cross-validation and Hyperparameter Tuning

Five-fold cross-validation was applied to all models to assess generalisation stability beyond the single train/test split. For regression, negative mean squared error was used as the scoring criterion, with results converted to RMSE for reporting. For classification, accuracy was used as the CV scoring criterion.

Hyperparameter tuning was then applied to the four best-performing models in each task. For Random Forest and XGBoost, `RandomizedSearchCV` was used with 10 iterations over a wider parameter space, given the computational cost of exhaustive search. For Gradient Boosting and SVR/SVC, `GridSearchCV` was used. All searches were conducted with five-fold cross-validation on the training set, with the held-out test set reserved strictly for final evaluation. Regression searches optimised for negative mean squared error; classification searches optimised for macro F1.

---

## 5. Results

### 5.1 Regression Performance

Table 1 reports test-set performance across all nine regression models.

| Model | MAE (years) | RMSE (years) | R² |
|---|---|---|---|
| Linear Regression | 7.15 | 9.76 | 0.891 |
| Ridge Regression | 7.09 | 9.68 | 0.893 |
| Decision Tree | 2.37 | 6.51 | 0.952 |
| **Random Forest** | **2.02** | **4.85** | **0.973** |
| Gradient Boosting | 3.65 | 6.32 | 0.954 |
| XGBoost | 2.60 | 5.33 | 0.968 |
| Support Vector Regressor | 6.84 | 11.13 | 0.859 |
| K-Nearest Neighbours | 4.93 | 8.48 | 0.918 |
| Neural Network | 6.69 | 9.34 | 0.901 |

*Table 1: Regression results on the held-out test set (n = 586). Best result per metric is shown in bold.*

Random Forest achieves the lowest RMSE (4.85 years) and highest R² (0.973), meaning that it explains 97.3% of the variance in construction year on unseen data. XGBoost is a close second (RMSE = 5.33, R² = 0.968). Both ensemble tree methods substantially outperform the linear baselines, which plateau around R² = 0.89, confirming that non-linear interactions among predictors are necessary to recover the full chronological signal.

The SVR performs below expectations (RMSE = 11.13, R² = 0.859), comparable to the linear regression models despite its non-linear kernel. Although feature scaling is handled by the pipeline, the default radial basis function bandwidth and regularisation parameters are likely suboptimal for this dataset, and the SVR may require more extensive tuning to reach competitive performance.

The neural network achieves R² = 0.901 on the test set, placing it between the linear and ensemble methods. However, its cross-validation behaviour is notably unstable: the five-fold CV RMSE is 16.66 ± 11.56 years, far higher than its single test-set result and much more variable than any other model. This suggests that the MLP's performance is sensitive to how the data are divided, which may reflect an interaction between its default architecture and the uneven temporal distribution of the training folds.

Table 2 summarises five-fold cross-validation results, which confirm the ranking observed on the test set.

| Model | CV RMSE (mean) | CV RMSE (std) |
|---|---|---|
| Linear Regression | 12.54 | ±5.33 |
| Ridge Regression | 12.52 | ±5.34 |
| Decision Tree | 7.69 | ±0.76 |
| **Random Forest** | **5.24** | **±0.45** |
| Gradient Boosting | 6.10 | ±0.52 |
| XGBoost | 5.59 | ±0.75 |
| Support Vector Regressor | 12.38 | ±1.57 |
| K-Nearest Neighbours | 8.76 | ±1.19 |
| Neural Network | 16.66 | ±11.56 |

*Table 2: Five-fold cross-validation results for regression (scored as RMSE in years).*

Note that the linear regression CV RMSE (12.54) is considerably higher than its test-set RMSE (9.76). A similar gap appears for Ridge. This discrepancy likely reflects that the 80/20 hold-out split happened to produce a test set that is somewhat easier to predict than the average fold, or that the five folds include at least one configuration in which the temporal distribution of training and validation data is particularly unfavourable for a linear model.

After hyperparameter tuning, Random Forest and XGBoost retained their lead positions. The tuning provided modest additional gains — the tuned Random Forest achieved a CV RMSE of approximately 4.9 years — suggesting that the default hyperparameters were already reasonable and that further improvement would require deeper architectural changes rather than parameter adjustments.

### 5.2 Classification Performance

Table 3 reports test-set performance across all eight classification models.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | 0.879 | 0.792 | 0.699 | 0.726 |
| Decision Tree | 0.850 | 0.736 | 0.754 | 0.744 |
| Random Forest | 0.899 | 0.897 | 0.728 | 0.763 |
| Gradient Boosting | 0.913 | 0.897 | 0.779 | 0.814 |
| **XGBoost** | **0.915** | 0.881 | **0.802** | **0.833** |
| Support Vector Classifier | 0.879 | **0.914** | 0.670 | 0.691 |
| K-Nearest Neighbours | 0.843 | 0.780 | 0.675 | 0.710 |
| Neural Network | 0.881 | 0.804 | 0.767 | 0.783 |

*Table 3: Classification results on the held-out test set (n = 586). Best result per metric is shown in bold.*

XGBoost achieves the highest accuracy (91.5%) and the highest macro F1 score (0.833), followed by Gradient Boosting (91.3%, F1 = 0.814). The gap between accuracy and macro F1 across most models reflects the class imbalance. For example, the SVC achieves 87.9% accuracy but a macro F1 of only 0.691 — the same accuracy as Logistic Regression, but with substantially lower recall (0.670 vs 0.699). The SVC's precision of 0.914 is the highest of any model, indicating that when it predicts a minority class it is usually correct, but it misses a substantial proportion of those instances. This pattern is consistent with a model that reserves minority-class predictions for cases with very high confidence, defaulting to the dominant class otherwise.

The Neural Network performs well in absolute terms (accuracy = 88.1%, F1 = 0.783), placing fifth overall. Unlike in the regression task, its behaviour is more stable: the five-fold CV accuracy of 87.6% ± 1.7% is consistent with the test-set result, suggesting that the classification task, with its smaller and more uniform feature set, is a better match for the default MLP architecture.

Table 4 reports the five-fold cross-validation classification results.

| Model | CV Accuracy (mean) | CV Accuracy (std) |
|---|---|---|
| Logistic Regression | 0.869 | ±0.015 |
| Decision Tree | 0.852 | ±0.018 |
| Random Forest | 0.904 | ±0.011 |
| Gradient Boosting | 0.905 | ±0.006 |
| **XGBoost** | **0.910** | ±0.010 |
| Support Vector Classifier | 0.874 | ±0.011 |
| K-Nearest Neighbours | 0.836 | ±0.024 |
| Neural Network | 0.876 | ±0.017 |

*Table 4: Five-fold cross-validation results for classification (accuracy).*

The CV rankings are consistent with the test-set rankings, and the standard deviations are small for most models, indicating stable generalisation. Gradient Boosting is the most consistent model (std = 0.006), while K-Nearest Neighbours shows the highest variability (std = 0.024), likely because its predictions are sensitive to the local density of the training data in different folds.

---

## 6. Discussion and Conclusion

### 6.1 Regression Task

The high R² values achieved by the tree-based ensemble methods — 0.952 for Decision Tree, 0.954 for Gradient Boosting, 0.968 for XGBoost, and 0.973 for Random Forest — strongly support the hypothesis that a property's physical attributes encode a legible chronological signal. Construction year is, in effect, recoverable from the materials, dimensions, and quality ratings of a house, at least for properties in this dataset.

The consistent performance advantage of ensemble methods over linear models is informative. Linear regression and Ridge regression plateau at R² ≈ 0.89, suggesting that roughly 11% of the variance in construction year cannot be explained by any linear combination of the predictors. Tree-based methods reduce this unexplained variance dramatically, implying that the relationship between physical attributes and construction year is substantially non-linear. This is consistent with the known history of residential architecture: materials and design standards did not change smoothly and continuously but were instead subject to step-changes driven by regulation, technological availability, and cultural fashion. Trees are well suited to capturing such threshold effects.

The SVR's underperformance relative to its theoretical capacity is somewhat unexpected. With appropriate kernel tuning it might be competitive, but the default radial basis function parameters are evidently not well matched to this task. The gap between the SVR's test RMSE (11.13) and its CV RMSE (12.38) is relatively small compared to other models, suggesting that the default configuration is at least consistent, if suboptimal.

The neural network's high CV variance (16.66 ± 11.56) is the most notable anomalous result. The model achieves an R² of 0.90 on the held-out test set, which is a respectable result, but the cross-validation evidence suggests this figure may be somewhat optimistic for that particular split. With only 2,344 training observations, the default multi-layer perceptron likely lacks the training data volume to exploit deep feature interactions, and its performance is accordingly more fold-dependent than the tree-based methods.

### 6.2 Classification Task

The classification results confirm that architectural typology is largely recoverable from geometric and structural measurements, with the best model achieving 91.5% accuracy and a macro F1 of 0.833. This is a practically meaningful result: without temporal or market information, the model correctly identifies the structural form of approximately nine in ten properties.

However, the gap between accuracy and macro F1 across most models reveals an important limitation. The dominant class (Single-story, 50.5%) is easy to predict correctly, and doing so drives high accuracy without necessarily recovering the minority classes well. The macro F1 metric makes this limitation visible. The SVC is the clearest example: its accuracy matches Logistic Regression, but its macro F1 (0.691) is the lowest in the comparison, driven by low recall for split-level and one-and-a-half story properties. This behaviour — high precision with low recall — suggests the SVC is applying a conservative decision boundary that rarely ventures into minority-class territory.

The systematic difficulty with split-level and one-and-a-half story properties is architecturally interpretable. Both types occupy an intermediate position between fully single-story and fully multi-story designs. A one-and-a-half story home typically has a partial upper storey, resulting in second-floor area values that fall between zero (single-story) and the full first-floor equivalent (two-story). A split-level home similarly has partial floors at different heights. The 30-feature geometric representation captures aggregate measurements — total area, room counts, basement footprint — but does not capture the volumetric arrangement that distinguishes these forms from a full two-story or a straightforward ranch layout. This geometrical ambiguity places a ceiling on classification accuracy that is independent of model choice.

### 6.3 Limitations

Both tasks are evaluated on data from a single city over a four-year window. The chronological signal exploited by the regression model reflects construction norms specific to Ames, Iowa; the relationships between physical attributes and construction year may differ in other geographic or cultural contexts. The four-class typology consolidation is a deliberate simplification: the original eight style codes include rare configurations such as `2.5Fin` (a finished two-and-a-half story property), and the grouping decisions made here, while reasonable, are not the only valid choices.

The exclusion of temporal predictors from the classification feature set was intended to isolate purely geometric information, but some of the retained geometric variables — garage size, basement finish level — are themselves correlated with construction era, and the model may partially exploit an indirect temporal signal rather than purely spatial information.

### 6.4 Conclusion

This study demonstrates that residential properties in the Ames Housing Dataset carry two forms of latent structural information that machine learning can recover with high accuracy: the year of construction and the architectural typology. Random Forest achieves the strongest regression performance (RMSE = 4.85 years, R² = 0.973), and XGBoost achieves the strongest classification performance (accuracy = 91.5%, macro F1 = 0.833). In both tasks, gradient-boosted ensemble methods consistently outperform linear, instance-based, and single-tree alternatives, reflecting the threshold-driven, non-linear nature of the relationships between physical design and these categorical outcomes.

The results suggest a straightforward practical interpretation: the materials, spaces, and amenities installed in a house at the time of its construction leave a physical record that is detailed enough to recover its approximate age and form decades later. This has applications wherever structured property data needs to be validated, imputed, or classified at scale — from municipal planning databases to real estate data platforms. More broadly, it illustrates that supervised learning applied to non-market housing attributes can yield high-quality predictions even when the target variable is not directly observable in the data.

---

## References

De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing data as an end-of-semester regression project. *Journal of Statistics Education*, 19(3).

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
