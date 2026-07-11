"""
Generator script — creates 3 separate Jupyter notebooks:
  preprocessing.ipynb  – full EDA + leakage-safe pipeline + pickle save
  regression.ipynb     – 9 regressors, CV, HPT, evaluation
  classification.ipynb – 8 classifiers (same structure), CV, HPT, evaluation
Run:  python create_notebooks.py
"""
import json

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_source(s):
    s = s.strip('\n')
    lines = s.split('\n')
    if len(lines) == 1:
        return [lines[0]]
    return [ln + '\n' for ln in lines[:-1]] + [lines[-1]]

def code(s):
    return {'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': make_source(s)}

def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': make_source(s)}

KERNELSPEC = {
    'kernelspec': {'display_name': 'ames-housing-venv',
                   'language': 'python', 'name': 'ames-housing-venv'},
    'language_info': {'name': 'python', 'version': '3.13.0'},
}

def notebook(cells):
    return {'cells': cells, 'metadata': KERNELSPEC,
            'nbformat': 4, 'nbformat_minor': 5}


# ═════════════════════════════════════════════════════════════════════════════
#  preprocessing.ipynb
# ═════════════════════════════════════════════════════════════════════════════

prep = [

md("""# Data Preprocessing

This notebook prepares the Ames Housing dataset for two downstream modelling tasks. The first is a regression problem — predicting the year a house was built — and the second is a multi-class classification problem — identifying architectural typology from floor-plan geometry. Raw data is cleaned, features are engineered and a leakage-safe preprocessing pipeline is assembled before the train/test splits are saved to `preprocessed_data.pkl`."""),

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import pickle

from sklearn.preprocessing import (PowerTransformer, StandardScaler,
                                    MinMaxScaler, OneHotEncoder, OrdinalEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import BinaryEncoder
from sklearn.model_selection import train_test_split"""),

code("""df = pd.read_csv('AmesHousing.csv')
print(f'Shape: {df.shape}')
df.head(5)"""),

code("""df.isnull().sum()"""),

code("""threshhold = 0.70
missing = df.isnull().mean()
print('Dropped columns (>70% missing):')
print(missing[missing > threshhold])
df.drop(columns=missing[missing > threshhold].index, inplace=True)
print(f'Shape after drop: {df.shape}')"""),

md("""## Feature Engineering

`Year Remod/Add` is replaced by `Year Remod/Add Delta` — the number of years between original construction and the most recent remodel. This captures renovation recency without retaining a raw calendar year in the feature set. `House Style` is copied to a separate series before being dropped so it remains available for the classification task without contaminating the regression features."""),

code("""df.dropna(subset=['Year Built', 'House Style'], inplace=True)
df['Year Remod/Add Delta'] = df['Year Remod/Add'] - df['Year Built']
house_style_target = df['House Style'].copy()
df.drop(columns=['Order', 'PID', 'SalePrice', 'House Style',
                 'Year Remod/Add', 'Yr Sold', 'Mo Sold'], inplace=True)
print(f'Shape: {df.shape}')
df.head(3)"""),

md("""## Exploratory Analysis: Target Variable"""),

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['Year Built'], bins=40, kde=True, color='steelblue', ax=axes[0])
axes[0].set_title('Distribution of Year Built')
axes[0].set_xlabel('Year Built')
sns.boxplot(x=df['Year Built'], color='lightcoral', ax=axes[1])
axes[1].set_title('Boxplot of Year Built')
plt.tight_layout()
plt.show()
print(df['Year Built'].describe())
print(f"Skewness : {df['Year Built'].skew():.3f}")
print(f"Kurtosis : {df['Year Built'].kurtosis():.3f}")"""),

code("""num_features = df.select_dtypes(include=['float64', 'int64'])
corr_with_target = (num_features.corr(numeric_only=True)['Year Built']
                    .drop('Year Built').sort_values())
plt.figure(figsize=(10, 10))
colors = ['red' if x < 0 else 'green' for x in corr_with_target.values]
sns.barplot(x=corr_with_target.values, y=corr_with_target.index, palette=colors)
plt.title('Correlation of Numerical Predictors with Year Built')
plt.xlabel('Pearson Correlation')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()"""),

code("""key_predictors = ['Overall Qual', 'Gr Liv Area', 'Garage Yr Blt',
                  'Total Bsmt SF', 'Garage Cars', 'Year Remod/Add Delta']
plt.figure(figsize=(15, 9))
for i, col in enumerate(key_predictors, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=df[col], y=df['Year Built'], alpha=0.4)
    plt.xlabel(col)
    plt.ylabel('Year Built')
    plt.title(f'{col} vs Year Built')
plt.tight_layout()
plt.show()"""),

md("""## Missing Values"""),

code("""lst_null_values = df.columns[df.isnull().any()].tolist()
print('Columns with missing values:', lst_null_values)"""),

code("""num_clos = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols_null = [col for col in lst_null_values if col in num_clos]
print('Numerical columns with NaNs:', num_cols_null)"""),

code("""df[num_cols_null].hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.tight_layout()
plt.show()"""),

code("""plt.figure(figsize=(15, 12))
for i, column in enumerate(num_cols_null, 1):
    plt.subplot(4, 5, i)
    stats.probplot(df[column].dropna(), dist='norm', plot=plt)
    plt.title(column)
plt.tight_layout()
plt.show()"""),

code("""skewness = df[num_cols_null].apply(lambda x: skew(x.dropna()))
kurt     = df[num_cols_null].apply(lambda x: kurtosis(x.dropna()))
norm_null_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurt})
print(norm_null_df.sort_values(by=['Skewness', 'Kurtosis']))"""),

code("""cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols_null = [col for col in lst_null_values if col not in num_cols_null]
print('Categorical columns:', cat_cols)
print('Categorical with NaNs:', cat_cols_null)"""),

code("""plt.figure(figsize=(15, 12))
for i, column in enumerate(cat_cols_null, 1):
    plt.subplot(4, 5, i)
    sns.countplot(data=df, x=column, palette='Set2')
    pct = df[column].isnull().mean() * 100
    plt.title(f'{column} ({pct:.1f}% missing)')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""),

md("""## Outlier Inspection"""),

code("""def Visualize_box_hist(df, n_rows, cols):
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    for i, column in enumerate(cols):
        sns.histplot(df[column], bins=30, color='skyblue', ax=axes[i, 0])
        axes[i, 0].set_title(f'Histogram of {column}')
        sns.boxplot(y=df[column], ax=axes[i, 1])
        axes[i, 1].set_title(f'Box Plot of {column}')
    plt.tight_layout()
    plt.show()

Visualize_box_hist(df, len(num_clos), num_clos)"""),

code("""skew_kurt = {}
for column in num_clos:
    col_skew = df[column].skew()
    col_kurt = df[column].kurtosis()
    skew_kurt[column] = {'Skewness': col_skew, 'Kurtosis': col_kurt}
    print(f'{column:30s}  skew={col_skew:+.3f}  kurt={col_kurt:+.3f}')"""),

code("""zero_skew_kurt = [k for k, v in skew_kurt.items()
                   if v['Skewness'] == 0 and v['Kurtosis'] == 0]
variance = df[zero_skew_kurt].var()
print('Zero-variance columns (if any):')
print(variance)"""),

code("""# Refresh num_clos after any zero-variance drops
num_clos = df.select_dtypes(include=['float64', 'int64']).columns.tolist()"""),

code("""# Confirm no negative values (safe to use Yeo-Johnson)
neg_cols = [col for col in num_clos if (df[col] < 0).any()]
print('Columns with negative values:', neg_cols if neg_cols else 'none')"""),

code("""# Final skew / kurtosis check for scaling assignment
skew_kurt = {}
for column in num_clos:
    col_skew = df[column].skew()
    col_kurt = df[column].kurtosis()
    skew_kurt[column] = {'Skewness': col_skew, 'Kurtosis': col_kurt}
    print(f'{column:30s}  skew={col_skew:+.3f}  kurt={col_kurt:+.3f}')"""),

code("""df_plot = df[num_clos].replace([np.inf, -np.inf], np.nan)
df_plot.hist(figsize=(15, 12), bins=30, edgecolor='black')
plt.tight_layout()
plt.show()"""),

md("""## Scaling and Encoding

Based on the skewness and kurtosis checks above, numerical features are split into two groups: those close to normal are standardised (Yeo-Johnson followed by `StandardScaler`), while more skewed features are normalised (Yeo-Johnson followed by `MinMaxScaler`). Categorical features are handled by ordinal encoding (ordered quality ratings), one-hot encoding (nominal low-cardinality) or binary encoding (high-cardinality, >10 unique values)."""),

code("""standardization = [
    'Lot Frontage', 'Lot Area', 'Overall Qual', 'Total Bsmt SF', '1st Flr SF',
    'Gr Liv Area', 'Full Bath', 'Year Remod/Add Delta', 'BsmtFin SF 1',
    'Bsmt Unf SF', 'Garage Yr Blt', 'Garage Cars', 'TotRms AbvGrd',
    'Garage Area', 'Overall Cond', 'Bedroom AbvGr',
]
normalization = [
    'MS SubClass', 'Mas Vnr Area', 'Half Bath', '2nd Flr SF',
    'Wood Deck SF', 'Open Porch SF', 'Bsmt Full Bath', 'Fireplaces',
]
print(f'Standardize : {len(standardization)} features')
print(f'Normalize   : {len(normalization)} features')"""),

code("""ordinal_cat = [
    'Lot Shape', 'Land Slope', 'Exter Qual', 'Exter Cond', 'Bsmt Qual',
    'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2',
    'Heating QC', 'Kitchen Qual', 'Fireplace Qu', 'Garage Qual', 'Garage Cond',
    'Garage Finish', 'Functional', 'Paved Drive', 'Street', 'Central Air',
    'Electrical', 'Utilities',
]
categories = [
    ['Reg', 'IR1', 'IR2', 'IR3'],
    ['Gtl', 'Mod', 'Sev'],
    ['Ex', 'Gd', 'TA', 'Fa'],
    ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'Unknown'],
    ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    ['Gd', 'Av', 'Mn', 'No'],
    ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'Unknown'],
    ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'],
    ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'Unknown'],
    ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
    ['Fin', 'RFn', 'Unf', 'Unknown'],
    ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
    ['Y', 'P', 'N'],
    ['Pave', 'Grvl'],
    ['Y', 'N'],
    ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
    ['AllPub', 'NoSewr', 'NoSeWa'],
]"""),

code("""def check_category_type(df, threshold=10):
    variable_info = {}
    nominal_cat   = []
    hc_cat        = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        is_ord   = col in ordinal_cat
        if n_unique > threshold:
            enc = 'High cardinality'
            hc_cat.append(col)
        elif is_ord:
            enc = 'Ordinal'
        else:
            enc = 'Nominal'
            nominal_cat.append(col)
        variable_info[col] = {'Unique Categories': n_unique, 'Category Type': enc}
    return pd.DataFrame(variable_info).T, nominal_cat, hc_cat"""),

code("""variable_analysis, nominal_cat, hc_cat = check_category_type(df)
print(variable_analysis)
print(f'\\nNominal         : {nominal_cat}')
print(f'High-cardinality: {hc_cat}')"""),

md("""## Preprocessing Pipeline

All stateful transformations — imputation, power transform, scaling and encoding — are built into a single `ColumnTransformer`. Fitting happens inside the pipeline on the training fold only, so no statistic from the test set influences the training process."""),

code("""numeric_std = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('power',  PowerTransformer(method='yeo-johnson', standardize=True)),
])
numeric_norm = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('power',  PowerTransformer(method='yeo-johnson', standardize=False)),
    ('scale',  MinMaxScaler()),
])
ordinal_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OrdinalEncoder(categories=categories,
                              handle_unknown='use_encoded_value',
                              unknown_value=-1)),
])
nominal_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
highcard_pipe = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', BinaryEncoder()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('std',     numeric_std,   standardization),
        ('norm',    numeric_norm,  normalization),
        ('ordinal', ordinal_pipe,  ordinal_cat),
        ('onehot',  nominal_pipe,  nominal_cat),
        ('binary',  highcard_pipe, hc_cat),
    ],
    remainder='drop',
)
print('Preprocessor built.')"""),

md("""## Building the Model Matrix

The CSV is re-read and only deterministic, non-stateful steps are applied at this stage — dropping sparse columns, removing rows with missing targets and computing the engineered delta. All imputation and scaling happens later, inside the pipeline, on the training fold exclusively."""),

code("""df_model = pd.read_csv('AmesHousing.csv')
_missing = df_model.isnull().mean()
df_model.drop(columns=_missing[_missing > 0.70].index, inplace=True)
df_model.dropna(subset=['Year Built', 'House Style'], inplace=True)
df_model['Year Remod/Add Delta'] = df_model['Year Remod/Add'] - df_model['Year Built']
house_style_target = df_model['House Style'].copy()
df_model.drop(columns=['Order', 'PID', 'SalePrice', 'House Style',
                       'Year Remod/Add', 'Yr Sold', 'Mo Sold'], inplace=True)

y = df_model['Year Built']
X = df_model.drop(columns='Year Built')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Feature matrix : {X.shape}')
print(f'Train {X_train.shape}  |  Test {X_test.shape}')"""),

md("""## Classification Feature Set

For the typology task, temporal predictors (`Year Built`, `Garage Yr Blt`, `Year Remod/Add Delta`) and `MS SubClass` are excluded, since they either encode construction era or directly encode story configuration. Only geometric and structural measurements remain."""),

code("""X_reg = X
y_reg = y
preprocessor_reg = preprocessor

y_clf = house_style_target.copy()
typology_map = {
    '1Story': 'Single-story',
    '2Story': 'Multi-story', '2.5Fin': 'Multi-story', '2.5Unf': 'Multi-story',
    '1.5Fin': '1.5-story',   '1.5Unf': '1.5-story',
    'SLvl':   'Split-level', 'SFoyer': 'Split-level',
}
y_clf = y_clf.map(typology_map).fillna(y_clf)

clf_blocklist = {'Year Built', 'Garage Yr Blt', 'Year Remod/Add Delta', 'MS SubClass'}
clf_features  = [c for c in X.select_dtypes(include=['number']).columns
                 if c not in clf_blocklist]
X_clf = X[clf_features]

preprocessor_clf = ColumnTransformer(
    transformers=[('geom', Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scale',  StandardScaler()),
    ]), clf_features)],
    remainder='drop',
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

print('Class distribution:')
print(y_clf.value_counts())
print(f'\\nRegression     : train {X_train_reg.shape}  test {X_test_reg.shape}')
print(f'Classification : train {X_train_clf.shape}  test {X_test_clf.shape}')
print(f'Geometry features ({len(clf_features)}): {clf_features}')"""),

md("""## Saving the Preprocessed Data"""),

code("""artifacts = {
    # Shared raw matrix
    'X': X, 'y': y,
    # Regression
    'X_reg': X_reg, 'y_reg': y_reg,
    'X_train': X_train, 'X_test': X_test,
    'y_train': y_train, 'y_test': y_test,
    'X_train_reg': X_train_reg, 'X_test_reg': X_test_reg,
    'y_train_reg': y_train_reg, 'y_test_reg': y_test_reg,
    'preprocessor': preprocessor,
    'preprocessor_reg': preprocessor_reg,
    # Classification
    'X_clf': X_clf, 'y_clf': y_clf,
    'X_train_clf': X_train_clf, 'X_test_clf': X_test_clf,
    'y_train_clf': y_train_clf, 'y_test_clf': y_test_clf,
    'preprocessor_clf': preprocessor_clf,
    'clf_features': clf_features,
}
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print('Saved: preprocessed_data.pkl')
print('Keys:', list(artifacts.keys()))"""),

]  # end prep cells


# ═════════════════════════════════════════════════════════════════════════════
#  regression.ipynb
# ═════════════════════════════════════════════════════════════════════════════

reg = [

md("""# Regression: Predicting Construction Year

This notebook trains and compares regression models that predict the year a house was built (`Year Built`) from its physical and quality attributes. Nine models are evaluated — ranging from a linear baseline to ensemble methods and a neural network — with the best performers subsequently tuned through cross-validated hyperparameter search.

> Run `preprocessing.ipynb` first to generate `preprocessed_data.pkl`."""),

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import uniform

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV"""),

code("""with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X            = data['X']
y            = data['y']
X_train      = data['X_train']
X_test       = data['X_test']
y_train      = data['y_train']
y_test       = data['y_test']
preprocessor = data['preprocessor']

print(f'X_train : {X_train.shape}  |  X_test : {X_test.shape}')
print(f'y range : {int(y.min())} to {int(y.max())}')"""),

md("""## Model Selection

Nine regressors are included to cover a broad range of learning strategies: a linear baseline, polynomial regression (degree 2), a decision tree, random forest, gradient boosting, XGBoost, support vector regression, k-nearest neighbours and a multi-layer perceptron. This spread makes it easy to see whether the problem favours parametric or tree-based approaches."""),

code("""models = {
    'Linear Regression':      LinearRegression(),
    'Polynomial Regression':  Pipeline(steps=[
        ('poly',    PolynomialFeatures(degree=2)),
        ('lin_reg', LinearRegression()),
    ]),
    'Decision Tree':          DecisionTreeRegressor(random_state=42),
    'Random Forest':          RandomForestRegressor(random_state=42),
    'Gradient Boosting':      GradientBoostingRegressor(random_state=42),
    'XGBoost':                XGBRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
    'K-Nearest Neighbors':    KNeighborsRegressor(n_neighbors=5),
    'Neural Network':         MLPRegressor(random_state=42, max_iter=500),
}"""),

code("""def evaluation_matrix(model_name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {
        'Model': model_name,
        'MAE':   round(mae,  4),
        'MSE':   round(mse,  4),
        'RMSE':  round(rmse, 4),
        'R2':    round(r2,   4),
    }"""),

md("""## Training and Initial Evaluation"""),

code("""trained_models = {}
eval_matrix    = []

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', clone(model)),
    ])
    pipeline.fit(X_train, y_train)
    X_test_t = pipeline[:-1].transform(X_test)
    y_pred   = pipeline[-1].predict(X_test_t)
    trained_models[model_name] = y_pred
    eval_matrix.append(evaluation_matrix(model_name, y_test, y_pred))

print('Training complete.')"""),

code("""eval_matrix_df = pd.DataFrame(eval_matrix)
eval_matrix_df"""),

code("""best_mae_idx  = eval_matrix_df['MAE'].idxmin()
best_rmse_idx = eval_matrix_df['RMSE'].idxmin()
best_r2_idx   = eval_matrix_df['R2'].idxmax()

best_row = {
    'Model': 'Best Model',
    'MAE':   eval_matrix_df.loc[best_mae_idx,  'Model'],
    'MSE':   eval_matrix_df.loc[best_mae_idx,  'Model'],
    'RMSE':  eval_matrix_df.loc[best_rmse_idx, 'Model'],
    'R2':    eval_matrix_df.loc[best_r2_idx,   'Model'],
}
pd.concat([eval_matrix_df, pd.DataFrame([best_row])], ignore_index=True)"""),

md("""## Cross-Validation

Five-fold cross-validation gives a more reliable estimate of generalisation performance than the single train/test split. Negative MSE is used as the internal scoring criterion; results are converted to RMSE for display."""),

code("""cv_results = []
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', clone(model))])
    scores      = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    cv_results.append({
        'Model':         name,
        'CV RMSE mean':  f'{rmse_scores.mean():.4f}',
        'CV RMSE std':   f'+/- {rmse_scores.std():.4f}',
    })

cv_results_df = pd.DataFrame(cv_results)
cv_results_df"""),

md("""## Predicted vs Actual Values

Scatter plots of predicted against actual construction year give a visual check on fit quality. The red dashed line marks perfect prediction; tighter clustering around it indicates lower error."""),

code("""plt.figure(figsize=(15, 10))
plt.suptitle('Predicted vs Actual Construction Year', fontsize=14)

for i, (name, y_pred) in enumerate(trained_models.items(), 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Year Built')
    plt.ylabel('Predicted Year Built')
    plt.title(name)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()"""),

md("""## Hyperparameter Tuning

The four best-performing models — Random Forest, Gradient Boosting, XGBoost and SVR — are tuned with cross-validated search. Random Forest and XGBoost use a randomised search over a wider parameter space; Gradient Boosting and SVR use an exhaustive grid search. All searches use MSE as the objective."""),

code("""param_grids = {
    'Random Forest': {
        'model__n_estimators':      [50, 100, 200],
        'model__max_depth':         [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf':  [1, 2, 4],
    },
    'Gradient Boosting': {
        'model__n_estimators':  [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth':     [3, 5, 7],
    },
    'XGBoost': {
        'model__n_estimators':  [50, 100, 200],
        'model__learning_rate': uniform(0.01, 0.3),
        'model__max_depth':     [3, 5, 7],
    },
    'Support Vector Regressor': {
        'model__kernel': ['linear', 'rbf'],
        'model__C':      [0.1, 1, 10],
        'model__gamma':  ['scale', 'auto'],
    },
}"""),

code("""tune_models = {
    'Random Forest':            RandomForestRegressor(random_state=42),
    'Gradient Boosting':        GradientBoostingRegressor(random_state=42),
    'XGBoost':                  XGBRegressor(random_state=42),
    'Support Vector Regressor': SVR(),
}
best_models = {}"""),

code("""for name, model in tune_models.items():
    print(f'Tuning {name}...')
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    if name in ['Random Forest', 'XGBoost']:
        search = RandomizedSearchCV(
            pipe, param_distributions=param_grids[name],
            n_iter=10, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, random_state=42,
        )
    else:
        search = GridSearchCV(
            pipe, param_grid=param_grids[name],
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1,
        )

    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    print(f'  Best params : {search.best_params_}')
    print(f'  Best CV RMSE: {np.sqrt(-search.best_score_):.4f}')
    print()"""),

md("""## Results after Tuning"""),

code("""eval_matrix = []
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    eval_matrix.append(evaluation_matrix(name, y_test, y_pred))

eval_matrix_df = pd.DataFrame(eval_matrix)
eval_matrix_df"""),

]  # end reg cells


# ═════════════════════════════════════════════════════════════════════════════
#  classification.ipynb
# ═════════════════════════════════════════════════════════════════════════════

clf = [

md("""# Classification: House Style Typology

The `House Style` variable in the Ames dataset describes a property's architectural form. Eight distinct style codes are grouped into four broader classes — Single-story, Multi-story, 1.5-story and Split-level — and a set of classifiers is trained to predict which class a house belongs to using only its geometric and structural measurements. Temporal predictors are deliberately excluded to avoid leakage.

> Run `preprocessing.ipynb` first to generate `preprocessed_data.pkl`."""),

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import uniform

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV"""),

code("""with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_clf            = data['X_clf']
y_clf            = data['y_clf']
X_train_clf      = data['X_train_clf']
X_test_clf       = data['X_test_clf']
y_train_clf      = data['y_train_clf']
y_test_clf       = data['y_test_clf']
preprocessor_clf = data['preprocessor_clf']

print('Class distribution:')
print(y_clf.value_counts())
print(f'\\nX_train_clf : {X_train_clf.shape}  |  X_test_clf : {X_test_clf.shape}')"""),

code("""plt.figure(figsize=(8, 5))
y_clf.value_counts().sort_values().plot(
    kind='barh', color='steelblue', edgecolor='black')
plt.title('Architectural Typology Distribution')
plt.xlabel('Count')
plt.tight_layout()
plt.show()"""),

md("""## Classifier Selection

Eight classifiers are compared under the same pipeline structure used for regression: a linear baseline (Logistic Regression), simple non-parametric methods (Decision Tree, KNN), ensemble approaches (Random Forest, Gradient Boosting, XGBoost), a margin-based method (SVC) and a small neural network (MLP). This gives a clear picture of which learning strategy handles the four-class typology problem best."""),

code("""models = {
    'Logistic Regression':       LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':             DecisionTreeClassifier(random_state=42),
    'Random Forest':             RandomForestClassifier(random_state=42),
    'Gradient Boosting':         GradientBoostingClassifier(random_state=42),
    'XGBoost':                   XGBClassifier(random_state=42,
                                               eval_metric='mlogloss', verbosity=0),
    'Support Vector Classifier': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors':       KNeighborsClassifier(n_neighbors=5),
    'Neural Network':            MLPClassifier(random_state=42, max_iter=500),
}"""),

code("""def evaluation_matrix(model_name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return {
        'Model':             model_name,
        'Accuracy':          round(acc,  4),
        'Precision (macro)': round(prec, 4),
        'Recall (macro)':    round(rec,  4),
        'F1 (macro)':        round(f1,   4),
    }"""),

md("""## Training and Initial Evaluation"""),

code("""trained_models = {}
eval_matrix    = []

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_clf),
        ('model', clone(model)),
    ])
    pipeline.fit(X_train_clf, y_train_clf)
    y_pred = pipeline.predict(X_test_clf)
    trained_models[model_name] = y_pred
    eval_matrix.append(evaluation_matrix(model_name, y_test_clf, y_pred))

print('Training complete.')"""),

code("""eval_matrix_df = pd.DataFrame(eval_matrix)
eval_matrix_df"""),

code("""best_acc_idx = eval_matrix_df['Accuracy'].idxmax()
best_f1_idx  = eval_matrix_df['F1 (macro)'].idxmax()

best_row = {
    'Model':             'Best Model',
    'Accuracy':          eval_matrix_df.loc[best_acc_idx, 'Model'],
    'Precision (macro)': eval_matrix_df.loc[best_f1_idx,  'Model'],
    'Recall (macro)':    eval_matrix_df.loc[best_f1_idx,  'Model'],
    'F1 (macro)':        eval_matrix_df.loc[best_f1_idx,  'Model'],
}
pd.concat([eval_matrix_df, pd.DataFrame([best_row])], ignore_index=True)"""),

md("""## Cross-Validation

Five-fold stratified cross-validation is used to get a more reliable accuracy estimate. Stratification ensures each fold maintains the same class proportions as the full dataset."""),

code("""cv_results = []
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor_clf), ('model', clone(model))])
    scores = cross_val_score(pipe, X_clf, y_clf, cv=5, scoring='accuracy')
    cv_results.append({
        'Model':          name,
        'CV Acc mean':    f'{scores.mean():.4f}',
        'CV Acc std':     f'+/- {scores.std():.4f}',
    })

cv_results_df = pd.DataFrame(cv_results)
cv_results_df"""),

md("""## Confusion Matrices

The matrices below reveal where each model misclassifies houses. Systematic confusion between adjacent categories — Multi-story and 1.5-story in particular — is informative about the separability of the typology classes."""),

code("""class_labels = sorted(y_clf.unique())
n_models = len(trained_models)
n_cols   = 4
n_rows   = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.ravel()

for i, (name, y_pred) in enumerate(trained_models.items()):
    cm = confusion_matrix(y_test_clf, y_pred, labels=class_labels)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i],
                xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()"""),

md("""## Per-class Performance"""),

code("""for name, y_pred in trained_models.items():
    print(f'=== {name} ===')
    print(classification_report(y_test_clf, y_pred,
                                target_names=class_labels, zero_division=0))"""),

md("""## Hyperparameter Tuning

The four best-performing classifiers are tuned using cross-validated search, with macro F1 as the scoring criterion to weight all four classes equally. Random Forest and XGBoost use a randomised search; Gradient Boosting and SVC use an exhaustive grid search."""),

code("""param_grids = {
    'Random Forest': {
        'model__n_estimators':      [50, 100, 200],
        'model__max_depth':         [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
    },
    'Gradient Boosting': {
        'model__n_estimators':  [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth':     [3, 5, 7],
    },
    'XGBoost': {
        'model__n_estimators':  [50, 100, 200],
        'model__learning_rate': uniform(0.01, 0.3),
        'model__max_depth':     [3, 5, 7],
    },
    'Support Vector Classifier': {
        'model__kernel': ['linear', 'rbf'],
        'model__C':      [0.1, 1, 10],
        'model__gamma':  ['scale', 'auto'],
    },
}"""),

code("""tune_models = {
    'Random Forest':             RandomForestClassifier(random_state=42),
    'Gradient Boosting':         GradientBoostingClassifier(random_state=42),
    'XGBoost':                   XGBClassifier(random_state=42,
                                               eval_metric='mlogloss', verbosity=0),
    'Support Vector Classifier': SVC(random_state=42),
}
best_models = {}"""),

code("""for name, model in tune_models.items():
    print(f'Tuning {name}...')
    pipe = Pipeline(steps=[('preprocessor', preprocessor_clf), ('model', model)])

    if name in ['Random Forest', 'XGBoost']:
        search = RandomizedSearchCV(
            pipe, param_distributions=param_grids[name],
            n_iter=10, cv=5, scoring='f1_macro',
            n_jobs=-1, random_state=42,
        )
    else:
        search = GridSearchCV(
            pipe, param_grid=param_grids[name],
            cv=5, scoring='f1_macro', n_jobs=-1,
        )

    search.fit(X_train_clf, y_train_clf)
    best_models[name] = search.best_estimator_
    print(f'  Best params      : {search.best_params_}')
    print(f'  Best CV F1 (mac) : {search.best_score_:.4f}')
    print()"""),

md("""## Results after Tuning"""),

code("""eval_matrix = []
for name, model in best_models.items():
    y_pred = model.predict(X_test_clf)
    eval_matrix.append(evaluation_matrix(name, y_test_clf, y_pred))

eval_matrix_df = pd.DataFrame(eval_matrix)
eval_matrix_df"""),

code("""fig, axes = plt.subplots(1, len(best_models), figsize=(5 * len(best_models), 5))
if len(best_models) == 1:
    axes = [axes]

for i, (name, model) in enumerate(best_models.items()):
    y_pred = model.predict(X_test_clf)
    cm = confusion_matrix(y_test_clf, y_pred, labels=class_labels)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i],
                xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    axes[i].set_title(f'{name} (tuned)')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()"""),

]  # end clf cells


# ─────────────────────────────────────────────────────────────────────────────
# Write notebooks
# ─────────────────────────────────────────────────────────────────────────────
notebooks = {
    'preprocessing.ipynb':  prep,
    'regression.ipynb':     reg,
    'classification.ipynb': clf,
}

for fname, cells in notebooks.items():
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(notebook(cells), f, indent=1, ensure_ascii=False)
    n_code = sum(1 for c in cells if c['cell_type'] == 'code')
    n_md   = sum(1 for c in cells if c['cell_type'] == 'markdown')
    print(f'Created {fname:30s}  ({n_code} code cells, {n_md} markdown cells)')
