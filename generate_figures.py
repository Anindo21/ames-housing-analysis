"""
Generate all report figures and save to figures/.
Run once before compiling ude_thesis.Rmd.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading preprocessed_data.pkl ...")
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
X, y            = data['X'], data['y']
preprocessor    = data['preprocessor']
X_clf           = data['X_clf']
X_train_clf, X_test_clf     = data['X_train_clf'], data['X_test_clf']
y_train_clf_raw, y_test_clf_raw = data['y_train_clf'], data['y_test_clf']
preprocessor_clf = data['preprocessor_clf']

le = LabelEncoder().fit(data['y_clf'])
class_labels    = list(le.classes_)
y_train_clf_enc = pd.Series(le.transform(y_train_clf_raw), index=y_train_clf_raw.index)
y_test_clf_enc  = pd.Series(le.transform(y_test_clf_raw),  index=y_test_clf_raw.index)

# EDA raw frame
df = pd.read_csv('AmesHousing.csv')
df.drop(columns=df.columns[df.isnull().mean() > 0.70], inplace=True)
df.dropna(subset=['Year Built', 'House Style'], inplace=True)
df['Year Remod/Add Delta'] = df['Year Remod/Add'] - df['Year Built']
house_style_target = df['House Style'].copy()

# ── Figure 1: Year Built distribution ────────────────────────────────────────
print("Figure 1 ...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df['Year Built'], bins=40, kde=True, color='steelblue', ax=axes[0])
axes[0].set_title('(a) Distribution of Year Built')
axes[0].set_xlabel('Year Built'); axes[0].set_ylabel('Frequency')
axes[1].boxplot(df['Year Built'].dropna(), vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', color='black'),
                medianprops=dict(color='black'))
axes[1].set_title('(b) Boxplot of Year Built')
axes[1].set_xlabel('Year Built'); axes[1].set_yticks([])
plt.tight_layout()
plt.savefig('figures/fig_yearbuilt_dist.png', bbox_inches='tight')
plt.close()

# ── Figure 2: House Style distribution ───────────────────────────────────────
print("Figure 2 ...")
typology_map = {
    '1Story': 'Single-story',
    '2Story': 'Multi-story', '2.5Fin': 'Multi-story', '2.5Unf': 'Multi-story',
    '1.5Fin': '1.5-story',   '1.5Unf': '1.5-story',
    'SLvl':   'Split-level', 'SFoyer': 'Split-level',
}
style_counts = house_style_target.value_counts().sort_values(ascending=False)
typo_counts  = house_style_target.map(typology_map).value_counts().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(style_counts.index, style_counts.values, color='steelblue', edgecolor='black', width=0.7)
axes[0].set_title('(a) Original House Style Codes')
axes[0].set_xlabel('Style Code'); axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
for bar, v in zip(axes[0].patches, style_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+8, str(v), ha='center', va='bottom', fontsize=8)

axes[1].bar(typo_counts.index, typo_counts.values, color='#2ca25f', edgecolor='black', width=0.5)
axes[1].set_title('(b) Consolidated Typology Classes (4)')
axes[1].set_xlabel('Typology Class'); axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=20)
for bar, v in zip(axes[1].patches, typo_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height()+8, str(v), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('figures/fig_housestyle_dist.png', bbox_inches='tight')
plt.close()

# ── Figure 3: Correlation with Year Built ────────────────────────────────────
print("Figure 3 ...")
num_f = df.select_dtypes(include=['float64', 'int64'])
corr = num_f.corr(numeric_only=True)['Year Built'].drop('Year Built').sort_values()
n = 12
corr_d = pd.concat([corr.iloc[:n], corr.iloc[-n:]])
colors_c = ['#d73027' if x < 0 else '#1a9850' for x in corr_d.values]

plt.figure(figsize=(8, 8))
plt.barh(range(len(corr_d)), corr_d.values, color=colors_c, edgecolor='white')
plt.yticks(range(len(corr_d)), corr_d.index)
plt.xlabel('Pearson Correlation Coefficient')
plt.title('Pearson Correlation with Year Built\n(12 largest positive and negative)')
plt.axvline(0, color='black', linewidth=0.8)
plt.legend(handles=[mpatches.Patch(color='#1a9850', label='Positive'),
                     mpatches.Patch(color='#d73027', label='Negative')],
           loc='lower right')
plt.tight_layout()
plt.savefig('figures/fig_correlation.png', bbox_inches='tight')
plt.close()

# ── Figure 4: Scatter plots ───────────────────────────────────────────────────
print("Figure 4 ...")
key_pred = ['Overall Qual', 'Gr Liv Area', 'Garage Yr Blt',
            'Total Bsmt SF', 'Garage Cars', 'Year Remod/Add Delta']
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.ravel()
for i, col in enumerate(key_pred):
    axes[i].scatter(df[col], df['Year Built'], alpha=0.25, s=8, color='steelblue', linewidths=0)
    axes[i].set_xlabel(col); axes[i].set_ylabel('Year Built'); axes[i].set_title(col)
plt.suptitle('Selected Predictors vs Year Built', y=1.01)
plt.tight_layout()
plt.savefig('figures/fig_scatter_predictors.png', bbox_inches='tight')
plt.close()

# ── Figure 5: Regression model comparison ────────────────────────────────────
print("Figure 5 ...")
mnames = ['Linear Reg.', 'Ridge Reg.', 'Decision Tree', 'Random Forest',
          'Grad. Boosting', 'XGBoost', 'SVR', 'KNN', 'Neural Network']
rmse_v = [9.7641, 9.6814, 6.5082, 4.8484, 6.3236, 5.3285, 11.1250, 8.4758, 9.3404]
r2_v   = [0.8913, 0.8931, 0.9517, 0.9732, 0.9544, 0.9676, 0.8589, 0.9181, 0.9005]

rank_r = np.argsort(rmse_v)
cmap_r = {}
for rank, idx in enumerate(rank_r):
    if rank < 2: cmap_r[idx] = '#1a9641'
    elif rank < 4: cmap_r[idx] = '#a6d96a'
    elif rank < 7: cmap_r[idx] = '#fdae61'
    else: cmap_r[idx] = '#d7191c'
colors_r = [cmap_r[i] for i in range(len(mnames))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
yp = range(len(mnames))
b0 = axes[0].barh(yp, rmse_v, color=colors_r, edgecolor='black', height=0.6)
axes[0].set_yticks(yp); axes[0].set_yticklabels(mnames)
axes[0].set_xlabel('RMSE (years)'); axes[0].set_title('Test-set RMSE')
for bar, v in zip(b0, rmse_v):
    axes[0].text(v+0.1, bar.get_y()+bar.get_height()/2, f'{v:.2f}', va='center', fontsize=8)

b1 = axes[1].barh(yp, r2_v, color=colors_r, edgecolor='black', height=0.6)
axes[1].set_yticks(yp); axes[1].set_yticklabels(mnames)
axes[1].set_xlim(0.82, 1.00); axes[1].set_xlabel('R²'); axes[1].set_title('Test-set R²')
for bar, v in zip(b1, r2_v):
    axes[1].text(v+0.001, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('figures/fig_reg_comparison.png', bbox_inches='tight')
plt.close()

# ── Figure 6: Regression CV ───────────────────────────────────────────────────
print("Figure 6 ...")
cv_r  = [12.5437, 12.5154, 7.6851, 5.2381, 6.1017, 5.5886, 12.3818, 8.7599, 16.6615]
cv_rs = [5.3257,  5.3378,  0.7638, 0.4485, 0.5189, 0.7490,  1.5672,  1.1936, 11.5587]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(mnames)), cv_r, xerr=cv_rs, color='steelblue', edgecolor='black',
        height=0.6, capsize=4, alpha=0.85)
ax.set_yticks(range(len(mnames))); ax.set_yticklabels(mnames)
ax.set_xlabel('CV RMSE (years)')
ax.set_title('5-Fold Cross-Validation RMSE — Regression\n(mean ± standard deviation)')
plt.tight_layout()
plt.savefig('figures/fig_reg_cv.png', bbox_inches='tight')
plt.close()

# ── Figure 7: Predicted vs Actual ────────────────────────────────────────────
print("Figure 7: fitting 4 models (may take ~60 s) ...")
top4 = {
    'Random Forest':     RandomForestRegressor(random_state=42),
    'XGBoost':           XGBRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Ridge Regression':  Ridge(alpha=1.0),
}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
for i, (name, model) in enumerate(top4.items()):
    pipe = Pipeline([('pre', preprocessor), ('m', clone(model))])
    pipe.fit(X_train, y_train)
    yp = pipe.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - yp)**2))
    r2   = 1 - np.sum((y_test - yp)**2) / np.sum((y_test - y_test.mean())**2)
    lo, hi = min(y_test.min(), yp.min()), max(y_test.max(), yp.max())
    axes[i].scatter(y_test, yp, alpha=0.35, s=10, color='steelblue', linewidths=0)
    axes[i].plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='Perfect fit')
    axes[i].set_xlabel('Actual Year Built'); axes[i].set_ylabel('Predicted Year Built')
    axes[i].set_title(f'{name}\nRMSE = {rmse:.2f} yrs,  R² = {r2:.3f}')
    axes[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/fig_predicted_vs_actual.png', bbox_inches='tight')
plt.close()

# ── Figure 8: Classification comparison ──────────────────────────────────────
print("Figure 8 ...")
cnames  = ['Logistic Reg.', 'Decision Tree', 'Random Forest', 'Grad. Boosting',
           'XGBoost', 'SVC', 'KNN', 'Neural Network']
acc_v  = [0.8788, 0.8498, 0.8993, 0.9130, 0.9147, 0.8788, 0.8430, 0.8805]
f1_v   = [0.7256, 0.7439, 0.7634, 0.8136, 0.8333, 0.6909, 0.7104, 0.7826]

rank_f = np.argsort(f1_v)
cmap_f = {}
for rank, idx in enumerate(rank_f):
    if rank >= 6: cmap_f[idx] = '#1a9641'
    elif rank >= 4: cmap_f[idx] = '#a6d96a'
    elif rank >= 2: cmap_f[idx] = '#fdae61'
    else: cmap_f[idx] = '#d7191c'
colors_f = [cmap_f[i] for i in range(len(cnames))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
yp = range(len(cnames))
b0 = axes[0].barh(yp, acc_v, color=colors_f, edgecolor='black', height=0.6)
axes[0].set_yticks(yp); axes[0].set_yticklabels(cnames)
axes[0].set_xlim(0.80, 0.94); axes[0].set_xlabel('Accuracy'); axes[0].set_title('Test-set Accuracy')
for bar, v in zip(b0, acc_v):
    axes[0].text(v+0.001, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)

b1 = axes[1].barh(yp, f1_v, color=colors_f, edgecolor='black', height=0.6)
axes[1].set_yticks(yp); axes[1].set_yticklabels(cnames)
axes[1].set_xlim(0.60, 0.89); axes[1].set_xlabel('Macro F1'); axes[1].set_title('Test-set Macro F1 Score')
for bar, v in zip(b1, f1_v):
    axes[1].text(v+0.004, bar.get_y()+bar.get_height()/2, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('figures/fig_clf_comparison.png', bbox_inches='tight')
plt.close()

# ── Figure 9: Classification CV ───────────────────────────────────────────────
print("Figure 9 ...")
cv_a  = [0.8686, 0.8522, 0.9044, 0.9048, 0.9099, 0.8737, 0.8358, 0.8758]
cv_as = [0.0147, 0.0179, 0.0112, 0.0056, 0.0103, 0.0106, 0.0240, 0.0173]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(cnames)), cv_a, xerr=cv_as, color='#2ca25f', edgecolor='black',
        height=0.6, capsize=4, alpha=0.85)
ax.set_yticks(range(len(cnames))); ax.set_yticklabels(cnames)
ax.set_xlim(0.78, 0.95); ax.set_xlabel('CV Accuracy')
ax.set_title('5-Fold Cross-Validation Accuracy — Classification\n(mean ± standard deviation)')
plt.tight_layout()
plt.savefig('figures/fig_clf_cv.png', bbox_inches='tight')
plt.close()

# ── Figure 10: XGBoost confusion matrix ──────────────────────────────────────
print("Figure 10: fitting XGBoost classifier ...")
xgb_pipe = Pipeline([
    ('pre', preprocessor_clf),
    ('m',   XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0))
])
xgb_pipe.fit(X_train_clf, y_train_clf_enc)
y_pred_xgb = xgb_pipe.predict(X_test_clf)
cm = confusion_matrix(y_test_clf_enc, y_pred_xgb, labels=[0,1,2,3])

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
cbar = fig.colorbar(im, ax=ax, fraction=0.046)
cbar.set_label('Count', rotation=270, labelpad=15)
ax.set_xticks([0,1,2,3]); ax.set_yticks([0,1,2,3])
ax.set_xticklabels(class_labels, rotation=30, ha='right')
ax.set_yticklabels(class_labels)
ax.set_title('XGBoost — Confusion Matrix (Test Set)')
ax.set_ylabel('Actual Typology'); ax.set_xlabel('Predicted Typology')
thresh = cm.max() / 2
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{cm[i,j]}', ha='center', va='center', fontsize=12,
                color='white' if cm[i,j] > thresh else 'black')
plt.tight_layout()
plt.savefig('figures/fig_confusion_matrix.png', bbox_inches='tight')
plt.close()

print(f"\nDone. {len(os.listdir('figures'))} figures saved to figures/")
