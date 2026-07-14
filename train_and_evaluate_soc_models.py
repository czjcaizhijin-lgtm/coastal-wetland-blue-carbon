import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import joblib
import shap
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import RFECV
from scipy.stats import gaussian_kde

# 基础环境配置与大刊中文字体支持
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']  # 切换为国际通用学术字体
plt.rcParams['axes.unicode_minus'] = False

# ==========================================================================
# 1. Data Loading, High Dynamic Range Filtering & Geomorphological Indexing
# ==========================================================================
print("=== Phase 1: Loading data and constructing coastal hydrodynamic indices ===")
file_path = r'D:\PyCharm2020(64bit)\pythonProject\模型对比\ALL_data2.csv'
df_raw = pd.read_csv(file_path).dropna()

# Exclude anomalous high dynamic range pixels to stabilize regression bounds
df = df_raw[df_raw['C_storage'] <= 40000].copy()

# Constructing advanced cross-shore geomorphological and tidal proxies
df['tidal_power'] = df['tidal_range_m'] * df['inundation_freq']
df['inundation_stress'] = df['tidal_range_m'] / (df['elevation_m'] + 1)

y = df['soc_perc']
y_log10 = np.log10(y + 0.01)  # Log-transforming to mitigate skewed distribution

# All 37 initial features spanning 5 earth science dimensions
initial_features = [
    'ndvi', 'EVI', 'LAI', 'LSWI', 'elevation_m', 'Slope', 'Aspect',
    'Curvature', 'TWI', 'tidal_range_m', 'inundation_freq', 'dist_to_river_m', 'Sediment_Proxy',
    'tsm_p', 'Salinity_psu', 'BIO1_Annual_Temp', 'BIO4_Temp_Seasonality', 'BIO5_Max_Temp_Warmest',
    'BIO6_Min_Temp_Coldest', 'precip_annual', 'BIO14_Precip_Dry_Month',
    'BIO15_Precip_Seasonality', 'Vapr_kPa', 'sst_k', 'soil_moisture',
    'bdod_1m', 'clay_1m', 'silt_1m', 'sand_1m', 'nitrogen_1m', 'phh2o_1m',
    'Geology_Class', 'HFI', 'Night_Light', 'Pop_Disturbance', 'tidal_power', 'inundation_stress'
]

X_raw = df[initial_features]
# One-hot encoding for the categorical geological feature
X_encoded = pd.get_dummies(X_raw, columns=['Geology_Class'], prefix='Geo', drop_first=True)
feature_names = X_encoded.columns.tolist()

# ==========================================================================
# 2. Train-Test Splitting & Standardization
# ==========================================================================
X_train_raw, X_test_raw, y_train_log, y_test_log = train_test_split(X_encoded, y_log10, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names)

# ==========================================================================
# 3. Hyperparameter Optimization via Multi-Dimensional Grid Search
# ==========================================================================
print("\n=== Phase 2: Optimizing LightGBM hyperparameters via Grid Search ===")
lgbm_base = LGBMRegressor(random_state=42, importance_type='gain', n_jobs=-1, verbosity=-1)

param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [8, 12],
    'num_leaves': [31, 63],
    'min_child_samples': [10, 20]
}

grid_search = GridSearchCV(lgbm_base, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train_log)
best_params_model = grid_search.best_estimator_

# ==========================================================================
# 4. Multi-collinearity Diagnosis & Full Feature Importance (37 Variables)
# ==========================================================================
print("\n=== Phase 3: Diagnosing collinearity and ranking full 37 features ===")

# 1. Full 37x37 Pearson Correlation Matrix (Lower Triangle Heatmap)
full_corr_matrix = X_train_raw[feature_names].corr(method='pearson')
plt.figure(figsize=(16, 14), dpi=600)
mask_all = np.triu(np.ones_like(full_corr_matrix, dtype=bool))

sns.heatmap(full_corr_matrix, mask=mask_all, cmap='coolwarm', vmax=1.0, vmin=-1.0, center=0,
            square=True, linewidths=0.2, cbar_kws={"shrink": 0.7, "label": "Pearson Correlation Coefficient (r)"},
            annot=True, fmt=".2f", annot_kws={"size": 4.5, "weight": "bold"})
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.yticks(fontsize=7)
plt.title('Diagnosing Multicollinearity Across All 37 Initial Environmental Features', fontsize=12, weight='bold',
          pad=25)
plt.tight_layout()
plt.savefig('Full_37_Features_Correlation_Matrix.png', dpi=600)
plt.show()

# 2. Full 37 Feature Importance Barplot (Based on Gain)
importances_all = grid_search.best_estimator_.feature_importances_
importances_all_normalized = importances_all / np.sum(importances_all)
full_feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances_all_normalized}).sort_values(
    by='Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 11), dpi=600)
bars_all = ax.barh(full_feat_imp_df['Feature'], full_feat_imp_df['Importance'], color='#1a365d', height=0.65)
ax.set_xlabel('Normalized Feature Importance (Based on Gain)', fontsize=11, labelpad=10)
ax.set_ylabel('Initial Environmental Forcing Features', fontsize=11, labelpad=10)
ax.set_title('Baseline Contribution of All 37 Covariates to Deep SOC Estimation', fontsize=12, weight='bold', pad=15)
ax.grid(True, axis='x', linestyle=':', alpha=0.5, color='#cbd5e0')
ax.set_xlim(0, full_feat_imp_df['Importance'].max() * 1.15)
sns.despine(top=True, right=True)

for bar in bars_all:
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', va='center', ha='left', fontsize=6.5,
            color='#4a5568', weight='bold')
plt.tight_layout()
plt.savefig('Full_37_Features_Importance_Barplot.png', dpi=600)
plt.show()

# ==========================================================================
# 5. Feature Decoupling via RFECV & Learning Curve Exportation
# ==========================================================================
print("\n=== Phase 4: Executing 5-fold RFECV to isolate optimal feature subset ===")
rfecv = RFECV(estimator=best_params_model, step=1, cv=5, scoring='r2', n_jobs=-1)
rfecv.fit(X_train, y_train_log)

selected_features = X_train.columns[rfecv.support_].tolist()
print(f"🎯 Decoupling complete! Optimal subset size: {rfecv.n_features_} / {len(feature_names)}")

# Plotting the English-version RFECV evolution curve (Cleaned Version)
plt.figure(figsize=(9, 6.5), dpi=600)
cv_scores = rfecv.cv_results_['mean_test_score']
cv_stds = rfecv.cv_results_['std_test_score']
feature_nums = range(1, len(cv_scores) + 1)

plt.plot(feature_nums, cv_scores, color='#1a365d', marker='o', markersize=3.5, lw=1.8, label='5-fold CV $R^2$ (Mean)')
plt.fill_between(feature_nums, cv_scores - cv_stds, cv_scores + cv_stds, color='#3182ce', alpha=0.15,
                 label='Uncertainty Bound ($\pm$1 SD)')
plt.axvline(x=rfecv.n_features_, color='#e53e3e', linestyle='--', lw=1.5,
            label=f'Optimal Feature Subset (n = {rfecv.n_features_})')

plt.xlabel('Number of Features Selected', fontsize=11)
plt.ylabel('Cross-Validated Verification $R^2$', fontsize=11)
plt.title('Evolution of Deep SOC Prediction Performance via RFECV Algorithm', fontsize=12, weight='bold', pad=15)
plt.xlim(0, len(cv_scores) + 1)
plt.grid(True, linestyle=':', alpha=0.5, color='#cbd5e0')
plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e2e8f0')
sns.despine(top=True, right=True)
plt.tight_layout()
plt.savefig('RFECV_Optimal_Features_Curve.png', dpi=600)
plt.show()

# ==========================================================================
# 6. Final Model Training & Fairness Benchmarking (R² vs. Inference Latency)
# ==========================================================================
print("\n=== Phase 5: Commencing cross-model validation and millisecond efficiency benchmarking ===")
best_model = grid_search.best_estimator_
best_model.fit(X_train[selected_features], y_train_log)

models_to_compare = {
    "LightGBM": best_model,
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
}

predictions = {}
r2_scores = {}
final_stats = []

X_train_fill = X_train[selected_features].fillna(0)
X_test_fill = X_test[selected_features].fillna(0)

for name, model in models_to_compare.items():
    X_tr_in = X_train_fill if name in ["Random Forest", "Gradient Boosting"] else X_train[selected_features]
    X_te_in = X_test_fill if name in ["Random Forest", "Gradient Boosting"] else X_test[selected_features]

    if name != "LightGBM":
        model.fit(X_tr_in, y_train_log)

    # Precise measurement of computational latency
    start_time = time.time()
    y_pred = model.predict(X_te_in)
    end_time = time.time()

    elapsed_time_ms = (end_time - start_time) * 1000
    mae = mean_absolute_error(y_test_log, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_log, y_pred))
    r2_val = r2_score(y_test_log, y_pred)

    predictions[name] = y_pred
    r2_scores[name] = r2_val

    final_stats.append({
        "Model": name,
        "R2": round(r2_val, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Predict_Time (ms)": round(elapsed_time_ms, 2)
    })

# Outputting Table 2 (Perfect copy-paste for the manuscript text)
df_final = pd.DataFrame(final_stats)
print("\n" + "=" * 65)
print("📊 Manuscript Table 2: Model Performance & Efficiency Assessment")
print("=" * 65)
print(df_final.to_string(index=False))
print("=" * 65)

# ==========================================================================
# 7. 2x2 High-Density Multi-Model Scatter Cross-Plots (NC Style)
# ==========================================================================
print("\n=== Phase 6: Plotting 2x2 cross-validated point densityクロス plots ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)
axes = axes.flatten()

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i]
    obs, pred = y_test_log, y_pred

    xy = np.vstack([obs, pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    obs_sorted, pred_sorted, z_sorted = obs.iloc[idx], pred[idx], z[idx]

    sc = ax.scatter(obs_sorted, pred_sorted, c=z_sorted, s=12, cmap='magma', alpha=0.85, zorder=2)

    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax.plot(lims, lims, 'k--', alpha=0.7, lw=1.2, zorder=1)

    ax.set_xlabel('Observed $\log_{10}(SOC)$', fontsize=10)
    ax.set_ylabel('Predicted $\log_{10}(SOC)$', fontsize=10)
    ax.set_title(f'{name} Spatial Mapping Alignment (Validation $R^2$ = {r2_scores[name]:.4f})', fontsize=11,
                 weight='bold')
    ax.grid(True, linestyle=':', alpha=0.4)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel Point Density', fontsize=9)

plt.tight_layout()
plt.savefig('Models_Comparison_Density_2x2.png', dpi=600)
plt.show()

# ==========================================================================
# 8. Advanced Biogeochemical Interpretation via SHAP Framework
# ==========================================================================
print("\n=== Phase 7: Initiating SHAP framework for biogeochemical mechanism explanation ===")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test[selected_features])
shap_values_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

# 1. Exporting feature optimization rankings to Excel
rank_df = pd.DataFrame({
    'Feature_Name': feature_names,
    'RFECV_Rank_Score': rfecv.ranking_,
    'Is_Core_Selected': rfecv.support_
}).sort_values('RFECV_Rank_Score')
rank_df.to_excel('Final_RFECV_Feature_Ranking_Output.xlsx', index=False)

# 2. Exporting SHAP summary plot (Beeswarm layout for validating BIO4 negative character)
plt.figure(figsize=(10, 8), dpi=300)
shap.summary_plot(shap_values_to_plot, X_test[selected_features], show=False)
plt.title("SHAP Summary Matrix for Deep Tidal Flat SOC Prediction", fontsize=12, weight='bold', pad=15)
plt.tight_layout()
plt.savefig('SHAP_Global_Summary_Beeswarm.png', dpi=600)
plt.show()

# 3. Generating 2x2 Interaction Dependence Profiles for Top 4 Core Drivers
feature_importance = np.abs(shap_values_to_plot).mean(0)
top_4_indices = np.argsort(feature_importance)[-4:][::-1]
top_4_features = [selected_features[i] for i in top_4_indices]

fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=600)
axes = axes.flatten()
for i, col in enumerate(top_4_features):
    shap.dependence_plot(
        col, shap_values_to_plot, X_test[selected_features],
        ax=axes[i], show=False, interaction_index='auto'
    )
    axes[i].set_title(f"Core Driver Profile {i + 1}: {col} Nonlinear Interaction", fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('Top4_Drivers_Interaction_Profiles.png', dpi=600)
plt.show()

# Freezing models and scalers for deployment mapping
joblib.dump(best_model, 'best_lgbm_model_refined.pkl')
joblib.dump(scaler, 'data_scaler.pkl')

print(
    "\n🚀 [🎉 All Tasks Successfully Completed!] All manuscript tables, full Pearson matrix plots, 2x2 density plots, and SHAP graphs have been exported at 300 DPI.")