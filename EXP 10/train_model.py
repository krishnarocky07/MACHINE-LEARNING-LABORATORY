"""
train_model.py
==============
Complete Machine Learning Training and Evaluation Pipeline for:
Medicine Stock Prediction Using Linear Regression and Random Forest Regression

This script:
1. Generates or loads a reproducible, documented pharmaceutical inventory dataset.
2. Performs data inspection, validation, and preprocessing.
3. Generates 6 Exploratory Data Analysis (EDA) visualizations.
4. Splits data into 80% training and 20% testing sets (random_state=42).
5. Builds Scikit-learn ColumnTransformer preprocessing pipelines.
6. Trains Linear Regression and Random Forest Regressor.
7. Performs 5-Fold Cross-Validation on training data to select the optimal model.
8. Evaluates the selected model on held-out test data (MAE, MSE, RMSE, R²).
9. Generates evaluation visualizations (Actual vs Predicted, Residuals, Feature Importance).
10. Saves the fitted pipeline and comprehensive metadata for the Flask web application.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for generating figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Configuration and Directories
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_PLOTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "medicine_stock.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "medicine_stock_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Styling for Matplotlib / Seaborn
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "medium",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------------------------------------------------------
# 1. Dataset Generation / Loading
# ---------------------------------------------------------
def generate_demonstration_dataset(num_records=700, random_seed=42):
    """
    Generates a realistic, documented simulated pharmaceutical inventory dataset.
    
    Academic & Educational Disclosure:
    Publicly available hospital pharmacy inventory records containing daily usage,
    stock buffers, and supplier delivery delays are strictly protected or confidential.
    Hence, this dataset is a carefully constructed educational demonstration dataset
    reflecting real clinical pharmacy inventory dynamics.
    """
    np.random.seed(random_seed)

    # Catalog of standard hospital medicines and categories
    catalog = [
        {"name": "Paracetamol", "category": "Pain Relief", "base_daily": 25, "lead_time_mean": 3},
        {"name": "Ibuprofen", "category": "Pain Relief", "base_daily": 18, "lead_time_mean": 4},
        {"name": "Tramadol", "category": "Pain Relief", "base_daily": 10, "lead_time_mean": 5},
        {"name": "Aspirin", "category": "Pain Relief", "base_daily": 15, "lead_time_mean": 3},
        {"name": "Amoxicillin", "category": "Antibiotics", "base_daily": 22, "lead_time_mean": 4},
        {"name": "Azithromycin", "category": "Antibiotics", "base_daily": 16, "lead_time_mean": 5},
        {"name": "Ciprofloxacin", "category": "Antibiotics", "base_daily": 14, "lead_time_mean": 6},
        {"name": "Cephalexin", "category": "Antibiotics", "base_daily": 12, "lead_time_mean": 5},
        {"name": "Cetirizine", "category": "Allergy Medicine", "base_daily": 20, "lead_time_mean": 3},
        {"name": "Loratadine", "category": "Allergy Medicine", "base_daily": 14, "lead_time_mean": 4},
        {"name": "Fexofenadine", "category": "Allergy Medicine", "base_daily": 11, "lead_time_mean": 4},
        {"name": "Metformin", "category": "Diabetes Medicine", "base_daily": 28, "lead_time_mean": 4},
        {"name": "Glimepiride", "category": "Diabetes Medicine", "base_daily": 15, "lead_time_mean": 4},
        {"name": "Insulin Glargine", "category": "Diabetes Medicine", "base_daily": 9, "lead_time_mean": 7},
        {"name": "Omeprazole", "category": "Gastric Medicine", "base_daily": 24, "lead_time_mean": 3},
        {"name": "Pantoprazole", "category": "Gastric Medicine", "base_daily": 21, "lead_time_mean": 4},
        {"name": "Ranitidine", "category": "Gastric Medicine", "base_daily": 13, "lead_time_mean": 4},
        {"name": "Vitamin C", "category": "Vitamins", "base_daily": 30, "lead_time_mean": 3},
        {"name": "Vitamin D3", "category": "Vitamins", "base_daily": 19, "lead_time_mean": 4},
        {"name": "Zinc Sulfate", "category": "Vitamins", "base_daily": 17, "lead_time_mean": 4},
        {"name": "Atorvastatin", "category": "Cardiovascular", "base_daily": 23, "lead_time_mean": 4},
        {"name": "Amlodipine", "category": "Cardiovascular", "base_daily": 19, "lead_time_mean": 3},
        {"name": "Salbutamol", "category": "Respiratory", "base_daily": 16, "lead_time_mean": 5},
        {"name": "Montelukast", "category": "Respiratory", "base_daily": 12, "lead_time_mean": 5},
    ]

    records = []
    # Pick medicines to form the dataset records
    chosen_indices = np.random.choice(len(catalog), size=num_records, replace=True)

    for idx in chosen_indices:
        item = catalog[idx]
        med_name = item["name"]
        med_category = item["category"]
        
        # Average Daily Usage with natural hospital variance (must be > 0)
        daily_usage = max(1.0, float(np.round(np.random.normal(item["base_daily"], item["base_daily"] * 0.22), 1)))
        
        # Previous monthly consumption (~30 days of usage with fluctuation)
        prev_consumption = max(10, int(np.round(daily_usage * 30 * np.random.uniform(0.85, 1.20))))
        
        # Supplier Delivery Time (days) - integer between 1 and 14
        lead_time = int(np.clip(np.random.poisson(item["lead_time_mean"]) + 1, 1, 14))
        
        # Current Stock on hand (varied from depleted to overstocked)
        stock_ratio = np.random.choice([0.2, 0.5, 0.8, 1.0, 1.3, 1.6], p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
        current_stock = max(0, int(np.round(daily_usage * 30 * stock_ratio + np.random.normal(0, 15))))
        
        # Target: Predicted Demand for the replenishment cycle (units)
        # Driven by:
        # 1. Base monthly requirement (daily_usage * 30)
        # 2. Historical consumption trend weight (0.28 * prev_consumption)
        # 3. Buffer required during supplier replenishment lead time (lead_time * daily_usage * 0.4)
        # 4. Seasonal & random hospital demand variation (noise term)
        seasonal_factor = np.random.uniform(0.92, 1.10)
        random_shock = np.random.normal(0, 18)
        
        demand_raw = (
            (daily_usage * 22) +
            (0.28 * prev_consumption) +
            (lead_time * daily_usage * 0.35)
        ) * seasonal_factor + random_shock

        # Demand must be strictly non-negative integer
        predicted_demand = max(5, int(np.round(demand_raw)))

        records.append({
            "Medicine Name": med_name,
            "Medicine Category": med_category,
            "Current Stock": current_stock,
            "Previous Consumption": prev_consumption,
            "Average Daily Usage": daily_usage,
            "Supplier Delivery Time": lead_time,
            "Predicted Demand": predicted_demand
        })

    df = pd.DataFrame(records)
    return df

# ---------------------------------------------------------
# 2. Dataset Preparation & Inspection
# ---------------------------------------------------------
def prepare_dataset():
    print("=" * 70)
    print("STAGE 1: Problem Definition & Data Collection")
    print("=" * 70)
    print("Target: Predict future medicine requirement (Predicted Demand) in units.")
    print("Features: Medicine Name, Medicine Category, Current Stock,")
    print("          Previous Consumption, Average Daily Usage, Supplier Delivery Time.")
    print("Dataset Status: Documented Educational Demonstration Dataset.")
    print("-" * 70)

    df = generate_demonstration_dataset(num_records=700, random_seed=42)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved successfully to: {DATA_PATH}")

    print("\n" + "=" * 70)
    print("STAGE 2: Data Inspection & Summary Statistics")
    print("=" * 70)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn Names & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:<26}: {dtype}")

    print("\nFirst 5 Records:")
    print(df.head().to_string(index=False))

    print("\nDescriptive Statistics (Numerical Features & Target):")
    print(df.describe().round(2).to_string())

    print("\nMedicine Category Distribution:")
    cat_dist = df["Medicine Category"].value_counts()
    for cat, count in cat_dist.items():
        print(f"  - {cat:<22}: {count} records ({count/len(df)*100:.1f}%)")

    print("\nMedicine Name Distribution (Top 10):")
    name_dist = df["Medicine Name"].value_counts().head(10)
    for name, count in name_dist.items():
        print(f"  - {name:<22}: {count} records")

    print("\nTarget Variable ('Predicted Demand') Summary:")
    print(f"  - Minimum : {df['Predicted Demand'].min()} units")
    print(f"  - Maximum : {df['Predicted Demand'].max()} units")
    print(f"  - Mean    : {df['Predicted Demand'].mean():.2f} units")
    print(f"  - Median  : {df['Predicted Demand'].median():.2f} units")
    print(f"  - Std Dev : {df['Predicted Demand'].std():.2f} units")

    print("\nCurrent Stock Statistics:")
    print(f"  - Minimum : {df['Current Stock'].min()} units")
    print(f"  - Maximum : {df['Current Stock'].max()} units")
    print(f"  - Mean    : {df['Current Stock'].mean():.2f} units")
    print(f"  - Median  : {df['Current Stock'].median():.2f} units")

    return df

# ---------------------------------------------------------
# 3. Data Preprocessing & Validation
# ---------------------------------------------------------
def preprocess_and_validate(df):
    print("\n" + "=" * 70)
    print("STAGE 3: Data Preprocessing & Integrity Verification")
    print("=" * 70)

    # 1. Check Missing Values
    missing_counts = df.isnull().sum()
    print("Missing Values Check:")
    for col, cnt in missing_counts.items():
        print(f"  - {col:<26}: {cnt} missing")
    if missing_counts.sum() > 0:
        df = df.dropna()

    # 2. Check Duplicate Records
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Records Check: {duplicates} duplicate(s) found.")
    if duplicates > 0:
        df = df.drop_duplicates()

    # 3. Numeric & Non-negative Verifications
    numeric_cols = [
        "Current Stock", "Previous Consumption",
        "Average Daily Usage", "Supplier Delivery Time", "Predicted Demand"
    ]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} must be numeric!"
        assert (df[col] >= 0).all(), f"Column {col} must be non-negative!"
        print(f"  [Verified] {col:<26}: Numeric & Non-negative (Min: {df[col].min()}, Max: {df[col].max()})")

    # 4. Categorical Verifications
    cat_cols = ["Medicine Name", "Medicine Category"]
    for col in cat_cols:
        unique_vals = df[col].nunique()
        print(f"  [Verified] {col:<26}: {unique_vals} valid categories/names")

    # 5. Separation into Features and Target
    feature_cols = [
        "Medicine Name", "Medicine Category", "Current Stock",
        "Previous Consumption", "Average Daily Usage", "Supplier Delivery Time"
    ]
    target_col = "Predicted Demand"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"\nInput Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target Variable: {target_col}")

    summary = {
        "total_records": int(len(df)),
        "input_features_count": len(feature_cols),
        "missing_values": int(missing_counts.sum()),
        "duplicate_records": int(duplicates),
        "categories_count": int(df["Medicine Category"].nunique()),
        "medicines_count": int(df["Medicine Name"].nunique()),
        "categories_list": sorted(df["Medicine Category"].unique().tolist()),
        "medicines_list": sorted(df["Medicine Name"].unique().tolist()),
        "target_stats": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(np.round(y.mean(), 2)),
            "median": float(np.round(y.median(), 2)),
            "std": float(np.round(y.std(), 2))
        },
        "stock_stats": {
            "min": float(df["Current Stock"].min()),
            "max": float(df["Current Stock"].max()),
            "mean": float(np.round(df["Current Stock"].mean(), 2)),
            "median": float(np.round(df["Current Stock"].median(), 2))
        }
    }

    return X, y, df, summary

# ---------------------------------------------------------
# 4. Exploratory Data Analysis (EDA) Visualizations
# ---------------------------------------------------------
def generate_eda_visualizations(df):
    print("\n" + "=" * 70)
    print("STAGE 4: Exploratory Data Analysis (Generating Visualizations)")
    print("=" * 70)

    # 1. Medicine Demand Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Predicted Demand"], kde=True, color="#0f4c81", bins=25, edgecolor="white", alpha=0.7)
    plt.axvline(df["Predicted Demand"].mean(), color="#e63946", linestyle="--", linewidth=2, label=f"Mean Demand ({df['Predicted Demand'].mean():.1f})")
    plt.axvline(df["Predicted Demand"].median(), color="#2a9d8f", linestyle=":", linewidth=2, label=f"Median Demand ({df['Predicted Demand'].median():.1f})")
    plt.title("Fig 3.1: Medicine Demand Distribution (Histogram with KDE)")
    plt.xlabel("Predicted Demand (units)")
    plt.ylabel("Frequency")
    plt.legend(frameon=True)
    p1 = os.path.join(STATIC_PLOTS_DIR, "demand_distribution.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p1}")

    # 2. Current Stock vs Medicine Demand
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df, x="Current Stock", y="Predicted Demand",
        hue="Medicine Category", palette="tab10", alpha=0.75, s=55
    )
    plt.title("Fig 3.2: Current Stock vs. Medicine Demand")
    plt.xlabel("Current Stock Quantity (units)")
    plt.ylabel("Predicted Demand (units)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0., fontsize=8)
    p2 = os.path.join(STATIC_PLOTS_DIR, "stock_vs_demand.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p2}")

    # 3. Average Demand by Medicine Category
    plt.figure(figsize=(8, 5))
    cat_avg = df.groupby("Medicine Category")["Predicted Demand"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=cat_avg, x="Predicted Demand", y="Medicine Category", hue="Medicine Category", palette="Blues_r", legend=False)
    plt.title("Fig 3.3: Average Demand by Medicine Category")
    plt.xlabel("Average Demand (units)")
    plt.ylabel("Medicine Category")
    for index, value in enumerate(cat_avg["Predicted Demand"]):
        plt.text(value + 8, index, f"{value:.1f}", va="center", fontsize=9, fontweight="bold")
    plt.xlim(0, cat_avg["Predicted Demand"].max() * 1.15)
    p3 = os.path.join(STATIC_PLOTS_DIR, "category_demand.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p3}")

    # 4. Medicine Consumption Analysis (Previous Consumption vs Demand)
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df, x="Previous Consumption", y="Predicted Demand",
        scatter_kws={"alpha": 0.5, "color": "#028090", "s": 35},
        line_kws={"color": "#e63946", "linewidth": 2}
    )
    plt.title("Fig 3.4: Previous Consumption vs. Predicted Demand (Trendline)")
    plt.xlabel("Previous Consumption (units)")
    plt.ylabel("Predicted Demand (units)")
    p4 = os.path.join(STATIC_PLOTS_DIR, "consumption_analysis.png")
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p4}")

    # 5. Supplier Delivery Time vs Medicine Demand
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Supplier Delivery Time", y="Predicted Demand", color="#48cae4", boxprops=dict(alpha=0.8))
    plt.title("Fig 3.5: Supplier Delivery Time vs. Medicine Demand Distribution")
    plt.xlabel("Supplier Delivery Time (days)")
    plt.ylabel("Predicted Demand (units)")
    p5 = os.path.join(STATIC_PLOTS_DIR, "delivery_vs_demand.png")
    plt.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p5}")

    # 6. Feature Relationships Correlation Heatmap
    plt.figure(figsize=(8, 5))
    num_cols = ["Current Stock", "Previous Consumption", "Average Daily Usage", "Supplier Delivery Time", "Predicted Demand"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, mask=mask, linewidths=0.5)
    plt.title("Fig 3.6: Feature Correlation Heatmap (Numerical Variables)")
    p6 = os.path.join(STATIC_PLOTS_DIR, "feature_correlation.png")
    plt.savefig(p6, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p6}")

# ---------------------------------------------------------
# 5. Preprocessing Pipeline & Model Training
# ---------------------------------------------------------
def train_and_compare_models(X, y):
    print("\n" + "=" * 70)
    print("STAGE 5: Train-Test Split & Preprocessing Pipeline")
    print("=" * 70)

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Total Records  : {len(X)}")
    print(f"Training Data  : {len(X_train)} records (80%)")
    print(f"Testing Data   : {len(X_test)} records (20%) [Held-Out]")

    # Features definition
    numeric_features = ["Current Stock", "Previous Consumption", "Average Daily Usage", "Supplier Delivery Time"]
    categorical_features = ["Medicine Name", "Medicine Category"]

    # Scikit-learn ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    print("\nPreprocessing Components:")
    print("  - Numerical   : StandardScaler applied to", numeric_features)
    print("  - Categorical : OneHotEncoder(handle_unknown='ignore') applied to", categorical_features)
    print("  - Fitting Rule: Transformer fitted strictly on training data only!")

    # Models definition
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    print("\n" + "=" * 70)
    print("STAGE 6: Cross-Validation & Model Selection (Training Data Only)")
    print("=" * 70)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
            return_train_score=False
        )

        cv_rmse = -scores["test_neg_root_mean_squared_error"]
        cv_mae = -scores["test_neg_mean_absolute_error"]
        cv_r2 = scores["test_r2"]

        cv_results[name] = {
            "mean_rmse": float(np.round(np.mean(cv_rmse), 3)),
            "std_rmse": float(np.round(np.std(cv_rmse), 3)),
            "mean_mae": float(np.round(np.mean(cv_mae), 3)),
            "mean_r2": float(np.round(np.mean(cv_r2), 4)),
            "fold_rmse": [float(np.round(s, 2)) for s in cv_rmse]
        }

        print(f"Algorithm: {name}")
        print(f"  - 5-Fold CV Mean RMSE : {cv_results[name]['mean_rmse']:.3f} ± {cv_results[name]['std_rmse']:.3f} units")
        print(f"  - 5-Fold CV Mean MAE  : {cv_results[name]['mean_mae']:.3f} units")
        print(f"  - 5-Fold CV Mean R²   : {cv_results[name]['mean_r2']:.4f}")

    # Model Selection based on lowest Cross-Validation RMSE
    best_model_name = min(cv_results, key=lambda k: cv_results[k]["mean_rmse"])
    print("\n" + "-" * 70)
    print(f"Selected Champion Model: '{best_model_name}'")
    print(f"Selection Metric: Lowest 5-Fold Cross-Validation RMSE ({cv_results[best_model_name]['mean_rmse']} units)")
    print("Note: Selection performed strictly on training data without touching the test set.")
    print("-" * 70)

    # ---------------------------------------------------------
    # Train Selected Model Pipeline on Full Training Set
    # ---------------------------------------------------------
    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", models[best_model_name])
    ])
    final_pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------
    # STAGE 7: Final Evaluation on Held-Out Test Data
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 7: Evaluation on Held-Out Test Set (20% Sample)")
    print("=" * 70)

    y_pred = final_pipeline.predict(X_test)

    # Ensure predictions are physically non-negative
    y_pred = np.clip(y_pred, 0, None)

    mae = float(np.round(mean_absolute_error(y_test, y_pred), 3))
    mse = float(np.round(mean_squared_error(y_test, y_pred), 3))
    rmse = float(np.round(np.sqrt(mse), 3))
    r2 = float(np.round(r2_score(y_test, y_pred), 4))

    print(f"Test Set Evaluation Results for '{best_model_name}':")
    print(f"  - Mean Absolute Error (MAE)  : {mae} units")
    print(f"  - Mean Squared Error (MSE)   : {mse} (units²)")
    print(f"  - Root Mean Squared Error (RMSE): {rmse} units")
    print(f"  - R-squared Score (R²)        : {r2} (explains {r2*100:.2f}% of variance)")

    # ---------------------------------------------------------
    # STAGE 8: Evaluation Visualizations
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 8: Generating Model Evaluation Plots")
    print("=" * 70)

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, color="#0f4c81", alpha=0.7, edgecolors="k", s=50, label="Test Predictions")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#e63946", linestyle="--", linewidth=2, label="Ideal Prediction (y = x)")
    plt.title(f"Fig 5.1: Actual vs. Predicted Medicine Demand\n({best_model_name} on Test Data)")
    plt.xlabel("Actual Medicine Demand (units)")
    plt.ylabel("Predicted Medicine Demand (units)")
    plt.annotate(f"R² = {r2:.4f}\nRMSE = {rmse:.2f} units", xy=(0.05, 0.85), xycoords="axes fraction",
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#0f4c81", lw=1.5), fontsize=10)
    plt.legend()
    p_act = os.path.join(STATIC_PLOTS_DIR, "actual_vs_predicted.png")
    plt.savefig(p_act, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p_act}")

    # 2. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, color="#2a9d8f", alpha=0.7, edgecolors="k", s=50)
    plt.axhline(0, color="#e63946", linestyle="--", linewidth=2)
    plt.title(f"Fig 5.2: Residual Plot (Residuals vs. Predicted)\n({best_model_name})")
    plt.xlabel("Predicted Medicine Demand (units)")
    plt.ylabel("Residuals: Actual - Predicted (units)")
    plt.annotate(f"Mean Residual = {np.mean(residuals):.2f}\nStd Dev = {np.std(residuals):.2f}", xy=(0.05, 0.85),
                 xycoords="axes fraction", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#2a9d8f", lw=1.5), fontsize=10)
    p_res = os.path.join(STATIC_PLOTS_DIR, "residual_plot.png")
    plt.savefig(p_res, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] {p_res}")

    # 3. Feature Importance Plot (if Random Forest or tree-based)
    p_feat = os.path.join(STATIC_PLOTS_DIR, "feature_importance.png")
    regressor = final_pipeline.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        # Extract transformed feature names
        cat_encoder = final_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
        all_feature_names = numeric_features + cat_feature_names
        importances = regressor.feature_importances_

        # Top 12 most important features
        feat_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
        feat_df = feat_df.sort_values(by="Importance", ascending=False).head(12)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
        plt.title(f"Fig 5.3: Top Feature Importances ({best_model_name})")
        plt.xlabel("Gini Feature Importance")
        plt.ylabel("Transformed Feature")
        plt.savefig(p_feat, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [Saved] {p_feat}")
    else:
        # If Linear Regression was selected, plot standardized coefficients
        cat_encoder = final_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
        all_feature_names = numeric_features + cat_feature_names
        coefs = regressor.coef_

        coef_df = pd.DataFrame({"Feature": all_feature_names, "Importance": np.abs(coefs)})
        coef_df = coef_df.sort_values(by="Importance", ascending=False).head(12)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=coef_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False)
        plt.title(f"Fig 5.3: Top Absolute Model Coefficients ({best_model_name})")
        plt.xlabel("Absolute Regression Coefficient Magnitude")
        plt.ylabel("Transformed Feature")
        plt.savefig(p_feat, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [Saved] {p_feat}")

    # ---------------------------------------------------------
    # STAGE 9: Model Artifact & Metadata Persistence
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 9: Saving Model Pipeline & Metadata")
    print("=" * 70)

    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"  [Saved] Pipeline model saved to: {MODEL_PATH}")

    metadata = {
        "selected_algorithm": best_model_name,
        "selection_metric": "Lowest 5-Fold Cross-Validation RMSE on Training Data",
        "dataset_type": "Simulated Educational Demonstration Dataset",
        "stock_unit": "units",
        "sample_counts": {
            "total": int(len(X)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "train_ratio": 0.80,
            "test_ratio": 0.20
        },
        "feature_names": list(X.columns),
        "target_name": "Predicted Demand",
        "cross_validation_results": cv_results,
        "test_evaluation": {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        },
        "inventory_policy": {
            "safety_stock_formula": "Average Daily Usage * ceil(Supplier Delivery Time * 0.5)",
            "recommended_stock_formula": "Predicted Demand + Safety Stock",
            "reorder_quantity_formula": "max(0, Recommended Stock - Current Stock)",
            "stock_status_thresholds": {
                "Low Stock": "Current Stock < (Predicted Demand * 0.5)",
                "Overstock": "Current Stock > (Recommended Stock * 1.25)",
                "Sufficient Stock": "Otherwise"
            }
        }
    }

    return metadata

# ---------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------
def main():
    print("Starting Medicine Stock Prediction Machine Learning Pipeline...")
    
    # 1. Dataset Generation & Preparation
    df = prepare_dataset()
    
    # 2. Preprocessing & Inspection
    X, y, df_clean, prep_summary = preprocess_and_validate(df)
    
    # 3. Generate EDA Visualizations
    generate_eda_visualizations(df_clean)
    
    # 4. Train, Compare, and Evaluate Models
    model_metadata = train_and_compare_models(X, y)
    
    # Combine summaries and save to JSON
    complete_metadata = {
        "preprocessing_summary": prep_summary,
        "model_metadata": model_metadata
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(complete_metadata, f, indent=4)
    print(f"  [Saved] Complete metadata saved to: {METADATA_PATH}")

    print("\n" + "=" * 70)
    print("SUCCESS: Machine Learning Training Pipeline Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
