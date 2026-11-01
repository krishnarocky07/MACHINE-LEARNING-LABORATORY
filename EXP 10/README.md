# Medicine Stock Prediction Using Linear Regression and Random Forest Regression

A complete, production-ready Machine Learning mini-project designed for hospital pharmacy inventory planning and college ML laboratory demonstration.

---

## 1. Project Description

**PharmaStock AI** is a local web application built with Python, Flask, Scikit-learn, and modern frontend technologies. It models pharmaceutical inventory dynamics to forecast future medicine stock requirements, avoid critical stock shortages, minimize expired medication wastage, and optimize reorder quantities for hospital pharmacies.

The project implements the complete, end-to-end Machine Learning lifecycle—from problem formulation and data inspection to automated pipeline preprocessing, 5-fold cross-validation model selection, residual diagnostics, and interactive web inference.

---

## 2. Project Objective

Hospital pharmacies must maintain adequate stocks of life-saving medications while avoiding excess holding costs and stock expiration. The objective of this project is to:
1. Formulate medicine demand forecasting as a continuous machine learning regression task.
2. Predict the required quantity of medicines (**Predicted Demand**) based on historical consumption, daily patient burn rate, current inventory levels, and supplier delivery lead times.
3. Apply a transparent, documented inventory planning policy to compute **Recommended Target Stock**, **Estimated Reorder Quantity**, and **Stock Status** (Low Stock, Sufficient Stock, Overstock).
4. Provide a clear, visual presentation suitable for academic viva, college laboratory evaluation, and portfolio presentation.

---

## 3. Dataset Source, Disclosure & Limitations

### Real vs. Simulated Dataset Disclosure
- **Statutory & Ethical Reality:** Genuine hospital pharmacy records containing daily dispensing quantities, patient bed occupancy rates, supplier contracts, and inventory shortages are legally confidential and protected under healthcare privacy regulations.
- **Demonstration Dataset:** To enable complete academic reproducibility, this project uses a carefully constructed, documented **Simulated Educational Demonstration Dataset** (`data/medicine_stock.csv`).
- **Data Fidelity:** The dataset contains **700 records** across **24 distinct medicines** and **8 therapeutic categories**. The generation logic models realistic hospital pharmacy relationships, including:
  - Base daily consumption per clinical therapeutic class.
  - Poisson-distributed supplier delivery lead times (1–14 days).
  - Seasonal demand fluctuations and stochastic random shocks.
  - Non-deterministic target formula ensuring genuine regression learning (no exact mathematical shortcut).
  - Non-negative constraints on all numerical inventory fields.
- **Limitations:** Simulated model metrics reflect the assumptions used during data generation and do not constitute clinical proof of performance in real hospital operations.

---

## 4. Input Features and Output Target

### Input Features (6)
1. **Medicine Name** (*Categorical*): Hospital formulary drug name (e.g., `Paracetamol`, `Amoxicillin`, `Metformin`, `Cetirizine`). Identifies specific therapeutic usage.
2. **Medicine Category** (*Categorical*): Therapeutic classification (e.g., `Pain Relief`, `Antibiotics`, `Diabetes Medicine`, `Vitamins`). Captures group-level consumption trends.
3. **Current Stock** (*Numerical, units*): Physical quantity of medicine currently available in the pharmacy store.
4. **Previous Consumption** (*Numerical, units*): Historical quantity dispensed over the previous 30-day review period.
5. **Average Daily Usage** (*Numerical, units/day*): Estimated daily consumption burn rate by inpatient and outpatient departments.
6. **Supplier Delivery Time** (*Numerical, days*): Lead time in days required for the distributor or manufacturer to replenish inventory.

### Model Target Variable
- **Predicted Demand** (*Numerical, units*): Continuous expected quantity of medicine required over the replenishment cycle.

### Additional Inventory Policy Outputs
- **Safety Stock Buffer** (*units*): Buffer to safeguard against distributor delays and emergency demand spikes.
- **Recommended Stock Quantity** (*units*): Ideal inventory target ($\text{Predicted Demand} + \text{Safety Stock}$).
- **Estimated Reorder Quantity** (*units*): Physical units to order immediately ($\max(0, \text{Recommended Stock} - \text{Current Stock})$).
- **Stock Status**:
  - 🔴 **Low Stock**: Current Stock $< 50\%$ of Predicted Demand (Critical reorder required).
  - 🟢 **Sufficient Stock**: Current Stock comfortably meets demand and buffer.
  - 🟡 **Overstock**: Current Stock $> 125\%$ of Recommended Stock (Holding cost & expiration risk).

---

## 5. Machine Learning Workflow Stages

The application systematically implements all 12 stages of the standard ML workflow:
1. **Problem Definition:** Formulated as a supervised continuous regression problem.
2. **Data Collection:** 700 realistic pharmaceutical records exported to CSV.
3. **Data Inspection:** Shape verification, data type checks, descriptive statistics, and category distributions.
4. **Data Preprocessing:** Missing-value checks, duplicate detection, and non-negative constraints.
5. **Exploratory Data Analysis (EDA):** Matplotlib and Seaborn graphs examining distributions, correlations, and lead-time patterns.
6. **Feature Selection:** 4 numerical and 2 categorical features chosen based on inventory replenishment theory.
7. **Train-Test Split:** 80% training set (560 samples) and 20% held-out test set (140 samples) with fixed `random_state=42`.
8. **Categorical Feature Encoding & Scaling:** `OneHotEncoder(handle_unknown='ignore')` and `StandardScaler()` encapsulated inside a `ColumnTransformer`.
9. **Model Training:** Fitting Linear Regression baseline and Random Forest Regressor ensemble.
10. **Model Comparison & Selection:** 5-fold cross-validation on training data only; model with lowest CV RMSE selected.
11. **Model Evaluation:** Held-out test set evaluation calculating MAE, MSE, RMSE, and $R^2$, paired with residual plots.
12. **Model Persistence & Deployment:** Model pipeline serialized with Joblib and deployed via a responsive Flask web application.

---

## 6. Preprocessing & Encoding Pipeline

To prevent data leakage, all scalers and encoders are fitted **strictly on the training data**:
- **Numerical Pipeline:** `StandardScaler()` applied to `Current Stock`, `Previous Consumption`, `Average Daily Usage`, and `Supplier Delivery Time`.
- **Categorical Pipeline:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to `Medicine Name` and `Medicine Category`.
- **Pipeline Integration:** Encapsulated in a Scikit-learn `Pipeline([('preprocessor', transformer), ('regressor', model)])`. This ensures incoming single-row prediction inputs are automatically transformed using identical statistical parameters.

---

## 7. Model Training & Cross-Validation Comparison

### Model 1: Linear Regression
- **Purpose:** Fast, interpretable parametric baseline. Models linear trends between past consumption, daily burn rate, and expected replenishment.

### Model 2: Random Forest Regressor
- **Configuration:** `n_estimators=100`, `random_state=42`.
- **Purpose:** Non-parametric ensemble capable of learning non-linear feature interactions and high-order thresholds.

### 5-Fold Cross-Validation Comparison (Training Data Only)

| Regression Algorithm | 5-Fold CV Mean RMSE | 5-Fold CV Mean MAE | 5-Fold CV Mean $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **35.15 ± 0.82 units** | **27.70 units** | **97.62%** | 🏆 **Selected Champion** |
| **Random Forest Regressor** | 40.94 ± 2.24 units | 31.58 units | 96.75% | Benchmarked |

*Note: Selection is based strictly on lowest cross-validation RMSE on the training partition. Linear Regression achieved the lowest cross-validation error and was exported for deployment.*

---

## 8. Test Set Evaluation Metrics

The champion model was evaluated once on the held-out 20% test partition (140 unseen samples):

| Metric | Measured Value | Definition & Unit Interpretation |
| :--- | :---: | :--- |
| **Mean Absolute Error (MAE)** | **28.89 units** | On average, predictions deviate by approximately 28.9 medicine units. |
| **Mean Squared Error (MSE)** | **1312.98 $\text{units}^2$** | Mean squared difference between actual and predicted demand. |
| **Root Mean Squared Error (RMSE)** | **36.24 units** | Standard deviation of residuals in original units (units of medicine). |
| **Coefficient of Determination ($R^2$)** | **0.9698 (96.98%)** | 96.98% of the total variance in medicine demand is explained by the model. |

---

## 9. Inventory Recommendation Calculation Policy

The inventory management policy is cleanly separated from the machine learning demand prediction:

1. **Safety Stock Buffer ($S_s$):**
   $$S_s = \lceil \text{Average Daily Usage} \times \lceil\text{Supplier Delivery Time} \times 0.5\rceil \rceil$$
   *Protects the hospital during the supplier replenishment lead time.*

2. **Recommended Target Stock ($S_r$):**
   $$S_r = \lceil \text{Predicted Demand} + S_s \rceil$$

3. **Estimated Reorder Quantity ($Q_r$):**
   $$Q_r = \max(0, S_r - \text{Current Stock})$$

4. **Stock Status:**
   - $\text{Current Stock} < 0.5 \times \text{Predicted Demand} \implies$ **Low Stock**
   - $\text{Current Stock} > 1.25 \times S_r \implies$ **Overstock**
   - Otherwise $\implies$ **Sufficient Stock**

---

## 10. Project Directory Structure

```text
MedicineStockPrediction/
│
├── app.py                     # Flask web server and routing logic
├── train_model.py             # Complete ML training, cross-validation & evaluation script
├── test_app.py                # Automated unit and integration test suite
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive project documentation
│
├── data/
│   └── medicine_stock.csv     # 700-record pharmaceutical inventory dataset
│
├── models/
│   ├── medicine_stock_model.pkl # Serialized Scikit-learn Pipeline
│   └── model_metadata.json      # Preprocessing stats, CV scores, and evaluation metrics
│
├── static/
│   ├── style.css              # Light-themed hospital pharmacy CSS
│   └── plots/                 # 9 Generated Matplotlib/Seaborn visualization figures
│       ├── demand_distribution.png
│       ├── stock_vs_demand.png
│       ├── category_demand.png
│       ├── consumption_analysis.png
│       ├── delivery_vs_demand.png
│       ├── feature_correlation.png
│       ├── actual_vs_predicted.png
│       ├── residual_plot.png
│       └── feature_importance.png
│
└── templates/
    └── index.html             # Main dashboard template with all 6 workflow figures
```

---

## 11. Installation & Setup Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your machine.
- Git (optional).

### Step 1: Open Terminal in Project Directory
```bash
cd MedicineStockPrediction
```

### Step 2: Create a Python Virtual Environment
**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 12. Running the Training Pipeline

Execute the automated training script to generate the dataset, train both models, execute cross-validation, evaluate on held-out test data, and produce all 9 diagnostic plots:

```bash
python train_model.py
```

### Expected Output Summary:
```text
STAGE 1: Problem Definition & Data Collection
STAGE 2: Data Inspection & Summary Statistics (700 rows, 7 cols)
STAGE 3: Data Preprocessing (0 missing, 0 duplicates, non-negative verified)
STAGE 4: Exploratory Data Analysis (6 EDA plots saved to static/plots/)
STAGE 5: Train-Test Split (560 train, 140 test)
STAGE 6: Cross-Validation & Model Selection (Linear Regression selected: CV RMSE 35.15)
STAGE 7: Evaluation on Held-Out Test Set (Test MAE: 28.89, RMSE: 36.24, R2: 0.9698)
STAGE 8: Evaluation Visualizations (Actual vs Predicted, Residuals, Coefficients saved)
STAGE 9: Saving Model Pipeline & Metadata (models/medicine_stock_model.pkl)
SUCCESS: Machine Learning Training Pipeline Completed Successfully!
```

---

## 13. Running the Flask Web Application

Start the local Flask development server:

```bash
python app.py
```

Open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 14. Running the Automated Test Suite

To verify all data structures, model loading, inventory policy logic, and web routes:

```bash
python test_app.py
```

Expected result:
```text
.......
Ran 7 tests in 0.08s
OK
```

---

## 15. How to Use the Prediction Form

1. Navigate to the **Interactive Medicine Stock Prediction** section (or click **Live Predictor** in the navigation bar).
2. Choose from the **Quick Presets** (e.g., *Low Stock Demo*, *Sufficient Stock Demo*, *Overstock Demo*) or enter custom values:
   - **Medicine Name:** e.g., `Paracetamol`
   - **Medicine Category:** e.g., `Pain Relief`
   - **Current Stock Quantity:** e.g., `100` units
   - **Previous Consumption:** e.g., `250` units
   - **Average Daily Usage:** e.g., `20.0` units/day
   - **Supplier Delivery Time:** e.g., `5` days
3. Click **Run ML Prediction & Inventory Optimization**.
4. Review the instant prediction card:
   - **ML Predicted Demand:** estimated consumption (e.g., ~554 units).
   - **Safety Stock Buffer:** lead time protection (60 units).
   - **Recommended Target Stock:** total target level (614 units).
   - **Recommended Reorder Qty:** units to purchase immediately (514 units).
   - **Stock Status Badge:** `Low Stock` with clear operational advice.

---

## 16. Common Troubleshooting

| Issue / Error | Cause | Resolution |
| :--- | :--- | :--- |
| `Model artifacts not found!` | `train_model.py` has not been executed yet. | Run `python train_model.py` to create the model `.pkl` and metadata JSON. |
| `Address already in use: 5000` | Another process is using port 5000. | Stop the other process or run `python app.py` on another port (e.g. `app.run(port=5001)`). |
| `Current Stock must be non-negative` | Negative number entered in form. | Enter positive integers or zero. |
| `ModuleNotFoundError` | Virtual environment not activated. | Activate `.venv` (`.venv\Scripts\activate` on Windows) and run `pip install -r requirements.txt`. |

---

## 17. Educational & Clinical Disclaimer

> **IMPORTANT:**
> This software is an **academic mini-project** created for machine learning laboratory demonstration and educational decision-support. 
> The dataset is a documented simulated demonstration dataset. 
> Predictions, target stock estimates, and reorder calculations should always be reviewed by certified clinical pharmacists and authorized healthcare procurement officers prior to actual purchasing or clinical deployment.
