"""
app.py
======
Flask Web Application for:
Medicine Stock Prediction Using Linear Regression and Random Forest Regression

Serves the college ML laboratory demonstration interface, showcasing:
- The complete machine learning workflow (Stages 1 through 12)
- Real dataset inspection statistics & preprocessing summaries
- 6 Exploratory Data Analysis (EDA) visualizations
- Model comparison and cross-validation selection results
- Held-out test set evaluation metrics & residual plots
- Interactive medicine stock prediction form and inventory planning calculations
"""

import os
import json
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = "medicine_stock_prediction_secret_key"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(MODELS_DIR, "medicine_stock_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
DATA_PATH = os.path.join(DATA_DIR, "medicine_stock.csv")

# Global variables for loaded model and metadata
model_pipeline = None
metadata = None
first_five_records = []

def load_artifacts():
    """Loads saved model pipeline, metadata, and dataset preview."""
    global model_pipeline, metadata, first_five_records
    
    if os.path.exists(MODEL_PATH):
        try:
            model_pipeline = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model pipeline: {e}")
            model_pipeline = None

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            metadata = None

    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            first_five_records = df.head(5).to_dict(orient="records")
        except Exception as e:
            print(f"Error loading dataset preview: {e}")
            first_five_records = []

# Initial load on server startup
load_artifacts()

def calculate_inventory_policy(predicted_demand, current_stock, daily_usage, delivery_time):
    """
    Computes inventory replenishment recommendations based on documented inventory policy:
    1. Safety Stock Buffer: Protects against supplier delay & demand surges.
       Safety Stock = daily_usage * ceil(delivery_time * 0.5)
    2. Recommended Target Stock: Predicted demand + Safety stock.
    3. Reorder Quantity: max(0, Recommended Stock - Current Stock).
    4. Stock Status:
       - Low Stock: Current Stock < (Predicted Demand * 0.5)
       - Overstock: Current Stock > (Recommended Stock * 1.25)
       - Sufficient Stock: Otherwise
    """
    safety_stock = int(math.ceil(daily_usage * math.ceil(delivery_time * 0.5)))
    recommended_stock = int(math.ceil(predicted_demand + safety_stock))
    reorder_quantity = max(0, recommended_stock - int(current_stock))

    if current_stock < (predicted_demand * 0.5):
        stock_status = "Low Stock"
        status_class = "status-low"
        status_message = "Urgent: Current inventory covers less than 50% of projected requirement. Immediate reorder required."
    elif current_stock > (recommended_stock * 1.25):
        stock_status = "Overstock"
        status_class = "status-over"
        status_message = "Excess Inventory: Current stock exceeds 125% of target stock. Defer procurement to prevent holding costs or expiration."
    else:
        stock_status = "Sufficient Stock"
        status_class = "status-sufficient"
        status_message = "Balanced Inventory: Stock on hand comfortably meets projected requirements and lead time buffer."

    return {
        "safety_stock": safety_stock,
        "recommended_stock": recommended_stock,
        "reorder_quantity": reorder_quantity,
        "stock_status": stock_status,
        "status_class": status_class,
        "status_message": status_message
    }

@app.route("/", methods=["GET", "POST"])
def index():
    # If artifacts are missing, prompt user to run train_model.py
    if model_pipeline is None or metadata is None:
        return render_template(
            "index.html",
            artifacts_missing=True,
            error_message="Model artifacts not found! Please run 'python train_model.py' to generate the dataset, train models, and create visualizations."
        )

    prediction_result = None
    form_data = {
        "medicine_name": "Paracetamol",
        "medicine_category": "Pain Relief",
        "current_stock": "100",
        "previous_consumption": "250",
        "average_daily_usage": "20.0",
        "supplier_delivery_time": "5"
    }

    if request.method == "POST":
        # Extract and validate form inputs
        med_name = request.form.get("medicine_name", "").strip()
        med_cat = request.form.get("medicine_category", "").strip()
        curr_stock_raw = request.form.get("current_stock", "").strip()
        prev_cons_raw = request.form.get("previous_consumption", "").strip()
        daily_usage_raw = request.form.get("average_daily_usage", "").strip()
        deliv_time_raw = request.form.get("supplier_delivery_time", "").strip()

        form_data = {
            "medicine_name": med_name,
            "medicine_category": med_cat,
            "current_stock": curr_stock_raw,
            "previous_consumption": prev_cons_raw,
            "average_daily_usage": daily_usage_raw,
            "supplier_delivery_time": deliv_time_raw
        }

        # Validation checks
        errors = []
        if not med_name:
            errors.append("Medicine Name is required.")
        if not med_cat:
            errors.append("Medicine Category is required.")

        try:
            curr_stock = float(curr_stock_raw)
            if curr_stock < 0 or not math.isfinite(curr_stock):
                errors.append("Current Stock must be a non-negative number.")
        except ValueError:
            errors.append("Current Stock must be a valid number.")

        try:
            prev_cons = float(prev_cons_raw)
            if prev_cons < 0 or not math.isfinite(prev_cons):
                errors.append("Previous Consumption must be a non-negative number.")
        except ValueError:
            errors.append("Previous Consumption must be a valid number.")

        try:
            daily_usage = float(daily_usage_raw)
            if daily_usage <= 0 or not math.isfinite(daily_usage):
                errors.append("Average Daily Usage must be greater than 0.")
        except ValueError:
            errors.append("Average Daily Usage must be a valid number.")

        try:
            deliv_time = float(deliv_time_raw)
            if deliv_time <= 0 or not math.isfinite(deliv_time):
                errors.append("Supplier Delivery Time must be at least 1 day.")
        except ValueError:
            errors.append("Supplier Delivery Time must be a valid integer.")

        if errors:
            for err in errors:
                flash(err, "error")
        else:
            try:
                # Prepare single-row DataFrame matching the exact training columns
                input_df = pd.DataFrame([{
                    "Medicine Name": med_name,
                    "Medicine Category": med_cat,
                    "Current Stock": int(round(curr_stock)),
                    "Previous Consumption": int(round(prev_cons)),
                    "Average Daily Usage": float(daily_usage),
                    "Supplier Delivery Time": int(round(deliv_time))
                }])

                # Generate Machine Learning Model prediction
                pred_raw = model_pipeline.predict(input_df)[0]
                predicted_demand = max(0, int(round(pred_raw)))

                # Calculate inventory planning recommendations
                inventory_policy = calculate_inventory_policy(
                    predicted_demand=predicted_demand,
                    current_stock=int(round(curr_stock)),
                    daily_usage=daily_usage,
                    delivery_time=int(round(deliv_time))
                )

                prediction_result = {
                    "input": {
                        "medicine_name": med_name,
                        "medicine_category": med_cat,
                        "current_stock": int(round(curr_stock)),
                        "previous_consumption": int(round(prev_cons)),
                        "average_daily_usage": daily_usage,
                        "supplier_delivery_time": int(round(deliv_time))
                    },
                    "predicted_demand": predicted_demand,
                    "policy": inventory_policy
                }
            except Exception as ex:
                flash(f"Prediction failed: {str(ex)}", "error")

    return render_template(
        "index.html",
        artifacts_missing=False,
        metadata=metadata,
        first_five_records=first_five_records,
        prediction_result=prediction_result,
        form_data=form_data
    )

if __name__ == "__main__":
    # Runs locally on http://127.0.0.1:5000/
    print("Starting Flask web server on http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000, debug=False)
