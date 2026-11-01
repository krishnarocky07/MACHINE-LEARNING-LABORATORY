"""
test_app.py
===========
Unit and integration tests for Medicine Stock Prediction web application.
Verifies:
1. Dataset exists and has required columns.
2. Preprocessor and model pipeline load successfully and produce valid predictions.
3. Inventory policy logic correctly computes safety stock, target stock, reorder qty, and stock status.
4. Flask routes return 200 OK and render required workflow elements and prediction cards.
5. Error handling for negative/invalid inputs works as expected.
"""

import os
import unittest
import pandas as pd
import joblib
from app import app, calculate_inventory_policy, MODEL_PATH, DATA_PATH, METADATA_PATH

class TestMedicineStockPrediction(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists(DATA_PATH), "medicine_stock.csv must exist.")
        self.assertTrue(os.path.exists(MODEL_PATH), "medicine_stock_model.pkl must exist.")
        self.assertTrue(os.path.exists(METADATA_PATH), "model_metadata.json must exist.")

    def test_dataset_structure(self):
        df = pd.read_csv(DATA_PATH)
        self.assertGreaterEqual(len(df), 500, "Dataset must have at least 500 records.")
        required_cols = [
            "Medicine Name", "Medicine Category", "Current Stock",
            "Previous Consumption", "Average Daily Usage",
            "Supplier Delivery Time", "Predicted Demand"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column '{col}' must be present in dataset.")
            
        # Target must be non-negative
        self.assertTrue((df["Predicted Demand"] >= 0).all(), "Predicted Demand must be non-negative.")
        self.assertTrue((df["Current Stock"] >= 0).all(), "Current Stock must be non-negative.")

    def test_model_pipeline_prediction(self):
        pipeline = joblib.load(MODEL_PATH)
        test_df = pd.DataFrame([{
            "Medicine Name": "Paracetamol",
            "Medicine Category": "Pain Relief",
            "Current Stock": 100,
            "Previous Consumption": 250,
            "Average Daily Usage": 20.0,
            "Supplier Delivery Time": 5
        }])
        pred = pipeline.predict(test_df)[0]
        self.assertIsInstance(float(pred), float)
        self.assertGreater(pred, 0, "Predicted demand should be positive for realistic usage.")

    def test_inventory_policy(self):
        # Case 1: Low stock scenario
        res_low = calculate_inventory_policy(
            predicted_demand=500, current_stock=100, daily_usage=20.0, delivery_time=5
        )
        self.assertEqual(res_low["stock_status"], "Low Stock")
        self.assertGreater(res_low["reorder_quantity"], 0)
        self.assertEqual(res_low["safety_stock"], 60) # 20 * ceil(5 * 0.5) = 20 * 3 = 60
        self.assertEqual(res_low["recommended_stock"], 560)
        self.assertEqual(res_low["reorder_quantity"], 460)

        # Case 2: Overstock scenario
        res_over = calculate_inventory_policy(
            predicted_demand=500, current_stock=800, daily_usage=20.0, delivery_time=5
        )
        self.assertEqual(res_over["stock_status"], "Overstock")
        self.assertEqual(res_over["reorder_quantity"], 0)

        # Case 3: Sufficient stock scenario
        res_suff = calculate_inventory_policy(
            predicted_demand=500, current_stock=550, daily_usage=20.0, delivery_time=5
        )
        self.assertEqual(res_suff["stock_status"], "Sufficient Stock")

    def test_flask_get_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("PharmaStock AI", content)
        self.assertIn("FIG. 1", content)
        self.assertIn("FIG. 2", content)
        self.assertIn("FIG. 3", content)
        self.assertIn("FIG. 4", content)
        self.assertIn("FIG. 5", content)
        self.assertIn("FIG. 6", content)

    def test_flask_post_valid_prediction(self):
        post_data = {
            "medicine_name": "Paracetamol",
            "medicine_category": "Pain Relief",
            "current_stock": "100",
            "previous_consumption": "250",
            "average_daily_usage": "20.0",
            "supplier_delivery_time": "5"
        }
        response = self.client.post("/", data=post_data)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Prediction Output", content)
        self.assertIn("ML Predicted Demand", content)
        self.assertIn("Recommended Target Stock", content)
        self.assertIn("Recommended Reorder Qty", content)
        self.assertIn("Low Stock", content)

    def test_flask_post_invalid_input(self):
        post_data = {
            "medicine_name": "Paracetamol",
            "medicine_category": "Pain Relief",
            "current_stock": "-50",  # Invalid negative
            "previous_consumption": "250",
            "average_daily_usage": "20.0",
            "supplier_delivery_time": "5"
        }
        response = self.client.post("/", data=post_data)
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Current Stock must be a non-negative number.", content)

if __name__ == "__main__":
    unittest.main()
