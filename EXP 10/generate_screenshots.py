"""
generate_screenshots.py
=======================
Generates high-resolution screenshots of the VS Code Explorer tree
and web application interface for inclusion in the project report PDF.
"""

import os
import time
import subprocess
import threading
import pandas as pd
from app import app, calculate_inventory_policy, metadata, first_five_records, model_pipeline
from flask import render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "static", "screenshots")
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def generate_vscode_screenshot():
    html_content = """<!DOCTYPE html>
<html>
<head>
<style>
  body {
    margin: 0; padding: 0; background-color: #181818; color: #cccccc;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px; width: 330px; height: 500px; overflow: hidden;
  }
  .sidebar { background-color: #181818; padding: 12px 14px; height: 100%; box-sizing: border-box; }
  .title { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; color: #bbbbbb; margin-bottom: 8px; text-transform: uppercase; }
  .root-folder { font-size: 11px; font-weight: 700; color: #ffffff; display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
  .tree { list-style: none; padding-left: 12px; margin: 0; }
  .tree li { padding: 3px 0; display: flex; align-items: center; gap: 6px; white-space: nowrap; }
  .icon { font-size: 13px; width: 16px; text-align: center; display: inline-block; }
  .folder { color: #dcb67a; font-weight: 600; }
  .py { color: #519aba; }
  .csv { color: #89d185; }
  .json { color: #cbcb41; }
  .pkl { color: #e37933; }
  .css { color: #42a5f5; }
  .html { color: #e44d26; }
  .md { color: #42a5f5; }
  .nested { padding-left: 16px; }
</style>
</head>
<body>
<div class="sidebar">
  <div class="title">Explorer</div>
  <div class="root-folder">
    <span>&#9660;</span>
    <span>MEDICINESTOCKPREDICTION</span>
  </div>
  <ul class="tree">
    <li><span class="icon folder">&#128193;</span> <span class="folder">data</span></li>
    <li class="nested"><span class="icon csv">&#128196;</span> medicine_stock.csv</li>
    <li><span class="icon folder">&#128193;</span> <span class="folder">models</span></li>
    <li class="nested"><span class="icon pkl">&#9881;</span> medicine_stock_model.pkl</li>
    <li class="nested"><span class="icon json">&#128203;</span> model_metadata.json</li>
    <li><span class="icon folder">&#128193;</span> <span class="folder">static</span></li>
    <li class="nested"><span class="icon folder">&#128193;</span> <span class="folder">plots</span></li>
    <li class="nested"><span class="icon css">&#127912;</span> style.css</li>
    <li><span class="icon folder">&#128193;</span> <span class="folder">templates</span></li>
    <li class="nested"><span class="icon html">&#127760;</span> index.html</li>
    <li><span class="icon py">&#128013;</span> app.py</li>
    <li><span class="icon md">&#128214;</span> README.md</li>
    <li><span class="icon json">&#128221;</span> requirements.txt</li>
    <li><span class="icon py">&#128013;</span> test_app.py</li>
    <li><span class="icon py">&#128013;</span> train_model.py</li>
  </ul>
</div>
</body>
</html>
"""
    tmp_path = os.path.join(SCREENSHOTS_DIR, "vscode_tree.html")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    img_path = os.path.join(SCREENSHOTS_DIR, "vscode_tree.png")
    cmd = [
        EDGE_PATH, "--headless=new", "--disable-gpu", "--window-size=340,510",
        f"--screenshot={img_path}", f"file:///{tmp_path.replace(os.sep, '/')}"
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("  [Generated] vscode_tree.png")

def render_html_to_screenshot(rendered_html, out_filename, window_size="1200,800"):
    tmp_path = os.path.join(SCREENSHOTS_DIR, "temp_render.html")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    img_path = os.path.join(SCREENSHOTS_DIR, out_filename)
    cmd = [
        EDGE_PATH, "--headless=new", "--disable-gpu", f"--window-size={window_size}",
        f"--screenshot={img_path}", f"file:///{tmp_path.replace(os.sep, '/')}"
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"  [Generated] {out_filename}")

def generate_web_screenshots():
    # Fix stylesheet and plot paths for local file rendering
    style_path = os.path.join(BASE_DIR, "static", "style.css").replace("\\", "/")
    plots_dir = os.path.join(BASE_DIR, "static", "plots").replace("\\", "/")

    with app.test_request_context():
        # 1. Base Homepage Preview
        base_html = render_template(
            "index.html",
            artifacts_missing=False,
            metadata=metadata,
            first_five_records=first_five_records,
            prediction_result=None,
            form_data={
                "medicine_name": "Paracetamol",
                "medicine_category": "Pain Relief",
                "current_stock": "100",
                "previous_consumption": "250",
                "average_daily_usage": "20.0",
                "supplier_delivery_time": "5"
            }
        )
        base_html = base_html.replace('/static/style.css', f'file:///{style_path}')
        base_html = base_html.replace('/static/plots/', f'file:///{plots_dir}/')
        render_html_to_screenshot(base_html, "fig2_web_interface.png", "1240,920")

        # 2. Prediction Result: Low Stock (Paracetamol)
        pred_low_df = pd.DataFrame([{
            "Medicine Name": "Paracetamol",
            "Medicine Category": "Pain Relief",
            "Current Stock": 100,
            "Previous Consumption": 250,
            "Average Daily Usage": 20.0,
            "Supplier Delivery Time": 5
        }])
        pred_low_val = int(round(model_pipeline.predict(pred_low_df)[0]))
        policy_low = calculate_inventory_policy(pred_low_val, 100, 20.0, 5)

        low_html = render_template(
            "index.html",
            artifacts_missing=False,
            metadata=metadata,
            first_five_records=first_five_records,
            prediction_result={
                "input": {
                    "medicine_name": "Paracetamol",
                    "medicine_category": "Pain Relief",
                    "current_stock": 100,
                    "previous_consumption": 250,
                    "average_daily_usage": 20.0,
                    "supplier_delivery_time": 5
                },
                "predicted_demand": pred_low_val,
                "policy": policy_low
            },
            form_data={
                "medicine_name": "Paracetamol",
                "medicine_category": "Pain Relief",
                "current_stock": "100",
                "previous_consumption": "250",
                "average_daily_usage": "20.0",
                "supplier_delivery_time": "5"
            }
        )
        low_html = low_html.replace('/static/style.css', f'file:///{style_path}')
        low_html = low_html.replace('/static/plots/', f'file:///{plots_dir}/')
        render_html_to_screenshot(low_html, "fig10_prediction_low_stock.png", "1240,950")

        # 3. Prediction Result: Sufficient Stock (Amoxicillin)
        pred_suff_df = pd.DataFrame([{
            "Medicine Name": "Amoxicillin",
            "Medicine Category": "Antibiotics",
            "Current Stock": 650,
            "Previous Consumption": 600,
            "Average Daily Usage": 22.0,
            "Supplier Delivery Time": 4
        }])
        pred_suff_val = int(round(model_pipeline.predict(pred_suff_df)[0]))
        policy_suff = calculate_inventory_policy(pred_suff_val, 650, 22.0, 4)

        suff_html = render_template(
            "index.html",
            artifacts_missing=False,
            metadata=metadata,
            first_five_records=first_five_records,
            prediction_result={
                "input": {
                    "medicine_name": "Amoxicillin",
                    "medicine_category": "Antibiotics",
                    "current_stock": 650,
                    "previous_consumption": 600,
                    "average_daily_usage": 22.0,
                    "supplier_delivery_time": 4
                },
                "predicted_demand": pred_suff_val,
                "policy": policy_suff
            },
            form_data={
                "medicine_name": "Amoxicillin",
                "medicine_category": "Antibiotics",
                "current_stock": "650",
                "previous_consumption": "600",
                "average_daily_usage": "22.0",
                "supplier_delivery_time": "4"
            }
        )
        suff_html = suff_html.replace('/static/style.css', f'file:///{style_path}')
        suff_html = suff_html.replace('/static/plots/', f'file:///{plots_dir}/')
        render_html_to_screenshot(suff_html, "fig11_prediction_sufficient_stock.png", "1240,950")

        # 4. Prediction Result: Overstock (Vitamin C)
        pred_over_df = pd.DataFrame([{
            "Medicine Name": "Vitamin C",
            "Medicine Category": "Vitamins",
            "Current Stock": 1400,
            "Previous Consumption": 850,
            "Average Daily Usage": 30.0,
            "Supplier Delivery Time": 3
        }])
        pred_over_val = int(round(model_pipeline.predict(pred_over_df)[0]))
        policy_over = calculate_inventory_policy(pred_over_val, 1400, 30.0, 3)

        over_html = render_template(
            "index.html",
            artifacts_missing=False,
            metadata=metadata,
            first_five_records=first_five_records,
            prediction_result={
                "input": {
                    "medicine_name": "Vitamin C",
                    "medicine_category": "Vitamins",
                    "current_stock": 1400,
                    "previous_consumption": 850,
                    "average_daily_usage": 30.0,
                    "supplier_delivery_time": 3
                },
                "predicted_demand": pred_over_val,
                "policy": policy_over
            },
            form_data={
                "medicine_name": "Vitamin C",
                "medicine_category": "Vitamins",
                "current_stock": "1400",
                "previous_consumption": "850",
                "average_daily_usage": "30.0",
                "supplier_delivery_time": "3"
            }
        )
        over_html = over_html.replace('/static/style.css', f'file:///{style_path}')
        over_html = over_html.replace('/static/plots/', f'file:///{plots_dir}/')
        render_html_to_screenshot(over_html, "fig12_prediction_overstock.png", "1240,950")

def main():
    print("Generating report visual assets...")
    generate_vscode_screenshot()
    generate_web_screenshots()
    print("All visual assets generated successfully.")

if __name__ == "__main__":
    main()
