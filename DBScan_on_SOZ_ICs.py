#!/usr/bin/env python3
# Standalone helper script for manually running DBSCAN on SOZ ICs.
# Not used by the web app: the app runs DBSCAN via final/pipeline.py,
# which calls IC_DBScan.generate_dbscan_visual with dynamic paths.

"""
Run DBSCAN-based SOZ localization for all ICs detected as SOZ.
"""

from pathlib import Path

import pandas as pd

from IC_DBScan import generate_dbscan_visual

# === CONFIG ===
DATA_DIR = Path("/home/local/ASUAD/abanerj3/Desktop/expertknowledge/DATA/ASUAI_001/MO/report/")
TEMPLATE_PATH = Path("/home/local/ASUAD/abanerj3/Desktop/expertknowledge/DATA/R.png")
RESULTS_FILE = Path(__file__).parent / "pipeline_results.csv"  # output of pipeline.py

def run_dbscan_for_all_soz():
    if not RESULTS_FILE.exists():
        print(f"[ERROR] {RESULTS_FILE} not found. Run pipeline.py first.")
        return
    
    # Load results
    df = pd.read_csv(RESULTS_FILE)
    soz_df = df[df["SOZ"] == True]
    
    if soz_df.empty:
        print("[INFO] No SOZ ICs detected. Nothing to process.")
        return
    
    print(f"[INFO] Found {len(soz_df)} SOZ IC(s): {soz_df['IC'].tolist()}")
    
    # Process each SOZ IC
    for _, row in soz_df.iterrows():
        ic_num = row["IC"]
        ic_filename = f"IC_{ic_num}_thresh.png"
        ic_path = DATA_DIR / ic_filename
        
        if not ic_path.exists():
            print(f"[WARNING] IC image not found: {ic_path}")
            continue
        
        print(f"\n[PROCESSING] Running DBSCAN on {ic_filename}")
        
        try:
            output_name = f"SOZ_localization_IC{ic_num}.png"
            output_path = DATA_DIR / output_name
            generate_dbscan_visual(
                main_image_path=str(ic_path),
                template_path=str(TEMPLATE_PATH),
                output_path=str(output_path),
                epsilon=3,
                min_samples=5,
                color_threshold=30,
                prioritize_red=True,
                use_contour_detection=True
            )
            print(f"[INFO] Localization saved for IC {ic_num} → {output_path}")

        except Exception as e:
            print(f"[ERROR] Failed to process IC {ic_num}: {e}")

if __name__ == "__main__":
    run_dbscan_for_all_soz()
