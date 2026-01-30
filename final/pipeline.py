#!/usr/bin/env python3
"""
Pipeline Script for Expert Knowledge Integrator
Runs both CNN and Knowledge Learning models, matches predictions, and applies SOZ logic.

Logic:
- For each IC we combine the CNN label (0/1) and Knowledge Integrator prediction (1/3).
- If DL label == 1 and knowledge prediction == 3 ➝ SOZ.
- If DL label == 0, knowledge prediction == 3, and P(class 3) > 0.9 ➝ SOZ.
- Otherwise the IC is NOT SOZ.
- If any IC is SOZ, the patient is flagged as SOZ.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from IC_DBScan import generate_dbscan_visual
except Exception as exc:  # pragma: no cover - best-effort import
    generate_dbscan_visual = None
    print(f"[WARN] Unable to import DBSCAN helper: {exc}")
DEFAULT_CASE_ROOT_DIR = BASE_DIR
DEFAULT_IMAGE_DIR = BASE_DIR / "images"
DEFAULT_CNN_MODEL_PATH = BASE_DIR / "CNN_modelTrainedPCH"
DEFAULT_CNN_OUTPUT = BASE_DIR / "predictions_DL.csv"
DEFAULT_TRAINED_MODEL_PATH = BASE_DIR / "trained_model.joblib"
DEFAULT_KL_OUTPUT = BASE_DIR / "predictions_KL.csv"
DEFAULT_FEATURES_OUTPUT = BASE_DIR / "ic_trainingdata_from_mat.csv"
DEFAULT_EXPLANATIONS_OUTPUT = BASE_DIR / "ic_explanations.csv"
DBSCAN_OUTPUT_ROOT = BASE_DIR / "dbscan_outputs"
DBSCAN_METADATA_FILE = BASE_DIR / "dbscan_outputs.json"
VOXELINFO_SCRIPT = REPO_ROOT / "VoxelInfoNewWhite.py"
EXPLANATION_SCRIPT = REPO_ROOT / "generate_ic_explanations.py"

CASE_ROOT_DIR = Path(os.environ.get("CASE_ROOT_DIR", DEFAULT_CASE_ROOT_DIR))
IMAGE_DIR = Path(os.environ.get("CNN_IMAGE_DIR", CASE_ROOT_DIR / "MO" / "report"))
if not IMAGE_DIR.exists():
    IMAGE_DIR = DEFAULT_IMAGE_DIR

CNN_MODEL_PATH = Path(os.environ.get("CNN_MODEL_PATH", DEFAULT_CNN_MODEL_PATH))
CNN_OUTPUT_CSV = Path(os.environ.get("CNN_OUTPUT_CSV", DEFAULT_CNN_OUTPUT))

TRAINED_MODEL_PATH = Path(os.environ.get("TRAINED_MODEL_PATH", DEFAULT_TRAINED_MODEL_PATH))
KNOWLEDGE_SUBJECT_DIR = Path(os.environ.get("KNOWLEDGE_SUBJECT_DIR", CASE_ROOT_DIR))
WORKSPACE_ROOT_DIR = os.environ.get("WORKSPACE_ROOT_DIR", str(CASE_ROOT_DIR))
WORKSPACE_FILE = os.environ.get("WORKSPACE_FILE")
KNOWLEDGE_OUTPUT_CSV = Path(os.environ.get("KNOWLEDGE_OUTPUT_CSV", DEFAULT_KL_OUTPUT))
FEATURES_OUTPUT_CSV = Path(os.environ.get("KNOWLEDGE_FEATURES_CSV", DEFAULT_FEATURES_OUTPUT))
if not FEATURES_OUTPUT_CSV.is_absolute():
    FEATURES_OUTPUT_CSV = BASE_DIR / FEATURES_OUTPUT_CSV
EXPLANATIONS_OUTPUT_CSV = Path(os.environ.get("KNOWLEDGE_EXPLANATIONS_CSV", DEFAULT_EXPLANATIONS_OUTPUT))
if not EXPLANATIONS_OUTPUT_CSV.is_absolute():
    EXPLANATIONS_OUTPUT_CSV = BASE_DIR / EXPLANATIONS_OUTPUT_CSV


def sanitize_case_id(case_id: Optional[str]) -> str:
    sanitized = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in (case_id or '').strip())
    return sanitized or "case"


CASE_ID = sanitize_case_id(os.environ.get("CASE_ID", CASE_ROOT_DIR.name))
DBSCAN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def run_cnn_evaluation():
    """
    Run CNN_Evaluate_for_labels.py and save output with _DL suffix
    """
    print("=" * 60)
    print("STEP 1: Running CNN Evaluation")
    print("=" * 60)
    
    # Get the current directory
    current_dir = BASE_DIR
    cnn_script = current_dir / "CNN_Evaluate_for_labels.py"
    
    if not cnn_script.exists():
        raise FileNotFoundError(f"CNN_Evaluate_for_labels.py not found at {cnn_script}")
    
    try:
        # Run the CNN evaluation script
        env = os.environ.copy()
        env.update({
            "CNN_IMAGE_DIR": str(IMAGE_DIR),
            "CNN_MODEL_PATH": str(CNN_MODEL_PATH),
            "CNN_OUTPUT_CSV": str(CNN_OUTPUT_CSV),
        })
        result = subprocess.run(
            [sys.executable, str(cnn_script)],
            capture_output=True,
            text=True,
            cwd=current_dir,
            env=env,
        )
        
        if result.returncode != 0:
            print(f"Error running CNN evaluation: {result.stderr}")
            return None
            
        print("CNN evaluation completed successfully!")
        print(result.stdout)
        
        if CNN_OUTPUT_CSV.exists():
            print(f"CNN predictions saved as: {CNN_OUTPUT_CSV}")
            return CNN_OUTPUT_CSV
        else:
            print(f"Warning: CNN output not found at {CNN_OUTPUT_CSV}")
            return None
            
    except Exception as e:
        print(f"Error running CNN evaluation: {e}")
        return None

def run_inference():
    """
    Run inference.py and save output with _KL suffix
    """
    print("\n" + "=" * 60)
    print("STEP 2: Running Knowledge Learning Inference")
    print("=" * 60)
    
    current_dir = BASE_DIR
    inference_script = current_dir / "inference.py"
    
    if not inference_script.exists():
        raise FileNotFoundError(f"inference.py not found at {inference_script}")
    
    try:
        # Run the inference script
        env = os.environ.copy()
        env.update({
            "KNOWLEDGE_SUBJECT_DIR": str(KNOWLEDGE_SUBJECT_DIR),
            "TRAINED_MODEL_PATH": str(TRAINED_MODEL_PATH),
            "WORKSPACE_ROOT_DIR": WORKSPACE_ROOT_DIR,
            "KNOWLEDGE_OUTPUT_CSV": str(KNOWLEDGE_OUTPUT_CSV),
            "KNOWLEDGE_FEATURES_CSV": str(FEATURES_OUTPUT_CSV),
        })
        if WORKSPACE_FILE:
            env["WORKSPACE_FILE"] = WORKSPACE_FILE

        result = subprocess.run(
            [sys.executable, str(inference_script)],
            capture_output=True,
            text=True,
            cwd=current_dir,
            env=env,
        )
        
        if result.returncode != 0:
            print(f"Error running inference: {result.stderr}")
            return None
            
        print("Inference completed successfully!")
        print(result.stdout)
        
        if KNOWLEDGE_OUTPUT_CSV.exists():
            print(f"KL predictions saved as: {KNOWLEDGE_OUTPUT_CSV}")
            return KNOWLEDGE_OUTPUT_CSV
        else:
            print(f"Warning: Knowledge output not found at {KNOWLEDGE_OUTPUT_CSV}")
            return None
            
    except Exception as e:
        print(f"Error running inference: {e}")
        return None


def resolve_template_path() -> Optional[Path]:
    explicit = os.environ.get("VOXELINFO_TEMPLATE_PATH")
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate

    candidates = [
        REPO_ROOT / "R.png",
        BASE_DIR / "R.png",
        CASE_ROOT_DIR / "R.png",
        CASE_ROOT_DIR / "MO" / "R.png",
        CASE_ROOT_DIR / "report" / "R.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_voxelinfo(workspace_output: Path) -> Optional[Path]:
    """
    Run VoxelInfoNewWhite.py to generate a workspace .mat file.
    """
    if not VOXELINFO_SCRIPT.exists():
        print(f"[WARN] VoxelInfoNewWhite.py not found at {VOXELINFO_SCRIPT}. Skipping MAT generation.")
        return None

    template_path = resolve_template_path()
    if not template_path:
        print("[WARN] Template image (R.png) not found. Skipping MAT generation.")
        return None

    env = os.environ.copy()
    env.update({
        "VOXELINFO_SUBJECT_DIR": str(CASE_ROOT_DIR),
        "VOXELINFO_REPORT_DIR": str(IMAGE_DIR),
        "VOXELINFO_TEMPLATE_PATH": str(template_path),
        "VOXELINFO_OUTPUT_DIR": str(workspace_output.parent),
        "VOXELINFO_WORKSPACE_NAME": workspace_output.name,
    })

    if os.environ.get("VOXELINFO_LABELS_PATH"):
        env["VOXELINFO_LABELS_PATH"] = os.environ.get("VOXELINFO_LABELS_PATH")

    result = subprocess.run(
        [sys.executable, str(VOXELINFO_SCRIPT)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    if result.returncode != 0:
        print(f"Error running VoxelInfoNewWhite.py: {result.stderr}")
        return None

    if workspace_output.exists():
        print(f"Workspace generated at: {workspace_output}")
        return workspace_output

    print(f"[WARN] Expected workspace file not found: {workspace_output}")
    return None


def run_explanations(input_csv: Path, output_csv: Path) -> Optional[Path]:
    """
    Run generate_ic_explanations.py to produce per-IC explanations.
    """
    if not EXPLANATION_SCRIPT.exists():
        print(f"[WARN] generate_ic_explanations.py not found at {EXPLANATION_SCRIPT}. Skipping explanations.")
        return None

    if not input_csv.exists():
        print(f"[WARN] Feature CSV not found at {input_csv}. Skipping explanations.")
        return None

    result = subprocess.run(
        [sys.executable, str(EXPLANATION_SCRIPT), str(input_csv), str(output_csv)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print(f"Error generating explanations: {result.stderr}")
        return None

    if output_csv.exists():
        print(f"Explanations saved to: {output_csv}")
        return output_csv

    print(f"[WARN] Explanation CSV not found at {output_csv}")
    return None

def load_and_match_predictions(dl_file, kl_file):
    """
    Load both CSV files and match predictions by IC number
    """
    print("\n" + "=" * 60)
    print("STEP 3: Loading and Matching Predictions")
    print("=" * 60)
    
    try:
        # Load DL predictions (CNN)
        dl_df = pd.read_csv(dl_file)
        print(f"DL predictions loaded: {len(dl_df)} records")
        print("DL columns:", dl_df.columns.tolist())
        print("DL sample:")
        print(dl_df.head())
        
        # Load KL predictions (Knowledge Learning)
        kl_df = pd.read_csv(kl_file)
        print(f"\nKL predictions loaded: {len(kl_df)} records")
        print("KL columns:", kl_df.columns.tolist())
        print("KL sample:")
        print(kl_df.head())
        
        # Match by IC number
        # DL has 'IC' column, KL has 'IC_number' column
        merged_df = pd.merge(dl_df, kl_df, left_on='IC', right_on='IC_number', how='inner')
        
        print(f"\nMerged predictions: {len(merged_df)} records")
        print("Merged columns:", merged_df.columns.tolist())
        
        return merged_df
        
    except Exception as e:
        print(f"Error loading/matching predictions: {e}")
        return None

def apply_soz_logic(merged_df, explanation_map: Optional[Dict[int, str]] = None):
    """
    Apply the SOZ logic to matched predictions
    """
    print("\n" + "=" * 60)
    print("STEP 4: Applying SOZ Logic")
    print("=" * 60)
    
    soz_results = []
    
    for idx, row in merged_df.iterrows():
        ic_num = row['IC']
        dl_label = row['Label']  # DL prediction (0 or 1)
        kl_prediction = row['prediction']  # KL prediction (1 or 3)
        prob_class_1 = row.get('prob_class_1')
        prob_class_3 = row.get('prob_class_3')
        
        # Apply updated SOZ logic from requirements
        soz = False
        reason = ""

        if dl_label == 1 and kl_prediction == 3:
            soz = True
            reason = "DL label 1 with KL prediction 3"
        elif (
            dl_label == 0
            and kl_prediction == 3
            and prob_class_3 is not None
            and prob_class_3 > 0.9
        ):
            soz = True
            reason = f"High-confidence KL prediction 3 (conf={prob_class_3:.3f}) with DL label 0"
        else:
            soz = False
            conf1 = f"{prob_class_1:.3f}" if prob_class_1 is not None else "N/A"
            conf3 = f"{prob_class_3:.3f}" if prob_class_3 is not None else "N/A"
            reason = f"DL={dl_label}, KL={kl_prediction}, conf_1={conf1}, conf_3={conf3}"
        
        explanation = explanation_map.get(int(ic_num)) if explanation_map else None
        soz_results.append({
            'IC': ic_num,
            'DL_Label': dl_label,
            'KL_Prediction': kl_prediction,
            'Prob_Class_1': prob_class_1,
            'Prob_Class_3': prob_class_3,
            'SOZ': soz,
            'Reason': reason,
            'Explanation': explanation
        })

        conf1 = f"{prob_class_1:.3f}" if prob_class_1 is not None else "N/A"
        conf3 = f"{prob_class_3:.3f}" if prob_class_3 is not None else "N/A"
        print(f"IC {ic_num}: DL={dl_label}, KL={kl_prediction}, conf_1={conf1}, conf_3={conf3} -> {'SOZ' if soz else 'NOT SOZ'} ({reason})")
    
    return soz_results


def find_ic_image_file(ic_number: int) -> Optional[Path]:
    """Locate the source IC image for DBSCAN visualization."""
    if not IMAGE_DIR.exists():
        return None

    normalized = f"IC_{int(ic_number)}"
    candidate_patterns = [
        f"{normalized}_thresh.png",
        f"{normalized}_thresh.jpg",
        f"{normalized}_thresh.jpeg",
        f"{normalized}_thresh.PNG",
        f"{normalized}_thresh.JPG",
        f"{normalized}_thresh.JPEG",
    ]

    for pattern in candidate_patterns:
        candidate = IMAGE_DIR / pattern
        if candidate.exists():
            return candidate

    matches = sorted(IMAGE_DIR.glob(f"{normalized}*_thresh*"))
    return matches[0] if matches else None


def write_dbscan_metadata(outputs: List[Dict[str, Any]]):
    payload = {
        "caseId": CASE_ID,
        "outputs": outputs,
    }
    with open(DBSCAN_METADATA_FILE, "w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2)


def generate_dbscan_assets(soz_results: List[Dict[str, Any]]):
    """Run DBSCAN localization for each SOZ IC and persist visualization paths."""
    outputs: List[Dict[str, Any]] = []

    if not soz_results:
        write_dbscan_metadata(outputs)
        return outputs

    if generate_dbscan_visual is None:
        print("[WARN] DBSCAN helper not available; skipping localization export.")
        write_dbscan_metadata(outputs)
        return outputs

    case_dir = DBSCAN_OUTPUT_ROOT / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in case_dir.glob("*.png"):
        try:
            stale_file.unlink()
        except OSError:
            pass

    for result in soz_results:
        ic_num = int(result.get('IC'))
        if not result.get('SOZ'):
            continue

        source_path = find_ic_image_file(ic_num)
        if not source_path:
            print(f"[WARN] Could not find IC image for IC {ic_num}; skipping DBSCAN output")
            continue

        output_file = case_dir / f"IC_{ic_num}_dbscan.png"
        try:
            generate_dbscan_visual(
                main_image_path=str(source_path),
                output_path=str(output_file),
                use_contour_detection=True,
                epsilon=3,
                min_samples=5,
                color_threshold=30,
                prioritize_red=True,
            )
            outputs.append({
                "ic": ic_num,
                "relative_path": f"{CASE_ID}/{output_file.name}",
            })
            print(f"[INFO] DBSCAN visualization created for IC {ic_num}: {output_file}")
        except Exception as exc:
            print(f"[WARN] Failed to generate DBSCAN visualization for IC {ic_num}: {exc}")

    write_dbscan_metadata(outputs)
    return outputs

def main():
    """
    Main pipeline execution
    """
    global WORKSPACE_FILE
    print("EXPERT KNOWLEDGE INTEGRATOR - PIPELINE")
    print("=" * 60)
    print(f"Case root: {CASE_ROOT_DIR}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Knowledge subject directory: {KNOWLEDGE_SUBJECT_DIR}")
    print(f"Workspace root: {WORKSPACE_ROOT_DIR}")
    print(f"CNN model path: {CNN_MODEL_PATH}")
    print(f"Trained model path: {TRAINED_MODEL_PATH}")
    
    current_dir = BASE_DIR
    
    # Step 1: Run CNN evaluation
    dl_file = run_cnn_evaluation()
    if dl_file is None:
        print("Failed to run CNN evaluation. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Generate workspace MAT file (VoxelInfoNewWhite), unless it already exists
    workspace_output_dir = Path(WORKSPACE_ROOT_DIR)
    workspace_output_dir.mkdir(parents=True, exist_ok=True)
    workspace_output = workspace_output_dir / f"Workspace-{CASE_ID}V4.mat"
    existing_workspace = None
    repo_workspace = REPO_ROOT / f"Workspace-{CASE_ID}V4.mat"
    if WORKSPACE_FILE:
        candidate = Path(WORKSPACE_FILE)
        if candidate.exists():
            existing_workspace = candidate
    if workspace_output.exists():
        existing_workspace = workspace_output
    if repo_workspace.exists():
        existing_workspace = repo_workspace

    if existing_workspace:
        print(f"[INFO] Using existing workspace file: {existing_workspace}")
        WORKSPACE_FILE = str(existing_workspace)
    else:
        generated_workspace = run_voxelinfo(workspace_output)
        if generated_workspace:
            WORKSPACE_FILE = str(generated_workspace)

    # Step 3: Run inference
    kl_file = run_inference()
    if kl_file is None:
        print("Failed to run knowledge inference. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 4: Generate IC explanations from features
    explanations_file = run_explanations(FEATURES_OUTPUT_CSV, EXPLANATIONS_OUTPUT_CSV)
    explanation_map = {}
    if explanations_file and explanations_file.exists():
        try:
            exp_df = pd.read_csv(explanations_file)
            for _, row in exp_df.iterrows():
                explanation_map[int(row["IC"])] = row.get("explanation")
        except Exception as exc:
            print(f"[WARN] Unable to load explanations: {exc}")
            explanation_map = {}

    # Step 5: Load and match predictions
    merged_df = load_and_match_predictions(dl_file, kl_file)
    if merged_df is None:
        print("Failed to load/match predictions. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Step 6: Apply SOZ logic
    soz_results = apply_soz_logic(merged_df, explanation_map=explanation_map)
    
    # Step 5: Final result
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    soz_count = sum(1 for result in soz_results if result['SOZ'])
    total_count = len(soz_results)
    
    print(f"Total ICs analyzed: {total_count}")
    print(f"SOZ count: {soz_count}")
    print(f"NOT SOZ count: {total_count - soz_count}")
    
    # Save detailed results
    results_df = pd.DataFrame(soz_results)
    results_file = current_dir / "pipeline_results.csv"
    results_df.to_csv(results_file, index=False)
    results_for_json = results_df.where(pd.notnull(results_df), None)
    results_for_json.to_json(current_dir / "pipeline_results.json", orient='records')
    print(f"\nDetailed results saved to: {results_file}")
    
    # Final decision
    if soz_count > 0:
        print("\n" + "=" * 60)
        print("FINAL DECISION: SOZ")
        print("=" * 60)
        print(f"Found {soz_count} SOZ components out of {total_count} total components.")
        
        # Show which ICs are SOZ
        soz_ics = [result['IC'] for result in soz_results if result['SOZ']]
        print(f"SOZ ICs: {soz_ics}")
    else:
        print("\n" + "=" * 60)
        print("FINAL DECISION: NOT SOZ")
        print("=" * 60)
        print("No SOZ components found.")

    soz_ics = [int(result['IC']) for result in soz_results if result['SOZ']]
    summary = {
        "total_components": total_count,
        "soz_count": soz_count,
        "patient_is_soz": soz_count > 0,
        "soz_ics": soz_ics
    }
    with open(current_dir / "analysis_summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    try:
        generate_dbscan_assets(soz_results)
    except Exception as exc:  # pragma: no cover - resilience only
        print(f"[WARN] Unexpected error during DBSCAN export: {exc}")

if __name__ == "__main__":
    main()
