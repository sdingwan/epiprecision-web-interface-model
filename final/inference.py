"""
Inference Script for Expert Knowledge Integrator
Loads a trained model and makes predictions on new data.

Environment variables (optional):
  KNOWLEDGE_SUBJECT_DIR  - Root folder for the subject (expects MO/report under it)
  WORKSPACE_ROOT_DIR     - Directory that contains Workspace-<subject>V4.mat (defaults to subject dir)
  WORKSPACE_FILE         - Explicit path to workspace .mat file
  TRAINED_MODEL_PATH     - Path to trained_model.joblib (defaults to final/trained_model.joblib)
  KNOWLEDGE_OUTPUT_CSV   - Output CSV path (defaults to final/predictions_ASUAI_001.csv)

Requirements:
    pip install numpy scipy pandas scikit-learn imbalanced-learn pywavelets joblib
"""

import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pywt
from scipy.io import loadmat
from sklearn.linear_model import Lasso

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SUBJECT_DIR = BASE_DIR / "ASUAI_001"
DEFAULT_MODEL_PATH = BASE_DIR / "trained_model.joblib"
DEFAULT_OUTPUT_CSV = BASE_DIR / "predictions_ASUAI_001.csv"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def mat_cell_to_list(mat_cell):
    """Convert a MATLAB-style cell array loaded by scipy into a Python list if possible."""
    if isinstance(mat_cell, np.ndarray) and mat_cell.dtype == "object":
        return [mat_cell.flatten()[i] for i in range(mat_cell.size)]
    return mat_cell


def gini_index(x):
    """Compute Gini index similar to MATLAB GiniIndex used in code."""
    arr = np.abs(np.asarray(x).flatten()).astype(float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(arr)
    if cumulative[-1] == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def read_dlm_txt(path):
    """Read numeric text file (like MATLAB dlmread)."""
    return np.loadtxt(str(Path(path)))


def resolve_workspace_path(subject_folder, workspace_root=None, workspace_override=None):
    """
    Determine the location of the workspace .mat file based on overrides or conventions.
    """
    subject_folder = Path(subject_folder)

    if workspace_override:
        candidate = Path(workspace_override)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Workspace override not found: {candidate}")

    folder_name = subject_folder.name
    workspace_filename = f"Workspace-{folder_name}V4.mat"
    candidates = [
        subject_folder / workspace_filename,
    ]

    if workspace_root:
        candidates.append(Path(workspace_root) / workspace_filename)

    candidates.append(BASE_DIR / workspace_filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Workspace .mat not found. Tried: {', '.join(str(c) for c in candidates)}"
    )


# ---------------------------------------------------------------------------
# Feature extraction for inference
# ---------------------------------------------------------------------------

def extract_features_for_subject(
    subject_folder,
    pixel_limit_default=50,
    swt_length=256,
    sine_bases=10,
    workspace_root=None,
    workspace_override=None,
):
    """
    Extract features for a single subject folder for inference.
    Same as training but without noise considerations.
    """
    subject_folder = Path(subject_folder)
    report_candidates = [
        subject_folder / "MO" / "report",
        subject_folder / "report",
    ]
    if subject_folder.name.lower() == "report":
        report_candidates.append(subject_folder)

    report_path = next((p for p in report_candidates if p.exists()), None)
    if report_path is None:
        for candidate in subject_folder.rglob("*"):
            if candidate.is_dir() and candidate.name.lower() == "report":
                report_path = candidate
                break

    if report_path is None or not report_path.exists():
        raise FileNotFoundError(
            f"Unable to locate report directory under {subject_folder}. "
            "Ensure the upload contains the fMRI report folder (…/MO/report or …/report)."
        )

    thresh_files = sorted(report_path.glob("*_thresh*"))
    txt_files = sorted(report_path.glob("t*.txt"))

    n_files = len(txt_files)
    if n_files == 0:
        raise FileNotFoundError(f"No 't*.txt' files found in {report_path}")

    workspace_path = resolve_workspace_path(
        subject_folder, workspace_root, workspace_override
    )

    ws = loadmat(str(workspace_path), squeeze_me=False, struct_as_record=False)
    pixel_limit = ws.get("pixelLimit", pixel_limit_default)
    try:
        pixel_limit = int(np.squeeze(pixel_limit))
    except Exception:
        pixel_limit = pixel_limit_default

    if "ClusterSizePat" not in ws or "OverallWhiteFile" not in ws:
        raise KeyError(
            f"'ClusterSizePat' and/or 'OverallWhiteFile' missing in {workspace_path}"
        )
    ClusterSizePat_raw = mat_cell_to_list(ws["ClusterSizePat"])
    OverallWhiteFile_raw = mat_cell_to_list(ws["OverallWhiteFile"])

    # Prepare outputs
    GiniI = np.zeros(n_files, dtype=float)
    GiniS = np.zeros(n_files, dtype=float)
    num_clusters = np.zeros(n_files, dtype=int)
    white_overlap_cnt = np.zeros(n_files, dtype=int)
    big_cluster_cnt = np.zeros(n_files, dtype=int)
    filename_numbers = np.zeros(n_files, dtype=int)

    # iterate through files
    for i, txt_path in enumerate(txt_files):
        # read ICA time-course
        ica = read_dlm_txt(txt_path)
        if ica.ndim == 1:
            ica = ica.reshape(-1, 1)

        samples = ica.shape[0]

        # ---- GiniI via SWT + Lasso (sliding windows) ----
        max_gini = 0.0
        step = swt_length
        for start in range(0, samples - swt_length + 1, step):
            segment = ica[start : start + swt_length, 0]
            try:
                coeffs = pywt.swt(segment, "bior2.2", level=3)
            except Exception:
                continue
            detail_cols = [cD for _, cD in coeffs]
            design = np.vstack(detail_cols).T
            try:
                lasso = Lasso(alpha=0.5, max_iter=1000)
                lasso.fit(design, segment)
                coefs = lasso.coef_
            except Exception:
                coefs = np.zeros(design.shape[1])
            bn = gini_index(coefs)
            if bn > max_gini:
                max_gini = bn
        GiniI[i] = max_gini

        # ---- GiniS via sine-domain sparse representation ----
        t = np.arange(samples)
        sin_dict = np.zeros((samples, sine_bases))
        for k in range(1, sine_bases + 1):
            sin_dict[:, k - 1] = np.sin(2 * np.pi * k * t / samples)
        try:
            lasso_sine = Lasso(alpha=0.01, max_iter=1000)
            lasso_sine.fit(sin_dict, ica[:, 0])
            y2 = lasso_sine.coef_
        except Exception:
            y2 = np.zeros(sine_bases)
        GiniS[i] = gini_index(y2)

        # ---- Cluster counts and white overlap ----
        try:
            cluster_entry = ClusterSizePat_raw[i]
            white_entry = OverallWhiteFile_raw[i]
        except Exception:
            cluster_entry = np.ravel(ClusterSizePat_raw)[i]
            white_entry = np.ravel(OverallWhiteFile_raw)[i]

        try:
            sizes = np.array(cluster_entry).astype(int).flatten()
        except Exception:
            sizes = np.zeros(0, dtype=int)
        if sizes.size == 0:
            big_cluster_cnt[i] = 0
            white_overlap_cnt[i] = 0
        else:
            big_mask = sizes >= pixel_limit
            big_cluster_cnt[i] = int(np.sum(big_mask))
            try:
                white_array = np.array(white_entry).flatten()
                if white_array.size >= big_mask.size:
                    white_overlap_cnt[i] = int(np.sum(white_array[big_mask] == 1))
                else:
                    white_overlap_cnt[i] = int(np.sum(white_array == 1))
            except Exception:
                white_overlap_cnt[i] = 0

        num_clusters[i] = int(np.sum(sizes > pixel_limit))

        # parse filename number
        if i < len(thresh_files):
            fname = thresh_files[i].name
            m = re.search(r"IC[_\-]?(\d+)", fname)
            filename_numbers[i] = int(m.group(1)) if m else i + 1
        else:
            filename_numbers[i] = i + 1

    # Sort arrays by filename_numbers
    sorted_idx = np.argsort(filename_numbers)
    out = {
        "GiniI": GiniI[sorted_idx],
        "GiniS": GiniS[sorted_idx],
        "num_clusters": num_clusters[sorted_idx],
        "white_overlap": white_overlap_cnt[sorted_idx],
        "big_clusters": big_cluster_cnt[sorted_idx],
        "filename_numbers": filename_numbers[sorted_idx],
    }
    return out


# ---------------------------------------------------------------------------
# Model loading and prediction
# ---------------------------------------------------------------------------

def load_trained_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load the trained model and metadata.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = joblib.load(model_path)
    model = model_data["model"]
    metadata = model_data["metadata"]

    print(f"Model loaded from: {model_path}")
    print(f"Model type: {metadata['model_type']}")
    print(f"Training date: {metadata['training_date']}")
    print(f"Feature names: {metadata['feature_names']}")
    print(f"Number of features: {metadata['n_features']}")

    return model, metadata


def predict_subject(
    subject_folder,
    model_path=DEFAULT_MODEL_PATH,
    workspace_root=None,
    workspace_override=None,
    pixel_limit_default=50,
):
    """
    Make predictions for a single subject using the trained model.
    """
    subject_folder = Path(subject_folder)
    if not subject_folder.exists():
        raise FileNotFoundError(f"Subject folder not found: {subject_folder}")

    # Load model
    model, metadata = load_trained_model(model_path)

    pixel_limit = metadata.get("pixel_limit", pixel_limit_default)

    # Extract features
    print(f"Extracting features for subject: {subject_folder.name}")
    features = extract_features_for_subject(
        subject_folder,
        pixel_limit_default=pixel_limit,
        workspace_root=workspace_root,
        workspace_override=workspace_override,
    )

    # Prepare feature matrix
    X = np.column_stack(
        [
            features["white_overlap"],
            features["big_clusters"],
            features["GiniS"],
            features["GiniI"],
        ]
    )

    # Make predictions
    predictions = model.predict(X)
    probabilities = None
    try:
        probabilities = model.predict_proba(X)
    except Exception:
        probabilities = None

    # Create results dataframe
    results = pd.DataFrame(
        {
            "IC_number": features["filename_numbers"],
            "prediction": predictions,
        }
    )

    if probabilities is not None and hasattr(model, "classes_"):
        for idx, label in enumerate(model.classes_):
            if idx < probabilities.shape[1]:
                results[f"prob_class_{label}"] = probabilities[:, idx]

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """
    Run inference using environment configuration.
    """
    print("=" * 60)
    print("EXPERT KNOWLEDGE INTEGRATOR - INFERENCE")
    print("=" * 60)

    subject_folder = Path(
        os.environ.get("KNOWLEDGE_SUBJECT_DIR", DEFAULT_SUBJECT_DIR)
    )
    model_path = Path(os.environ.get("TRAINED_MODEL_PATH", DEFAULT_MODEL_PATH))
    workspace_root = os.environ.get("WORKSPACE_ROOT_DIR")
    workspace_override = os.environ.get("WORKSPACE_FILE")
    output_csv = Path(os.environ.get("KNOWLEDGE_OUTPUT_CSV", DEFAULT_OUTPUT_CSV))

    try:
        results = predict_subject(
            subject_folder,
            model_path=model_path,
            workspace_root=workspace_root,
            workspace_override=workspace_override,
        )

        output_path = output_csv if output_csv.is_absolute() else (BASE_DIR / output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

        print(f"\nPredictions for {subject_folder.name}:")
        print(results.head())
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
