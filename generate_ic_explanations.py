"""
Generate IC Explanations based on Top 2 Contributing Features

Usage:
    python generate_ic_explanations.py <input_csv> [output_csv]

Example:
    python generate_ic_explanations.py ic_trainingdata_from_mat.csv final/ic_explanations.csv
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib

# Feature descriptions for explanations
FEATURE_DESC = {
    'white_overlap': "Activation extended from grey matter towards ventricles",
    'big_clusters': "One significantly large cluster",
    'GiniS': "Sparsity in frequency domain of BOLD time courses",
    'GiniI': "Sparsity in activelet domain of BOLD time courses"
}

FEATURE_NAMES = ['white_overlap', 'big_clusters', 'GiniS', 'GiniI']


def load_model(model_path="final/trained_model.joblib"):
    """Load the trained SVM model."""
    model_data = joblib.load(model_path)
    return model_data['model']


def generate_explanation(pct_contributions, predicted_class):
    """
    Generate explanation based on top 2 contributing features.
    
    Parameters:
        pct_contributions: Array of 4 percentage values [white_overlap, big_clusters, GiniS, GiniI]
        predicted_class: 'SOZ' or 'Non-SOZ'
    
    Returns:
        explanation: String explanation
        top_features: List of top 2 feature names
        top_pcts: List of top 2 percentages
    """
    # Get indices of top 2 features by percentage
    top_indices = np.argsort(pct_contributions)[::-1][:2]
    
    top_features = [FEATURE_NAMES[i] for i in top_indices]
    top_pcts = [pct_contributions[i] for i in top_indices]
    
    # Get descriptions (lowercase for natural flow)
    desc1 = FEATURE_DESC[top_features[0]].lower()
    desc2 = FEATURE_DESC[top_features[1]].lower()
    
    # Build explanation
    explanation = f'{predicted_class} because "{desc1} and {desc2}"'
    
    return explanation, top_features, top_pcts


def generate_explanations(input_csv, output_csv=None, model_path="final/trained_model.joblib"):
    """
    Generate explanations for all ICs in input CSV.
    
    Parameters:
        input_csv: Path to CSV with columns: IC, WhiteOverlapCnt, BigClusterCnt, GiniS, GiniI
        output_csv: Path to save results (default: input_name_explanations.csv)
        model_path: Path to trained model
    
    Returns:
        DataFrame with IC explanations
    """
    print("="*70)
    print("GENERATE IC EXPLANATIONS")
    print("="*70)
    
    # Set default output path
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = f"final/{base_name}_explanations.csv"
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = load_model(model_path)
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    print(f"\nModel Coefficients:")
    for name, c in zip(FEATURE_NAMES, coef):
        direction = "→ SOZ" if c > 0 else "→ Non-SOZ"
        print(f"  {name:15s}: {c:+.6f} {direction}")
    print(f"  {'Intercept':15s}: {intercept:+.6f}")
    
    # Load data
    print(f"\nLoading data: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Total ICs: {len(df)}")
    
    # Map column names
    column_mapping = {
        'WhiteOverlapCnt': 'white_overlap',
        'BigClusterCnt': 'big_clusters'
    }
    df_renamed = df.rename(columns=column_mapping)
    
    # Extract features
    try:
        X = df_renamed[['white_overlap', 'big_clusters', 'GiniS', 'GiniI']].values
    except KeyError:
        X = df[['WhiteOverlapCnt', 'BigClusterCnt', 'GiniS', 'GiniI']].values
    
    # Get IC IDs
    if 'IC' in df.columns:
        ic_ids = df['IC'].values
    elif 'ic_id' in df.columns:
        ic_ids = df['ic_id'].values
    else:
        ic_ids = np.arange(1, len(df) + 1)
    
    # Predictions
    predictions = model.predict(X)
    decision_values = model.decision_function(X)
    
    # Calculate contributions
    contributions = X * coef
    abs_contributions = np.abs(contributions)
    total_abs = abs_contributions.sum(axis=1, keepdims=True)
    total_abs[total_abs == 0] = 1  # Avoid division by zero
    pct_contributions = (abs_contributions / total_abs) * 100
    
    # Build results
    results = []
    
    for i in range(len(df)):
        ic = int(ic_ids[i])
        pred_class = 'SOZ' if predictions[i] == 3 else 'Non-SOZ'
        pcts = pct_contributions[i]
        
        # Generate explanation
        explanation, top_features, top_pcts = generate_explanation(pcts, pred_class)
        
        results.append({
            'IC': ic,
            'predicted_class': pred_class,
            'decision_value': round(decision_values[i], 4),
            'white_overlap_pct': round(pcts[0], 2),
            'big_clusters_pct': round(pcts[1], 2),
            'GiniS_pct': round(pcts[2], 2),
            'GiniI_pct': round(pcts[3], 2),
            'top_feature_1': top_features[0],
            'top_feature_1_pct': round(top_pcts[0], 2),
            'top_feature_2': top_features[1],
            'top_feature_2_pct': round(top_pcts[1], 2),
            'explanation': explanation
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    pred_counts = results_df['predicted_class'].value_counts()
    print(f"\nPrediction Distribution:")
    for cls, count in pred_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {cls}: {count} ICs ({pct:.1f}%)")
    
    print(f"\nAverage Percentage Contribution:")
    for name in FEATURE_NAMES:
        col = f"{name}_pct"
        print(f"  {name:15s}: {results_df[col].mean():6.2f}%")
    
    # Print all explanations
    print("\n" + "="*70)
    print("IC EXPLANATIONS")
    print("="*70)
    
    for i in range(len(results_df)):
        row = results_df.iloc[i]
        print(f"\nIC {row['IC']:3d}: {row['explanation']}")
        print(f"        Top: {row['top_feature_1']} ({row['top_feature_1_pct']}%), "
              f"{row['top_feature_2']} ({row['top_feature_2_pct']}%)")
    
    print("\n" + "="*70)
    print(f"✓ Saved to: {output_csv} ({len(results_df)} ICs)")
    print("="*70)
    
    return results_df


# Main execution
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate_ic_explanations.py <input_csv> [output_csv]")
        print("Example: python generate_ic_explanations.py ic_trainingdata_from_mat.csv final/ic_explanations.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_csv):
        print(f"Error: File not found: {input_csv}")
        sys.exit(1)
    
    results = generate_explanations(input_csv, output_csv)
