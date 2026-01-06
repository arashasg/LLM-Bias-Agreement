import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr, wilcoxon

# --- Configuration ---
LODO_DATASETS = [
    'wino', 'stereoset', 'crows', 'reddit', 
    'bug', 'holistic', 'realToxicityPrompts', 'bold'
]

def filter_and_prepare_data(data):
    """
    Filters models based on strict dataset count requirements.
    Updated Logic:
    1. Honest and Toxicity MUST exist (not empty).
    2. Cat and Lmb must have at least 3 datasets.
    """
    valid_models = []
    
    for entry in data:
        cat_data = entry.get('cat', {})
        lmb_data = entry.get('lmb', {})
        honest_data = entry.get('honest', {})
        toxicity_data = entry.get('toxicity', {})
        icat_val = entry.get('icat')

        # 1. Honest and Toxicity: Must not be empty
        # If they are not calculated based on ANY datasets, filter out.
        if not honest_data or not toxicity_data:
            continue
            
        # 2. Cat and Lmb: at least 3 datasets (Probabilistic metrics)
        if len(cat_data) < 3 or len(lmb_data) < 3:
            continue
            
        # 3. ICAT: must exist (not None)
        if icat_val is None:
            continue
            
        valid_models.append(entry)
        
    print(f"Filtered Models: kept {len(valid_models)} out of {len(data)} inputs.")
    return valid_models

def calculate_metric_rank(model_entry, metric_key, exclude_dataset=None):
    """
    Calculates the mean rank for a specific metric.
    
    Logic Update for Stereoset LODO:
    If 'toxicity' or 'honest' are ONLY calculated based on 'stereoset',
    we DO NOT exclude stereoset for them (to preserve the score).
    We ONLY exclude stereoset for 'cat' and 'lmb' in that specific edge case.
    """
    dataset_ranks = model_entry.get(metric_key, {})
    
    # Check for the specific Stereoset Edge Case
    if exclude_dataset == 'stereoset' and metric_key in ['honest', 'toxicity']:
        # If Stereoset is the ONLY dataset available for this metric
        if list(dataset_ranks.keys()) == ['stereoset']:
            # Do NOT exclude it. Return the mean (which is just the stereoset score)
            return np.mean(list(dataset_ranks.values()))

    # Normal LODO process (Filter out the excluded dataset)
    valid_ranks = [rank for ds, rank in dataset_ranks.items() if ds != exclude_dataset]
    
    if not valid_ranks:
        return np.nan
        
    return np.mean(valid_ranks)

def generate_rank_dataframe(models, exclude_dataset=None):
    rows = []
    for model in models:
        row = {'model_name': model['model_name']}
        
        row['cat_score'] = calculate_metric_rank(model, 'cat', exclude_dataset)
        row['lmb_score'] = calculate_metric_rank(model, 'lmb', exclude_dataset)
        row['honest_score'] = calculate_metric_rank(model, 'honest', exclude_dataset)
        row['toxicity_score'] = calculate_metric_rank(model, 'toxicity', exclude_dataset)
        row['icat_rank'] = model['icat']
        
        rows.append(row)
        
    df = pd.DataFrame(rows).set_index('model_name')
    
    # Generate Ranks
    # Lower score is better -> ascending=True
    df['cat_rank'] = df['cat_score'].rank(method='min', ascending=True).astype(int)
    df['lmb_rank'] = df['lmb_score'].rank(method='min', ascending=True).astype(int)
    df['honest_rank'] = df['honest_score'].rank(method='min', ascending=True).astype(int)
    df['toxicity_rank'] = df['toxicity_score'].rank(method='min', ascending=True).astype(int)
    df['icat_rank'] = df['icat_rank'].astype(int)
    
    return df[['cat_rank', 'lmb_rank', 'honest_rank', 'toxicity_rank', 'icat_rank']]

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return np.tanh(z)

def calculate_agreement_stats(df):
    metric_corr_matrix = df.corr(method='pearson')
    df_t = df.transpose()
    model_corr_matrix = df_t.corr(method='pearson')
    
    # MeAS
    meas_results = {}
    for m_target in df.columns:
        z_scores = []
        for m_other in df.columns:
            if m_target == m_other: continue
            if m_target in metric_corr_matrix.columns:
                corr = metric_corr_matrix.loc[m_target, m_other]
                z_scores.append(fisher_z(corr))
        avg_z = np.mean(z_scores) if z_scores else 0
        meas_results[m_target] = inverse_fisher_z(avg_z)
        
    # MoAS
    moas_results = {}
    for m_target in df_t.columns:
        z_scores = []
        for m_other in df_t.columns:
            if m_target == m_other: continue
            if m_target in model_corr_matrix.columns:
                corr = model_corr_matrix.loc[m_target, m_other]
                if not np.isnan(corr):
                    z_scores.append(fisher_z(corr))
        avg_z = np.mean(z_scores) if z_scores else 0
        moas_results[m_target] = inverse_fisher_z(avg_z)
        
    return meas_results, moas_results, metric_corr_matrix, model_corr_matrix

def run_stability_test(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    valid_models = filter_and_prepare_data(data)
    if not valid_models:
        print("No models met criteria. Exiting.")
        return

    output_dir = "stability_analysis_pearson"
    os.makedirs(output_dir, exist_ok=True)
    summary_results = []
    
    # ---------------------------------------------------------
    # 1. Baseline Calculation
    # ---------------------------------------------------------
    print("\n--- Calculating Baseline (All Datasets) ---")
    df_baseline = generate_rank_dataframe(valid_models, exclude_dataset=None)
    _, moas_base, _, _ = calculate_agreement_stats(df_baseline)
    
    # Ordered list of baseline scores
    baseline_moas_values = [moas_base.get(m, 0) for m in df_baseline.index]

    summary_results.append({
        "condition": "Baseline",
        "avg_moas": np.mean(baseline_moas_values),
        "pearson_stability": 1.0,
        "mad_stability": 0.0,
        "p_value": np.nan 
    })

    # ---------------------------------------------------------
    # 2. LODO Loop
    # ---------------------------------------------------------
    for dataset in LODO_DATASETS:
        condition_name = f"Left_Out_{dataset}"
        print(f"\n--- Processing: {condition_name} ---")
        
        # A. Recalculate Ranks & Stats
        df_lodo = generate_rank_dataframe(valid_models, exclude_dataset=dataset)
        _, moas_lodo, _, _ = calculate_agreement_stats(df_lodo)
        
        # B. Prepare Vectors (Matched by Model Name)
        lodo_moas_values = [moas_lodo.get(m, 0) for m in df_baseline.index]
        
        # C. Calculate Metrics
        pearson_corr, _ = pearsonr(baseline_moas_values, lodo_moas_values)
        diffs = np.abs(np.array(baseline_moas_values) - np.array(lodo_moas_values))
        mad = np.mean(diffs)
        
        # D. Wilcoxon Signed-Rank Test (Significance)
        try:
            stat, p_val = wilcoxon(baseline_moas_values, lodo_moas_values)
        except ValueError:
            p_val = 1.0 
        
        summary_results.append({
            "condition": condition_name,
            "avg_moas": np.mean(lodo_moas_values),
            "pearson_stability": pearson_corr,
            "mad_stability": mad,
            "p_value": p_val
        })

    # ---------------------------------------------------------
    # 3. Save to CSV
    # ---------------------------------------------------------
    df_summary = pd.DataFrame(summary_results)
    csv_path = os.path.join(output_dir, "moas_stability_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSuccessfully saved summary CSV to: {csv_path}")

    # ---------------------------------------------------------
    # 4. Print Summary as Table
    # ---------------------------------------------------------
    print("\n\n=== MoAS Stability Analysis Summary ===")
    
    # Helper function to format P-values
    def format_p_value(val):
        if pd.isna(val):
            return "N/A"
        sig_char = "*" if val < 0.05 else ""
        return f"{val:.4f}{sig_char}"

    # Prepare DataFrame for Display
    display_df = df_summary.copy()
    display_df = display_df.rename(columns={
        "condition": "Condition",
        "avg_moas": "Avg MoAS",
        "pearson_stability": "Pearson",
        "mad_stability": "MAD",
        "p_value": "P-Value"
    })
    
    # Format Floats
    display_df["Avg MoAS"] = display_df["Avg MoAS"].apply(lambda x: f"{x:.4f}")
    display_df["Pearson"] = display_df["Pearson"].apply(lambda x: f"{x:.4f}")
    display_df["MAD"] = display_df["MAD"].apply(lambda x: f"{x:.4f}")
    display_df["P-Value"] = df_summary["p_value"].apply(format_p_value)

    # Print nicely aligned table
    print(display_df.to_string(index=False))

    print("\nInterpretation of P-Value (Wilcoxon Signed-Rank Test):")
    print(" - p < 0.05 (*): The change in MoAS is statistically significant.")
    print(" - p >= 0.05   : The change is likely due to random noise.")

if __name__ == "__main__":
    run_stability_test('all_model_ranks.json')