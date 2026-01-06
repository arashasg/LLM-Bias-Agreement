import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kendalltau

# --- Configuration ---
LODO_DATASETS = [
    'wino', 'stereoset', 'crows', 'reddit', 
    'bug', 'holistic', 'realToxicityPrompts', 'bold'
]

def filter_and_prepare_data(data):
    """
    Filters models based on strict dataset count requirements.
    Returns a list of valid model objects.
    """
    valid_models = []
    
    for entry in data:
        # Extract sub-dictionaries (default to empty dict if missing)
        cat_data = entry.get('cat', {})
        lmb_data = entry.get('lmb', {})
        honest_data = entry.get('honest', {})
        toxicity_data = entry.get('toxicity', {})
        icat_val = entry.get('icat')

        # Check counts
        # 1. Honest and Toxicity: at least 2 datasets
        if len(honest_data) < 2 or len(toxicity_data) < 2:
            continue
            
        # 2. Cat and Lmb: at least 6 datasets
        if len(cat_data) < 5 or len(lmb_data) < 5:
            continue
            
        # 3. ICAT: must exist (not None)
        if icat_val is None:
            continue
            
        valid_models.append(entry)
        
    print(f"Filtered Models: kept {len(valid_models)} out of {len(data)} inputs.")
    return valid_models

def calculate_metric_rank(model_entry, metric_key, exclude_dataset=None):
    """
    Calculates the mean rank for a specific metric (e.g., 'cat', 'lmb'),
    excluding the specified dataset if present.
    """
    dataset_ranks = model_entry.get(metric_key, {})
    
    # Filter out the excluded dataset and get values
    valid_ranks = [
        rank for ds, rank in dataset_ranks.items() 
        if ds != exclude_dataset
    ]
    
    if not valid_ranks:
        return np.nan
        
    return np.mean(valid_ranks)

def generate_rank_dataframe(models, exclude_dataset=None):
    """
    Generates a DataFrame with final ranks for all metrics, 
    re-calculated based on the 'leave-one-out' logic.
    """
    rows = []
    
    for model in models:
        row = {'model_name': model['model_name']}
        
        # Calculate means for the composite metrics
        row['cat_score'] = calculate_metric_rank(model, 'cat', exclude_dataset)
        row['lmb_score'] = calculate_metric_rank(model, 'lmb', exclude_dataset)
        row['honest_score'] = calculate_metric_rank(model, 'honest', exclude_dataset)
        row['toxicity_score'] = calculate_metric_rank(model, 'toxicity', exclude_dataset)
        
        # ICAT is a single value, just take it directly
        row['icat_rank'] = model['icat']
        
        rows.append(row)
        
    df = pd.DataFrame(rows).set_index('model_name')
    
    # Rank the scores to get integer ranks (1 to N)
    # Lower score (mean rank) is better -> ascending=True
    df['cat_rank'] = df['cat_score'].rank(method='min', ascending=True).astype(int)
    df['lmb_rank'] = df['lmb_score'].rank(method='min', ascending=True).astype(int)
    df['honest_rank'] = df['honest_score'].rank(method='min', ascending=True).astype(int)
    df['toxicity_rank'] = df['toxicity_score'].rank(method='min', ascending=True).astype(int)
    
    # Ensure icat is int
    df['icat_rank'] = df['icat_rank'].astype(int)
    
    return df[['cat_rank', 'lmb_rank', 'honest_rank', 'toxicity_rank', 'icat_rank']]

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return np.tanh(z)

def calculate_agreement_stats(df):
    """
    Calculates Correlation Matrices, MeAS, and MoAS using PEARSON.
    """
    # --- UPDATED: Use Pearson correlation ---
    metric_corr_matrix = df.corr(method='pearson')
    
    df_t = df.transpose()
    # --- UPDATED: Use Pearson correlation ---
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

def plot_and_save_heatmap(corr_matrix, title, filename):
    plt.figure(figsize=(10, 8))
    
    # Capture the axes object 'ax'
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation'})
    
    # --- UPDATED: Set font size for column (x-axis) and row (y-axis) labels ---
    plt.xticks(fontsize=12, rotation=45, ha='right') # Bigger font, rotated for readability
    plt.yticks(fontsize=12)                          # Bigger font for rows
    
    # Optional: Increase title font size as well
    plt.title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_stability_test(file_path):
    # 1. Load Data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # 2. Filter Models
    valid_models = filter_and_prepare_data(data)
    
    if not valid_models:
        print("No models met the strict filtering criteria. Exiting.")
        return

    # Output Directory - Updated to reflect Pearson
    output_dir = "stability_analysis_output_pearson"
    os.makedirs(output_dir, exist_ok=True)
    
    summary_results = []
    
    # 3. Calculate Baseline First (All Datasets Included)
    print("\n--- Calculating Baseline (All Datasets) ---")
    df_baseline = generate_rank_dataframe(valid_models, exclude_dataset=None)
    
    # Run Baseline Agreement Stats
    meas_base, moas_base, metric_corr_base, model_corr_base = calculate_agreement_stats(df_baseline)
    
    # Save Baseline Outputs
    plot_and_save_heatmap(metric_corr_base, "Metric Correlation (Baseline)", 
                          os.path.join(output_dir, "Baseline_metric_corr.png"))
    plot_and_save_heatmap(model_corr_base, "Model Correlation (Baseline)", 
                          os.path.join(output_dir, "Baseline_model_corr.png"))
    
    # Save Baseline MoAS to CSV
    baseline_moas_df = pd.DataFrame(list(moas_base.items()), columns=['model_name', 'moas_score'])
    baseline_moas_df.to_csv(os.path.join(output_dir, "Baseline_moas.csv"), index=False)
    
    # Store Baseline Summary
    # Flatten the stability dictionary so it prints nicely as columns
    baseline_entry = {
        "Condition": "Baseline (None Left Out)",
        "Avg_MoAS": np.mean(list(moas_base.values())) if moas_base else 0
    }
    # Add stability columns (all 1.0 for baseline)
    for col in df_baseline.columns:
        baseline_entry[f"Tau_{col}"] = 1.0
        
    summary_results.append(baseline_entry)

    # 4. LODO Loop
    for dataset in LODO_DATASETS:
        condition_name = f"Left_Out_{dataset}"
        print(f"\n--- Processing: {condition_name} ---")
        
        # A. Generate Ranks (Recalculate means excluding 'dataset')
        df_ranks = generate_rank_dataframe(valid_models, exclude_dataset=dataset)
        
        # B. Calculate Stats (MeAS, MoAS)
        meas, moas, metric_corr, model_corr = calculate_agreement_stats(df_ranks)
        
        # C. Calculate Kendall's Tau Stability (vs Baseline)
        stability_scores = {}
        for col in df_baseline.columns:
            # Calculate Tau between the Baseline column and the LODO column
            tau, _ = kendalltau(df_baseline[col], df_ranks[col])
            stability_scores[col] = tau
            
        # D. Save Heatmaps
        plot_and_save_heatmap(metric_corr, f"Metric Correlation ({condition_name})", 
                              os.path.join(output_dir, f"{condition_name}_metric_corr.png"))
        
        plot_and_save_heatmap(model_corr, f"Model Correlation ({condition_name})", 
                              os.path.join(output_dir, f"{condition_name}_model_corr.png"))
        
        # E. Save MoAS to CSV
        moas_filename = os.path.join(output_dir, f"{condition_name}_moas.csv")
        moas_df = pd.DataFrame(list(moas.items()), columns=['model_name', 'moas_score'])
        moas_df.to_csv(moas_filename, index=False)
        print(f"Saved MoAS data to {moas_filename}")
        
        # F. Store Summary
        lodo_entry = {
            "Condition": condition_name,
            "Avg_MoAS": np.mean(list(moas.values())) if moas else 0
        }
        # Add the calculated Tau scores
        for col, tau in stability_scores.items():
            lodo_entry[f"Tau_{col}"] = tau
            
        summary_results.append(lodo_entry)

    # 5. Print Summary as Table
    print("\n\n=== Stability Analysis Summary Table ===")
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_results)
    
    # Format floating point numbers for better readability
    pd.options.display.float_format = '{:,.4f}'.format
    
    # Print the DataFrame as a string (to ensure it prints fully in console)
    print(summary_df.to_string(index=False))
    
    # Optional: Save summary table to CSV
    summary_csv_path = os.path.join(output_dir, "stability_summary_table.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary table saved to: {summary_csv_path}")

# Example Usage
if __name__ == "__main__":
    run_stability_test('all_model_ranks.json')