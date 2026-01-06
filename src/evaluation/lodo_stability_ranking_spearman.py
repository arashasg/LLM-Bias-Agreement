import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kendalltau, spearmanr, wilcoxon

# --- Configuration ---
# Explicit list of 10 models to keep (Matches your LODO output)
TARGET_MODELS = {
    "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Llama-4-Scout-17B-16E",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Phi-3-mini-4k-instruct",
    "gemma-2-9b-it",
    "phi-2",
    "phi-4"
}

# LODO Datasets list (needed for the loop)
LODO_DATASETS = [
    'wino', 'stereoset', 'crows', 'reddit', 
    'bug', 'holistic', 'realToxicityPrompts', 'bold'
]

def filter_and_prepare_data(data):
    """
    Filters models to strictly match the TARGET_MODELS list.
    """
    valid_models = []
    
    for entry in data:
        model_name = entry.get('model_name')
        
        # 1. PRIMARY FILTER: Check if model is in our target list
        if model_name not in TARGET_MODELS:
            continue

        # 2. Basic Sanity Check: Ensure data exists to avoid errors
        cat_data = entry.get('cat', {})
        lmb_data = entry.get('lmb', {})
        honest_data = entry.get('honest', {})
        toxicity_data = entry.get('toxicity', {})
        icat_val = entry.get('icat')

        # Honest and Toxicity must not be empty
        if not honest_data or not toxicity_data:
            continue
            
        # Cat and Lmb must have some data (at least 3 to be safe for stats)
        if len(cat_data) < 3 or len(lmb_data) < 3:
            continue
            
        if icat_val is None:
            continue
            
        valid_models.append(entry)
        
    print(f"Filtered Models: kept {len(valid_models)} out of {len(data)} inputs.")
    
    # Validation Warning
    if len(valid_models) != len(TARGET_MODELS):
        found_names = {m['model_name'] for m in valid_models}
        missing = TARGET_MODELS - found_names
        print(f"WARNING: Could not find data for these expected models: {missing}")
        
    return valid_models

def calculate_metric_rank(model_entry, metric_key, exclude_dataset=None):
    """
    Calculates the mean rank for a specific metric.
    """
    dataset_ranks = model_entry.get(metric_key, {})
    
    # Check for the specific Stereoset Edge Case
    if exclude_dataset == 'stereoset' and metric_key in ['honest', 'toxicity']:
        # If Stereoset is the ONLY dataset available for this metric
        if list(dataset_ranks.keys()) == ['stereoset']:
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
    """
    Calculates Correlation Matrices and MeAS/MoAS using SPEARMAN.
    """
    # --- UPDATED: Use Spearman correlation ---
    metric_corr_matrix = df.corr(method='spearman')
    
    df_t = df.transpose()
    # --- UPDATED: Use Spearman correlation ---
    model_corr_matrix = df_t.corr(method='spearman')
    
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
    
    # --- UPDATED: Label to 'Spearman Correlation' ---
    ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Spearman Correlation'})
    
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
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

    # 2. Filter Models (Using Whitelist)
    valid_models = filter_and_prepare_data(data)
    
    if not valid_models:
        print("No models met the strict filtering criteria. Exiting.")
        return

    # Output Directory - Spearman
    output_dir = "stability_analysis_output_spearman"
    os.makedirs(output_dir, exist_ok=True)
    
    summary_results = []
    
    # 3. Calculate Baseline First (All Datasets Included)
    print("\n--- Calculating Baseline (All Datasets) ---")
    df_baseline = generate_rank_dataframe(valid_models, exclude_dataset=None)
    _, moas_base, metric_corr_base, model_corr_base = calculate_agreement_stats(df_baseline)
    
    # Save Baseline Outputs
    plot_and_save_heatmap(metric_corr_base, "Metric Correlation (Baseline)", 
                          os.path.join(output_dir, "Baseline_metric_corr.png"))
    plot_and_save_heatmap(model_corr_base, "Model Correlation (Baseline)", 
                          os.path.join(output_dir, "Baseline_model_corr.png"))
    
    # Save Baseline MoAS to CSV
    baseline_moas_df = pd.DataFrame(list(moas_base.items()), columns=['model_name', 'moas_score'])
    baseline_moas_df.to_csv(os.path.join(output_dir, "Baseline_moas.csv"), index=False)
    
    # Store Baseline Summary
    baseline_entry = {
        "Condition": "Baseline (None Left Out)",
        "Avg_MoAS": np.mean(list(moas_base.values())) if moas_base else 0
    }
    for col in df_baseline.columns:
        baseline_entry[f"Tau_{col}"] = 1.0
        
    summary_results.append(baseline_entry)
    
    # Ordered list of baseline scores for stability calc
    baseline_moas_values = [moas_base.get(m, 0) for m in df_baseline.index]

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
        for col, tau in stability_scores.items():
            lodo_entry[f"Tau_{col}"] = tau
            
        summary_results.append(lodo_entry)

    # 5. Print Summary as Table
    print("\n\n=== Stability Analysis Summary Table (Spearman) ===")
    
    summary_df = pd.DataFrame(summary_results)
    pd.options.display.float_format = '{:,.4f}'.format
    print(summary_df.to_string(index=False))
    
    summary_csv_path = os.path.join(output_dir, "stability_summary_table.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSummary table saved to: {summary_csv_path}")

# Example Usage
if __name__ == "__main__":
    run_stability_test('all_model_ranks.json')