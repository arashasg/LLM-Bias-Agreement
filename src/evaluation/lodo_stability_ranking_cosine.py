import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity

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

def calculate_cosine_matrix(df):
    """
    Calculates the Cosine Similarity matrix for the columns of the input DataFrame.
    Note: sklearn calculates similarity between rows, so we transpose first
    to get column-to-column similarity.
    """
    # 1. Compute Cosine Similarity (returns numpy array)
    # Transpose df so columns (metrics) become rows for the calculation
    similarity_matrix = cosine_similarity(df.T)
    
    # 2. Convert back to DataFrame with correct labels
    sim_df = pd.DataFrame(
        similarity_matrix,
        index=df.columns,
        columns=df.columns
    )
    return sim_df

def calculate_agreement_stats(df):
    """
    Calculates Correlation Matrices, MeAS, and MoAS using COSINE SIMILARITY.
    """
    # --- UPDATED: Use Cosine Similarity ---
    # Metric Agreement: Correlation between columns (Metrics)
    metric_corr_matrix = calculate_cosine_matrix(df)
    
    # Model Agreement: Correlation between rows (Models)
    # Transpose df so models become columns, then apply same function
    df_t = df.transpose()
    model_corr_matrix = calculate_cosine_matrix(df_t)
    
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
    # --- UPDATED: Label to 'Cosine Similarity' ---
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title(title)
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

    # Output Directory - Updated to reflect Cosine
    output_dir = "stability_analysis_output_cosine"
    os.makedirs(output_dir, exist_ok=True)
    
    summary_results = []
    
    # 3. Calculate Baseline First (All Datasets Included)
    print("\n--- Calculating Baseline (All Datasets) ---")
    df_baseline = generate_rank_dataframe(valid_models, exclude_dataset=None)
    
    # Run Baseline Agreement Stats
    meas_base, moas_base, metric_corr_base, model_corr_base = calculate_agreement_stats(df_baseline)
    
    # Save Baseline Outputs
    plot_and_save_heatmap(metric_corr_base, "Metric Similarity (Baseline)", 
                          os.path.join(output_dir, "Baseline_metric_sim.png"))
    plot_and_save_heatmap(model_corr_base, "Model Similarity (Baseline)", 
                          os.path.join(output_dir, "Baseline_model_sim.png"))
    
    # Save Baseline MoAS to CSV
    baseline_moas_df = pd.DataFrame(list(moas_base.items()), columns=['model_name', 'moas_score'])
    baseline_moas_df.to_csv(os.path.join(output_dir, "Baseline_moas.csv"), index=False)
    
    # Store Baseline Summary
    summary_results.append({
        "Condition": "Baseline (None Left Out)",
        "Avg_MoAS": np.mean(list(moas_base.values())) if moas_base else 0,
        # Flatten the tau dictionary into individual columns for the summary
        **{f"Tau_{col}": 1.0 for col in df_baseline.columns}
    })

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
            stability_scores[f"Tau_{col}"] = tau
            
        # D. Save Heatmaps
        plot_and_save_heatmap(metric_corr, f"Metric Similarity ({condition_name})", 
                              os.path.join(output_dir, f"{condition_name}_metric_sim.png"))
        
        plot_and_save_heatmap(model_corr, f"Model Similarity ({condition_name})", 
                              os.path.join(output_dir, f"{condition_name}_model_sim.png"))
        
        # E. Save MoAS to CSV
        moas_filename = os.path.join(output_dir, f"{condition_name}_moas.csv")
        moas_df = pd.DataFrame(list(moas.items()), columns=['model_name', 'moas_score'])
        moas_df.to_csv(moas_filename, index=False)
        print(f"Saved MoAS data to {moas_filename}")
        
        # F. Store Summary
        summary_results.append({
            "Condition": condition_name,
            "Avg_MoAS": np.mean(list(moas.values())) if moas else 0,
            **stability_scores
        })

    # 5. Print Summary as Table
    print("\n\n=== Stability Analysis Summary Table ===")
    
    # Create DataFrame from the list of dictionaries
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