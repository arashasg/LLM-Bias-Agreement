import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr

# --- Configuration ---
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

SHORT_NAME_MAPPING = {
    "DeepSeek-R1-Distill-Llama-8B": "DS-R1-Llama",
    "DeepSeek-R1-Distill-Qwen-7B": "DS-R1-Qwen",
    "Llama-4-Scout-17B-16E": "Llama-4-Scout",
    "Meta-Llama-3-8B-Instruct": "Llama-3",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1",
    "Mistral-7B-Instruct-v0.3": "Mistral-v0.3",
    "Phi-3-mini-4k-instruct": "Phi-3-Mini",
    "gemma-2-9b-it": "Gemma-2",
    "phi-2": "Phi-2",
    "phi-4": "Phi-4"
}

def filter_and_prepare_data(data):
    valid_models = []
    for entry in data:
        model_name = entry.get('model_name')
        if model_name not in TARGET_MODELS:
            continue
        cat_data = entry.get('cat', {})
        lmb_data = entry.get('lmb', {})
        honest_data = entry.get('honest', {})
        toxicity_data = entry.get('toxicity', {})
        icat_val = entry.get('icat')

        if not honest_data or not toxicity_data: continue
        if len(cat_data) < 3 or len(lmb_data) < 3: continue
        if icat_val is None: continue
        valid_models.append(entry)
    return valid_models

def calculate_metric_rank(model_entry, metric_key):
    dataset_ranks = model_entry.get(metric_key, {})
    valid_ranks = list(dataset_ranks.values())
    if not valid_ranks: return np.nan
    return np.mean(valid_ranks)

def generate_rank_dataframe(models):
    rows = []
    for model in models:
        row = {'model_name': model['model_name']}
        row['cat_score'] = calculate_metric_rank(model, 'cat')
        row['lmb_score'] = calculate_metric_rank(model, 'lmb')
        row['honest_score'] = calculate_metric_rank(model, 'honest')
        row['toxicity_score'] = calculate_metric_rank(model, 'toxicity')
        row['icat_rank'] = model['icat']
        rows.append(row)
        
    df = pd.DataFrame(rows).set_index('model_name')
    df = df.rename(index=SHORT_NAME_MAPPING)
    
    df['CAT'] = df['cat_score'].rank(method='min', ascending=True).astype(int)
    df['LMB'] = df['lmb_score'].rank(method='min', ascending=True).astype(int)
    df['HONEST'] = df['honest_score'].rank(method='min', ascending=True).astype(int)
    df['Toxicity'] = df['toxicity_score'].rank(method='min', ascending=True).astype(int)
    df['ICAT'] = df['icat_rank'].astype(int)
    
    return df[['CAT', 'LMB', 'HONEST', 'Toxicity', 'ICAT']]

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return np.tanh(z)

def calculate_corr_and_pvalues(df):
    df = df.dropna()
    cols = df.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p_value_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i in cols:
        for j in cols:
            if i == j:
                corr_matrix.loc[i, j] = 1.0
                p_value_matrix.loc[i, j] = 0.0
            else:
                r, p = spearmanr(df[i], df[j])
                corr_matrix.loc[i, j] = r
                p_value_matrix.loc[i, j] = p
    return corr_matrix, p_value_matrix

def calculate_agreement_stats(df):
    metric_corr_matrix, metric_p_matrix = calculate_corr_and_pvalues(df)
    df_t = df.transpose()
    model_corr_matrix, model_p_matrix = calculate_corr_and_pvalues(df_t)
    
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
        
    return meas_results, moas_results, metric_corr_matrix, metric_p_matrix, model_corr_matrix, model_p_matrix

def plot_and_save_heatmap(corr_matrix, p_matrix, filename, label_fontsize=40, label_rotation=0, annot_fontsize=35):
    """
    Plots heatmap with stars for significant correlations (p < 0.05).
    """
    plt.figure(figsize=(12, 10)) 
    
    annot_labels = corr_matrix.copy().astype(str)
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            val = corr_matrix.loc[i, j]
            label = f"{val:.2f}"
            annot_labels.loc[i, j] = label
    
    # --- HERE IS THE FIX ---
    # annot_kws={'size': annot_fontsize} controls the size of numbers INSIDE the cells
    ax = sns.heatmap(corr_matrix, annot=annot_labels, fmt='', cmap='coolwarm_r', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': ''}, 
                annot_kws={'size': annot_fontsize}) # <--- Uses the variable passed in function
    
    ax.tick_params(axis='x', labelsize=label_fontsize, rotation=label_rotation)
    ax.tick_params(axis='y', labelsize=label_fontsize, rotation=0)

    # Colorbar tick size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30) 

    if label_rotation > 0:
        plt.setp(ax.get_xticklabels(), ha="right")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def extract_moas_scores(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    valid_models = filter_and_prepare_data(data)
    if not valid_models: return

    output_dir = "moas_output_spearman"
    os.makedirs(output_dir, exist_ok=True)
    
    df_ranks = generate_rank_dataframe(valid_models)
    meas, moas, metric_corr, metric_p, model_corr, model_p = calculate_agreement_stats(df_ranks)
    
    # --- FUNCTION CALLS WITH UPDATED SIZES ---
    
    # 1. Metric Correlation
    # annot_fontsize=35 will make the numbers inside the cells VERY large
    plot_and_save_heatmap(metric_corr, metric_p,
                          os.path.join(output_dir, "metric_correlation.png"),
                          label_fontsize=25,  
                          label_rotation=0,
                          annot_fontsize=35) # <--- INCREASED THIS for cell numbers
                          
    # 2. Model Correlation 
    # annot_fontsize=20 (Smaller because there are more cells, 35 might overlap)
    plot_and_save_heatmap(model_corr, model_p,
                          os.path.join(output_dir, "model_correlation.png"),
                          label_fontsize=25,  
                          label_rotation=45,
                          annot_fontsize=20) # <--- INCREASED THIS slightly
    
    # Save CSVs
    df_moas = pd.DataFrame(list(moas.items()), columns=['Model Name', 'MoAS Score']).sort_values(by='MoAS Score', ascending=False)
    df_moas.to_csv(os.path.join(output_dir, "model_agreement_scores.csv"), index=False)
    
    df_meas = pd.DataFrame(list(meas.items()), columns=['Metric Name', 'MeAS Score']).sort_values(by='MeAS Score', ascending=False)
    df_meas.to_csv(os.path.join(output_dir, "metric_agreement_scores.csv"), index=False)

    print("Done. Heatmap cell numbers are now larger.")

if __name__ == "__main__":
    extract_moas_scores('all_model_ranks.json')