import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_agreement(file_path):
    # 1. Read Data from JSON
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None

    df = pd.DataFrame(data).set_index('model_name')

    # 2. Filter: Keep only models with NO missing values in the required metrics
    required_metrics = ['cat_rank', 'lmb_rank', 'honest_rank', 'toxicity_rank', 'icat_rank']
    
    # Check if columns exist
    available_metrics = [m for m in required_metrics if m in df.columns]
    
    if not available_metrics:
        print("Error: No valid metrics found in the data.")
        return None, None, None, None

    df_filtered = df.dropna(subset=available_metrics).copy()
    
    if df_filtered.empty:
        print("Error: No models remaining after filtering for missing values.")
        return None, None, None, None

    # 3. Re-Rank: Update ranks to be sequential (1 to N)
    for col in available_metrics:
        df_filtered[col] = df_filtered[col].rank(method='min').astype(int)

    # 4. Helper Functions for Agreement Calculation
    def fisher_z(r):
        r = np.clip(r, -0.999999, 0.999999)
        return 0.5 * np.log((1 + r) / (1 - r))

    def inverse_fisher_z(z):
        return np.tanh(z)

    # 5. Calculate Correlation Matrices
    # Metric Correlation Matrix (pairwise correlations between metrics)
    metric_corr_matrix = df_filtered.corr(method='pearson')

    # Model Correlation Matrix (pairwise correlations between models)
    df_t = df_filtered.transpose()
    model_corr_matrix = df_t.corr(method='pearson')

    # 6. Calculate MeAS (Metric Agreement Score)
    meas_results = {}
    for m_target in available_metrics:
        z_scores = []
        for m_other in available_metrics:
            if m_target == m_other: continue
            
            # Use the pre-calculated correlation matrix
            if m_target in metric_corr_matrix.columns and m_other in metric_corr_matrix.index:
                corr = metric_corr_matrix.loc[m_target, m_other]
                z_scores.append(fisher_z(corr))
                
        avg_z = np.mean(z_scores) if z_scores else 0
        meas_results[m_target] = inverse_fisher_z(avg_z)

    # 7. Calculate MoAS (Model Agreement Score)
    moas_results = {}
    for model_target in df_t.columns:
        z_scores = []
        for model_other in df_t.columns:
            if model_target == model_other: continue
            
            # Use the pre-calculated correlation matrix
            if model_target in model_corr_matrix.columns and model_other in model_corr_matrix.index:
                corr = model_corr_matrix.loc[model_target, model_other]
                if not np.isnan(corr):
                    z_scores.append(fisher_z(corr))
                
        avg_z = np.mean(z_scores) if z_scores else 0
        moas_results[model_target] = inverse_fisher_z(avg_z)

    return meas_results, moas_results, metric_corr_matrix, model_corr_matrix

def plot_and_save_heatmap(corr_matrix, title, filename):
    """
    Generates and saves a heatmap from a correlation matrix.
    """
    plt.figure(figsize=(10, 8))
    # Create heatmap with annotations, using a diverging colormap (blue to red) similar to the paper
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Pearson Correlation'})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved heatmap to {filename}")

# --- Example Usage ---

if __name__ == "__main__":
    # Ensure you have your 'averaged_model_rankings.json' file ready
    meas, moas, metric_corr, model_corr = calculate_agreement('averaged_model_rankings.json')

    if meas:
        print("MeAS Results:", meas)
        print("\nMoAS Results:", moas)
        
        # Save the matrices as images
        plot_and_save_heatmap(metric_corr, "Pearson Correlation Matrix between Metrics", "metric_correlation_matrix.png")
        plot_and_save_heatmap(model_corr, "Pearson Correlation Matrix between Models", "model_correlation_matrix.png")