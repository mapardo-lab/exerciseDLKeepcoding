
def plot_score_contour(study, param1_name, param2_name):
    """
    Create contour plot based on optimization scores
    """
    # Get completed trials
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df.state == 'COMPLETE']
    
    # Extract parameters and scores
    param1_col = f'params_{param1_name}'
    param2_col = f'params_{param2_name}'
    
    if param1_col not in trials_df.columns or param2_col not in trials_df.columns:
        print(f"Parameters {param1_name} and/or {param2_name} not found in study")
        return
    
    sns.set_style("white")
    plt.figure(figsize=(12, 8))
    
    # Create score-based contour plot
    contour = sns.kdeplot(
        x=trials_df[param1_col],
        y=trials_df[param2_col],
        weights=trials_df['value'],  # Use objective values as weights
        cmap="RdYlBu_r",
        fill=True,
        bw_adjust=0.7,
        levels=20,
        alpha=0.7
    )
    
    # Add points colored by score
    scatter = plt.scatter(
        trials_df[param1_col],
        trials_df[param2_col],
        c=trials_df['value'],
        cmap='RdYlBu_r',
        s=50,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.9
    )
    
    # Mark best trial
    best_trial = study.best_trial
    plt.scatter(
        best_trial.params[param1_name],
        best_trial.params[param2_name],
        color='red',
        s=200,
        marker='*',
        edgecolor='gold',
        linewidth=2,
        label=f'Best (score: {best_trial.value:.4f})'
    )
    
    plt.colorbar(scatter, label='Objective Value')
    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.title(f'Score-Based Contour: {param1_name} vs {param2_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Usage example:
# plot_score_contour(study, 'learning_rate', 'hidden_size')

from scipy.interpolate import griddata

# Create a grid
xi = np.linspace(df.param1.min(), df.param1.max(), 100)
yi = np.linspace(df.param2.min(), df.param2.max(), 100)
XI, YI = np.meshgrid(xi, yi)

# Interpolate scores onto grid (this truly ignores point density)
ZI = griddata(
    (df.param1, df.param2), 
    df.score, 
    (XI, YI), 
    method='cubic'
)

sns.set_style("white")
plt.figure(figsize=(10, 8))

# Create filled contours from interpolated scores
contour = plt.contourf(XI, YI, ZI, levels=15, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='Score Value')

# Optional: Add contour lines
plt.contour(XI, YI, ZI, levels=15, colors='black', linewidths=0.5, alpha=0.5)

# Add points sized by score value
scatter = plt.scatter(
    df.param1, df.param2,
    s=df.score/10,  # Size points by score
    c=df.score,     # Color points by score
    cmap='viridis',
    edgecolor='white',
    linewidth=0.5
)

plt.xlabel('Parameter 1')
plt.ylabel('Parameter 2')
plt.title('True Score-Based Contour (Grid Interpolation)')
plt.show()