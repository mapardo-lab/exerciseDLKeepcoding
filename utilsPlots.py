import matplotlib.pyplot as plt
import numpy  as np  
import pandas as pd
import seaborn as sns
import optuna

def plot_density(df: pd.DataFrame, features: list , n_rows: int, n_cols: int):
  """
  Density plots for selected features

  Args:
    df: data
    features: selected features to be plotted
    n_rows: number rows for the plot
    n_cols: number columns for the plot
  """
  plt.figure(figsize=(8 * n_cols, 4 * n_rows))  # Dynamic figure size
  for i, feature in enumerate(features,start=1):
      plt.subplot(n_rows, n_cols, i)
      sns.kdeplot(df[feature], fill=True, color='skyblue', alpha=0.5)
      plt.xlabel(feature)
      plt.ylabel('Density')
  plt.show()

def plot_bars(df: pd.DataFrame, features: list , n_rows: int, n_cols: int):
  """
  Bar plots for selected features

  Args:
    df: data
    features: selected features to be plotted
    n_rows: number rows for the plot
    n_cols: number columns for the plot
  """
  plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Dynamic figure size
  for i, feature in enumerate(features, start=1):
    counts = df[feature].value_counts()
    counts = counts.loc[sorted(counts.index)]
    plt.subplot(n_rows, n_cols, i)
    # Plot value counts as bars
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    # Customize
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Prevent label clipping
  plt.show()