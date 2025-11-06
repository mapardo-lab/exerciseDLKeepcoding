import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_density(df: pd.DataFrame, features: list , n_rows: int, n_cols: int):
  """
  Density plots for selected features

  Args:
    df: data
    features: selected features to be plotted
    n_rows: number rows for the plot
    n_cols: number columns for the plot
  """
  plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Dynamic figure size
  for i, feature in enumerate(features,start=1):
      plt.subplot(n_rows, n_cols, i)
      sns.kdeplot(df[feature], fill=True, color='skyblue', alpha=0.5)
      plt.xlabel(feature)
      plt.ylabel('Density')
      plt.ticklabel_format(axis='both', style='sci', scilimits=(-3, 4))
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
  plt.figure(figsize=(5 * n_cols, 4 * n_rows))  # Dynamic figure size
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

def confusion_matrix_plot(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = cm / row_sums

    # Plot with actual counts as numbers but normalized values for colors
    plt.figure(figsize=(6, 4))
    sns.heatmap(normalized,  # Use normalized for colors
                annot=cm,       # Use actual counts for annotations
                fmt='d',        # Format annotations as integers
                cmap='Blues',
                vmin=0,         # Set minimum color value
                vmax=1,         # Set maximum color value
                cbar_kws={'label': 'Normalized values by rows'})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()