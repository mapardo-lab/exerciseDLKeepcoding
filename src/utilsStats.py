import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols

def cramers_v(x, y, bias_correction=False):
    """
    Calculate Cramer's V statistic with optional bias correction
    Bias correction for smaller sample size or when number of
    categories in the variables is large.
    
    Args:
        x, y: pandas Series or array-like categorical data
        bias_correction: Whether to apply Bergsma's bias correction
        
    Returns:
        Cramer's V between 0 (no association) and 1 (perfect association)
        0.0 - 0.1 No association
        0.1 - 0.3 Weak association
        0.3 - 0.5 Moderate association
        > 0.5 Strong association
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    if bias_correction:
        # Bergsma's bias correction
        phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        r_corr = r - ((r-1)**2)/(n-1)
        k_corr = k - ((k-1)**2)/(n-1)
        denominator = min((k_corr-1), (r_corr-1))
    else:
        phi2_corr = phi2
        denominator = min(k-1, r-1)
    
    # Handle edge case where denominator is 0
    if denominator <= 0:
        return 0.0
    
    v = np.sqrt(phi2_corr / denominator)
    return min(v, 1.0)  # Ensure result doesn't exceed 1 due to floating point

def eta_squared(df, formula):
    """
    Calculate eta-squared (η²) effect size for an ANOVA model.

    Eta-squared measures the proportion of total variance in the dependent variable
    that is attributable to the independent variable(s). It ranges from 0 to 1,
    where higher values indicate stronger effects.

    Values > 0.06 indicate a moderate effect
    Values > 0.14 indicate a large effect
    """

    # Fit ANOVA model
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Calculate Eta-squared
    ss_between = anova_table['sum_sq'].iloc[0]
    ss_total = ss_between + anova_table['sum_sq'].iloc[1]
    eta_squared = ss_between / ss_total
    
    return eta_squared
