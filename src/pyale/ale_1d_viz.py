import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyale.ale_1d_calcu import calculate_ale_1d

def ale_1d_plot(data, target_col, ale_ds, features_used=None, categorical_cols=[], centralization='median'):
    """
    Plots the ALE values, centralized as specified.

    Parameters:
    - data: DataFrame containing the dataset.
    - target_col: Name of the target column.
    - ale_ds: xarray Dataset containing ALE values.
    - features_used: List of features to plot. Defaults to all features in data excluding target_col.
    - categorical_cols: List of categorical columns.
    - centralization: Method of centralization ('zero', 'median', 'mean'). Defaults to 'median'.
    """
    # If features_used is not provided, default to the columns of X
    if features_used is None:
        features_used = data.drop(columns=[target_col]).columns.tolist()

    # Calculate the percentiles and centralization value for the target column
    percentiles_values = {
        '25%': data[target_col].quantile(0.25),
        '50%': data[target_col].median() if centralization in ['median', 'zero'] else data[target_col].mean(),
        '75%': data[target_col].quantile(0.75)
    }

    if centralization == 'zero':
        central_value = 0
        percentile_lines = {
            '25%': percentiles_values['25%'] - percentiles_values['50%'],
            '50%': percentiles_values['50%'] - percentiles_values['50%'],
            '75%': percentiles_values['75%'] - percentiles_values['50%']
        }
    else:
        central_value = percentiles_values['50%']
        percentile_lines = {
            '25%': percentiles_values['25%'],
            '50%': percentiles_values['50%'],
            '75%': percentiles_values['75%']
        }

    for feature in features_used:
        plt.figure(figsize=(10, 6))

        if feature in categorical_cols:
            # Categorical feature: bar plot
            ale_values = ale_ds[f'{feature}__LinearRegression__ale'].mean(dim='n_bootstrap').values
            if centralization == 'zero':
                centralised_ale_values = ale_values
            else:
                centralised_ale_values = ale_values + central_value

            bin_values = ale_ds[f'{feature}__bin_values'].values
            bin_labels = data[feature].cat.categories.tolist() if hasattr(data[feature], 'cat') else bin_values

            sns.barplot(x=bin_labels, y=centralised_ale_values, errorbar=None, palette="muted", alpha=0.5, label='ALE')
            plt.xlabel(feature)

        else:
            # Numerical feature: line plot
            ale_values = ale_ds[f'{feature}__LinearRegression__ale'].mean(dim='n_bootstrap').values
            if centralization == 'zero':
                centralised_ale_values = ale_values
            else:
                centralised_ale_values = ale_values + central_value
            bin_values = ale_ds[f'{feature}__bin_values'].values
            plt.plot(bin_values, centralised_ale_values, label='ALE', color='green')
            plt.xlabel(feature)

        plt.ylabel(f'{target_col} (Centralised ALE)')

        # Add percentile lines
        for label, value in percentile_lines.items():
            plt.axhline(y=value, linestyle='--', label=f'{label}: {value:.2f}', color='gray')

        plt.title(f'ALE Plot for {feature} (Centralised around {centralization})')
        plt.legend()
        plt.show()

