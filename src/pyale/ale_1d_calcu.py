# ale_1d_calcu.py

import skexplain

def calculate_ale_1d(model_type, model, X, y, n_bootstrap=2, subsample=10000, n_jobs=1, n_bins=20):
    """
    Calculate 1D Accumulated Local Effects (ALE) for a given model.

    Parameters:
        model_type (str): The type of the model, e.g., 'LinearRegression'.
        model (sklearn.base.BaseEstimator): The trained model instance.
        X (np.ndarray or pd.DataFrame): The feature matrix.
        y (np.ndarray or pd.Series): The target vector.
        n_bootstrap (int): Number of bootstrap samples. Default is 2.
        subsample (int): Number of samples to use in each bootstrap sample. Default is 10000.
        n_jobs (int): Number of parallel jobs. Default is 1.
        n_bins (int): Number of bins to use for discretizing continuous features. Default is 20.

    Returns:
        ale_1d_ds: The ALE results as an xarray.Dataset.
    """
    # Supported model types - ensure that these are compatible with skexplain
    supported_model_types = [
        'LinearRegression', 'RandomForestRegressor', 'GradientBoostingRegressor', 
        'LogisticRegression', 'DecisionTreeRegressor', 'SVR'
    ]

    if model_type not in supported_model_types:
        raise ValueError(f"Unsupported model type. Supported types are: {', '.join(supported_model_types)}.")

    # Initialize the explainer
    explainer = skexplain.ExplainToolkit((model_type, model), X=X, y=y)

    # Calculate ALE
    ale_1d_ds = explainer.ale(
        features='all',
        n_bootstrap=n_bootstrap,
        subsample=subsample,
        n_jobs=n_jobs,
        n_bins=n_bins
    )

    return ale_1d_ds
