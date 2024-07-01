# -*- coding: utf-8 -*-
"""
Created on Thursday June 24 2024

@author: Laura Molero González
@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import numpy as np # type: ignore
import pandas as pd # type: ignore
import misc_functions as mf
import estimate_entropy as ee
import get_financial_time_series as get_fts

from functools import partial
from scipy.linalg import eigh # type: ignore
from scipy.stats import linregress # type: ignore

# Estimation of Sharpe model ----
def estimate_sharpe_model(df):
    """Sharpe model (diagonal model for a market returns):

    Args:
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe of financial time series with the following columns:
            - Adjusted closed value ("Adj Close")
            - Logarithmic return ("log_return")
            - Normalized logarithmic return through z-score transformation
            ("z_score_log_return")

    Returns:
    ---------------------------------------------------------------------------
    df_sharpe : pandas DataFrame
        Dataframe of financial time series with the following added columns (
        per component or stock for logarithmic returns and normalized
        logarithmic returns):
            - Overall market performance ("market_(z_score)_log_return")
            - Regressor estimated from the Sharpe model ("regressor_(z)lr")
            - Volatility estimated from the Sharpe model ("volatility_(z)lr")
            - Standard deviation of regressor estimated from the Sharpe model
            ("regressor_(z)lr_std")
            - Standard deviation of volatility estimated from the Sharpe model
            ("volatility_(z)lr_std")
            - Coefficient of determination of the Sharpe model
            ("rsquared_(z)lr")
            - p-value of the Sharpe model ("pvalue_(z)lr")
            - No market component (residuals after Sharpe model fitting is
            substracted in log-returns and normalized log-returns
            ("lr_no_market" and "zlr_no_market")
            - Final normalization through z-score of the no market component
            ("z_score_lr_no_market" and "z_score_zlr_no_market")
    """

    # Average Market Return (Herinafter LR: log-returns and ZLR: z-score log-returns)
    df["market_log_return"] = df.groupby(["date"])["log_return"].transform("mean")
    df["market_z_score_log_return"] = df.groupby(["date"])["z_score_log_return"].transform("mean")

    # Linear regression per component of financial time series (shares for stock indexes)
    df_sharpe = []
    for stock in df["symbol"].unique():
        lr = df[df["symbol"] == stock]["log_return"].values
        zlr = df[df["symbol"] == stock]["z_score_log_return"].values
        market_lr = df[df["symbol"] == stock]["market_log_return"].values
        market_zlr = df[df["symbol"] == stock]["market_z_score_log_return"].values
        linear_regression_lr = linregress(market_lr, lr)
        linear_regression_zlr = linregress(market_zlr, zlr)

        df_sharpe.append(
            pd.DataFrame(
                {
                    "symbol" : [stock],
                    "regressor_lr" : [linear_regression_lr.intercept],
                    "volatility_lr" : [linear_regression_lr.slope],
                    "regressor_lr_std" : [linear_regression_lr.intercept_stderr],
                    "volatility_lr_std" : [linear_regression_lr.stderr],
                    "rsquared_lr" : [linear_regression_lr.rvalue**2],
                    "pvalue_lr" : [linear_regression_lr.pvalue],
                    "regressor_zlr" : [linear_regression_zlr.intercept],
                    "volatility_zlr" : [linear_regression_zlr.slope],
                    "regressor_zlr_std" : [linear_regression_zlr.intercept_stderr],
                    "volatility_zlr_std" : [linear_regression_zlr.stderr],
                    "rsquared_zlr" : [linear_regression_zlr.rvalue**2],
                    "pvalue_zlr" : [linear_regression_zlr.pvalue]
                }
            )
        )

        print("- Finished Sharpe model for stock: {}".format(stock))
    
    # Final Dataframe of sharpe model
    df_sharpe = pd.concat(df_sharpe)
    df_sharpe = df.merge(right = df_sharpe, how = "left", on = ["symbol"])

    # Estimation of residuals or errors in Sharpe model (non-market components)
    df_sharpe["lr_no_market"] = (
        df_sharpe["log_return"]
            - df_sharpe["regressor_lr"]
            - df_sharpe["volatility_lr"] * df_sharpe["market_log_return"]
    )

    df_sharpe["zlr_no_market"] = (
        df_sharpe["z_score_log_return"]
            - df_sharpe["regressor_zlr"]
            - df_sharpe["volatility_zlr"] * df_sharpe["market_z_score_log_return"]
    )

    df_sharpe["temp_lr_mean"] = df_sharpe.groupby(["symbol"])["lr_no_market"].transform("mean")
    df_sharpe["temp_zlr_mean"] = df_sharpe.groupby(["symbol"])["zlr_no_market"].transform("mean")

    df_sharpe["temp_lr_std"] = df_sharpe.groupby(["symbol"])["lr_no_market"].transform("std")
    df_sharpe["temp_zlr_std"] = df_sharpe.groupby(["symbol"])["zlr_no_market"].transform("std")
    
    df_sharpe["z_score_lr_no_market"] = (
        (df_sharpe["lr_no_market"] - df_sharpe["temp_lr_mean"]) / df_sharpe["temp_lr_std"]
    )
    df_sharpe["z_score_zlr_no_market"] = (
        (df_sharpe["zlr_no_market"] - df_sharpe["temp_zlr_mean"]) / df_sharpe["temp_zlr_std"]
    )

    del df_sharpe["temp_lr_mean"], df_sharpe["temp_zlr_mean"], df_sharpe["temp_lr_std"], df_sharpe["temp_zlr_std"]

    return df_sharpe

# Apply Bouchaud's clipping filter ----
def clipping_covariance_matrix(covariance_matrix, n):
    """Bouchaud's clipping filter for the noise and non-noise decomposition of
    the eigenvalues ​​of the covariance random matrix (Marchenko-Pastur
    threshold):

    Args:
    ---------------------------------------------------------------------------
    covariance_matrix : pandas DataFrame or numpy 2D array
        Covariance matrix to be diagonalized
    n : float
        Theoretical length of the time series such that q = p/n, where p is the
        number of components in financial time series (shares in a stock index)

    Returns:
    ---------------------------------------------------------------------------
    dropped_eigenvalues : int
        Number of dropped eigenvalues that corresponds to the noise of random
        matrix
    new_covariance_matrix : float
        New covariance matrix constructed by averaging the eigenvalues ​​that
        correspond to the noise
    """

    # Diagonalized matrix
    eigenvalues, eigenvectors = eigh(covariance_matrix)
    
    # Marchenko-Pastur limit MPL (Theoretical maximum eigenvalue)
    eigenvalue_max = (1.0 + np.sqrt(len(covariance_matrix) / n))**2

    # Replace the noise (Eigenvalues below MPL)
    noise = eigenvalues[eigenvalues < eigenvalue_max]
    if len(noise) > 0:
        eigenvalues[eigenvalues < eigenvalue_max] = np.mean(noise)
    
    dropped_eigenvalues = len(covariance_matrix) - len(noise)

    # Covariance matrix after decompose the noise
    new_covariance_matrix = np.cov(eigenvectors@np.diag(eigenvalues)@eigenvectors.T)

    return(dropped_eigenvalues, new_covariance_matrix)

# Estimate probability of Tracy-Widom distribution given z-score value (quantile function) ----
def estimate_tracy_widom_probability(df_tracy_widom, z_score):
    """Calculate probability of Tracy-Widom distribution given a z-score table:

    Args:
    ---------------------------------------------------------------------------
    df_tracy_widom : pandas DataFrame
        Dataframe of quantiles of Tracy-Widom distribution:
            - z score ("z_score")
            - Probability of Tracy-Widom distribution ("probability")
    z_score : float
        z score used to estimate probability of Tracy-Widom distribution

    Returns:
    ---------------------------------------------------------------------------
    prob : float
        Probability of Tracy-Widom distribution obtained from table or as
        linear interpolation between closest values to z_score
    """

    # z-score in left tail
    if z_score <= df_tracy_widom["z_score"].min():
        prob = 1.0
    # z-score in right tail
    elif z_score >= df_tracy_widom["z_score"].max():
        prob = 0.0
    else:
        df_local_tracy_widom = df_tracy_widom[df_tracy_widom["z_score"] < z_score].tail(2)
        
        # z-score as interpolation of two points
        if df_local_tracy_widom.shape[0] > 1:
            x_0 = df_local_tracy_widom["z_score"].iloc[0]
            y_0 = df_local_tracy_widom["probability"].iloc[0]
            delta_x = df_local_tracy_widom["z_score"].iloc[1] - x_0
            delta_y = df_local_tracy_widom["probability"].iloc[1] - y_0
            prob = y_0 + (delta_y / delta_x) * (z_score - x_0)
        
        # z-score as interpolation of one point
        else:
            prob = df_local_tracy_widom["probability"].iloc[0]
    
    return(prob)

# Estimate Wishart distribution (multivariate Gamma distribution) from Tracy-Widom probability ----
def estimate_wishart_order_2(p, n, df_tracy_widom, lambda_1):
    """Compute the probability of the Tracy-Widom distribution assuming a
    Wishart distribution for the eigenvalues ​​of the non-noise part of the
    random covariance matrix:

    Args:
    ---------------------------------------------------------------------------
    p : int
        Number of different components in covariance matrix (shares for stock
        indexes)
    n : int
        Theoretical length of the time series such that q = N/n, where N is the
        number of financial time series (shares in a stock index)
    df_tracy_widom : pandas DataFrame
        Dataframe of quantiles of Tracy-Widom distribution:
            - z score ("z_score")
            - Probability of Tracy-Widom distribution ("probability")
    lambda_1 : float
        Eigenvalue threshold value between noise and non-noise
        (Marchenko-Pastur threshold)

    Returns:
    ---------------------------------------------------------------------------
    probability : float
        Probability of Tracy-Widom distribution obtained from table or as
        linear interpolation between closest values to z_score_lambda
        (probability that lambda > lambda_1)
    """

    mu = np.power(np.sqrt(n - 0.5) + np.sqrt(p - 0.5), 2)
    sigma = np.power(np.sqrt(mu / (n - 0.5)) + np.sqrt(mu / (p - 0.5)), 1.0 / 3)
    z_score_lambda = (n * lambda_1 - mu) / sigma
    probability = estimate_tracy_widom_probability(df_tracy_widom = df_tracy_widom, z_score = z_score_lambda)
    return(probability)

# Estimate number of market components from Tracy-Widom probability given statistical significance (alpha) ----
def get_market_components(df_tracy_widom, eigen_values, n, alpha=0.01):
    """Calculate the number of components in a market using the probability of
    the Tracy-Widom distribution assuming a Wishart distribution for the
    eigenvalues ​​of the non-noise part of the random covariance matrix:

    Args:
    ---------------------------------------------------------------------------
    df_tracy_widom : pandas DataFrame
        Dataframe of quantiles of Tracy-Widom distribution:
            - z score ("z_score")
            - Probability of Tracy-Widom distribution ("probability")
    eigen_values : numpy array 1D
        Vector with the eigenvalues of the covariance random matrix
    n : int
        Theoretical length of the time series such that q = N/n, where N is the
        number of financial time series (shares in a stock index)
    alpha : float
        Level of statistical significance. For instance alpha=0.01 corresponds
        to a 99% confidence interval
    
    Returns:
    ---------------------------------------------------------------------------
    k : int
        Number of statistically significant eigenvalues (they are below the
        level of statistical significance)
    """
    
    statistical_significances = np.array([])
    for eig_val in eigen_values[np.argsort(-eigen_values)]:
        statistical_significances = np.append(
            statistical_significances,
            estimate_wishart_order_2(
                p = len(eigen_values),
                n = n,
                df_tracy_widom = df_tracy_widom,
                lambda_1 = eig_val
            )
        )

    k = np.argmax(statistical_significances >= alpha)
    return(k)

# Estimate Onatski R statistic vector ----
def estimate_onatski_statistic(df, k_max=8):
    """Compute the Onatski R statistics:

    Args:
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe of financial time series ordered such that every column
        corresponds to a component in financial time series (shares for stock
        indexes)
    k_max : int
        Maximum number of factors considered in the Onatski test (default value
        is 8)

    Returns:
    ---------------------------------------------------------------------------
    r_statistic : numpy array 1D
        Onatski statistical values used to determine the number of
        statistically significant factors in a random matrix
    """

    # Splitting matrix for construct Gaussian unitary ensemble (GUE)
    data_1 = df.iloc[:int(df.shape[0] / 2),:]
    data_2 = df.iloc[int(df.shape[0] / 2):,:]
    X = data_1.values + data_2.values * 1j
    gue = pd.DataFrame((1 / data_1.shape[0]) * X.conjugate().T@X)
    eigenvalues_gue = eigh(gue)[0]

    # Estimates R statitic (invariant under center and scaling of the eigenvalues)
    eigenvalues_gue = (np.flipud(eigenvalues_gue) - 2) * np.power(data_1.shape[0], 2.0 / 3)
    
    r_statistic = np.zeros(len(eigenvalues_gue) - 2)
    for i in range(r_statistic):
        delta_gue = eigenvalues_gue[i + 1] - eigenvalues_gue[i + 2]
        if delta_gue != 0:
            r_statistic[i] = (eigenvalues_gue[i] - eigenvalues_gue[i + 1]) / delta_gue
        else:
            r_statistic[i] = 0
    r_statistic = r_statistic[:k_max]
    
    return r_statistic

# Estimate number of market factors from Tracy-Widom probability given statistical significance (alpha) ----
def get_significal_test_onatski(df_onatski, r_statistics, level=1):
    """Calculate the number of statistically significant factors in a market
    using the z-score table of the Onatski test and Onatski R statistic:

    Args:
    ---------------------------------------------------------------------------
    df_onatski : pandas DataFrame
        Dataframe of quantiles of Onatski test distribution:
            - Level ("level")
            - z-score of the R statistic according to the level (other columns)
    r_statistics : numpy array 1D
        Vector with the R statistic values according to Onatski definition with
        same components of df_onatski columns (see logical values in this function)
    level : int
        Integer value for filtering df_onatski
    
    Returns:
    ---------------------------------------------------------------------------
    factors : int
        Number of statistically significant factors for a market
    """

    # Modify R statistics vector for taking into account the maximum value per component
    r_statistics_new = np.zeros(len(r_statistics))
    for i in range(len(r_statistics)):
        if i == 0:
            r_statistics_new[i] = r_statistics[0]
        else:
            r_statistics_new[i] = max(r_statistics_new[i - 1], r_statistics[i])

    # Estimation of number of factors
    z_scores = df_onatski[df_onatski["level"] == level].drop(columns = ["level"]).astype(float).values[0]
    logical = r_statistics_new > z_scores
    
    if np.all(logical) == True:
        factors = 8
    else:
        factors = np.argmax(logical == False)
    
    return factors

# Deployment of total Efficiency Analysis in a time window selected ----
def get_market_efficiency_data_window(
    df,
    column_,
    min_bins,
    precision,
    log_path,
    log_filename,
    log_filename_entropy,
    verbose,
    tqdm_bar,
    market_args_list
):
    """Estimate covariance and entropy matrix of financial time series for a
    selected time window such that:
        initial_date = market_args_list[0]
        final_date   = market_args_list[1]

    Args:
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe of financial time series with the following columns:
            - Adjusted closed value ("Adj Close")
            - Logarithmic return ("log_return")
            - Normalized logarithmic return through z-score transformation
            ("z_score_log_return")
            - No market components or residuals if Sharpe model is executed
            (see estimate_sharpe_model function in this module)
    column_ : str
        Column of financial time series dataframe used to estimate covariance
        matrix (default value "z_score_log_return")
    min_bins : int
        Minimum number of bins accepted to estimate the entropies
    precision : int
        The precision at which to store and display the bins labels (default
        value 12)
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_market_efficiency")
    log_filename_entropy : string
        Local filename for logs of entropy matrix (default value is "log_entropy")
    verbose : int
        Provides additional details as to what the computer is doing when
        entropy estimation is running (default value is 1)
    tqdm_bar : bool
        Progress bar in parallel run (default value is True)
    initial_date: str
        Initial date for time series in ISO format (YYYY-MM-DD)
    final_date : str
        Final date for time series in ISO format (YYYY-MM-DD)
    
    Returns:
    ---------------------------------------------------------------------------
    df_final : pandas DataFrame
        Dataframe with the covariances and entropies of different financial
        time series components (shares for stock indexes)
    """

    # Parameters definition and transform to datetime
    initial_date = market_args_list[0]
    final_date = market_args_list[1]

    # Filtered data per window dates
    df_local = df[((df["date"] >= initial_date) & (df["date"] <= final_date))]

    # Local Sharpe model
    df_local = estimate_sharpe_model(df = df_local)

    # Function development
    if verbose >= 1:
        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
            file.write("- Sharpe Model done for dates: {} - {}\n".format(initial_date, final_date))

    # Estimate covariance matrix
    df_cov = get_fts.estimate_covariance_stock_index(df = df_local, column_ = column_)
    symbols_dict = dict(enumerate(df_cov.columns))
    
    df_cov = df_cov.unstack().reset_index()
    df_cov.columns = ["symbol_x", "symbol_y", "correlation"]
    df_cov["symbol_y"] = df_cov["symbol_y"].replace(symbols_dict)

    # Estimate entropy matrix
    df_entropy = ee.estimate_entropy_matrix(
        df = df_local,
        min_bins = min_bins,
        precision = precision,
        column_= column_,
        log_path = log_path,
        log_filename = log_filename_entropy,
        verbose = verbose,
        tqdm_bar = tqdm_bar
    )
    df_entropy["modified_rajski_distance"] = 1 - df_entropy["rajski_distance"]

    # Function development
    if verbose >= 1:
        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
            file.write("- Covariance and entropy matrix done for dates: {} - {}\n".format(initial_date, final_date))

    # Merge all data in single dataframe
    df_final = df_entropy.merge(right = df_cov, how = "left", on = ["symbol_x", "symbol_y"])
    df_final["initial_date"] = initial_date
    df_final["final_date"] = final_date
    df_final.insert(0, "initial_date", df_final.pop("initial_date"))
    df_final.insert(1, "final_date", df_final.pop("final_date"))
    
    # Function development
    if verbose >= 1:
        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
            file.write("- Final data done for dates: {} - {}\n".format(initial_date, final_date))

    return df_final

def get_market_efficiency(
    df,
    column_,
    min_bins,
    precision,
    log_path,
    log_filename,
    log_filename_entropy,
    verbose,
    tqdm_bar,
    market_args_list,
    bouchaud_filter,
    n,
    df_tracy_widom,
    alphas
):
    """Estimate covariance and entropy matrix of financial time series
    according to:
        initial_date = market_args_list[k, 0]
        final_date   = market_args_list[k, 1]
    for k in {1, 2,..., n_dates}, where n_dates represent all possible pairs
    in dates of financial time series after a size of time window is selected
        
    Args:
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe of financial time series with the following columns:
            - Adjusted closed value ("Adj Close")
            - Logarithmic return ("log_return")
            - Normalized logarithmic return through z-score transformation
            ("z_score_log_return")
            - No market components or residuals if Sharpe model is executed
            (see estimate_sharpe_model function in this module)
    column_ : list of str
        Column of financial time series dataframe used to estimate covariance
        matrix (default value "z_score_log_return")
    min_bins : int
        Minimum number of bins accepted to estimate the entropies
    precision : int
        The precision at which to store and display the bins labels (default
        value 12)
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_market_efficiency")
    log_filename_entropy : string
        Local filename for logs of entropy matrix (default value is "log_entropy")
    verbose : int
        Provides additional details as to what the computer is doing when
        entropy estimation is running (default value is 1)
    tqdm_bar : bool
        Progress bar in parallel run (default value is True)
    initial_date: str
        Initial date for time series in ISO format (YYYY-MM-DD)
    final_date : str
        Final date for time series in ISO format (YYYY-MM-DD)
    bouchaud_filter : bool
        Bool variable for the application of Bouchaud filter
        (default value False)
    df_tracy_widom : pandas DataFrame
        Dataframe of quantiles of Tracy-Widom distribution:
            - z score ("z_score")
            - Probability of Tracy-Widom distribution ("probability")
    n : float
        Theoretical length of the time series such that q = p/n, where p is the
        number of components in financial time series (shares in a stock index)
    
    Returns:
    ---------------------------------------------------------------------------
    df_final : pandas DataFrame
        Dataframe with the covariances and entropies of different financial
        time series components (shares for stock indexes)
    """

    # Get data for estimation of market efficiency
    df_data = get_market_efficiency_data_window(
        df = df,
        column_ = column_,
        min_bins = min_bins,
        precision = precision,
        log_path = log_path,
        log_filename = log_filename,
        log_filename_entropy = log_filename_entropy,
        verbose = verbose,
        tqdm_bar = tqdm_bar,
        market_args_list = market_args_list
    )

    # Apply Bouchaud's clipping filter ----
    df_cov = (
        df_data[["symbol_x", "symbol_y", "correlation"]]
            .sort_values(["symbol_x", "symbol_y", "correlation"], ascending = [True, True, True])
            .drop_duplicates(subset = ["symbol_x", "symbol_y"])
            .pivot(index = "symbol_x", columns = "symbol_y", values = "correlation")
    )

    df_entropy = (
        df_data[["symbol_x", "symbol_y", "modified_rajski_distance"]]
            .sort_values(["symbol_x", "symbol_y", "modified_rajski_distance"], ascending = [True, True, True])
            .drop_duplicates(subset = ["symbol_x", "symbol_y"])
            .pivot(index = "symbol_x", columns = "symbol_y", values = "modified_rajski_distance")
    )

    if bouchaud_filter:
        dropped_eigenvalues_cov, df_cov = clipping_covariance_matrix(covariance_matrix = df_cov, n = n)
        dropped_eigenvalues_entropy, df_entropy = clipping_covariance_matrix(covariance_matrix = df_entropy, n = n)
    else:
        dropped_eigenvalues_cov = 0
        dropped_eigenvalues_entropy = 0

    # Estimation of eigenvalues for matrices (Covariance and Entropy)
    eigenvalues_cov = eigh(df_cov)[0]
    eigenvalues_entropy = eigh(df_cov)[0]

    # Apply Tracy-Widom test for market components estimation
    for alpha_ in alphas:
        try:
            components_cov = get_market_components(
                df_tracy_widom = df_tracy_widom,
                eigen_values = eigenvalues_cov,
                n = n,
                alpha = alpha_
            )

            components_entropy = get_market_components(
                df_tracy_widom = df_tracy_widom,
                eigen_values = eigenvalues_entropy,
                n = n,
                alpha = alpha_
            )
        except:
            components_cov = 0
            components_entropy = 0


    # dropped_eigenvalues_cov, dropped_eigenvalues_entropy
    # components_cov, components_entropy
    # RESIDUALS
	# Add number of dropped eigenvalues
	# Tracy widow test with Alpha 1,5,10 (try:, except)

    # Onatski
        # R statistics vectorc(try:, except) for returns and residuals
    # Factor Onatski Levels 1, 5, 10(fuera ciclo in range(m):)

    #Final data
    #    * Factors Returns L1,5,10
    #    * Factors Residuals L1,5,10
    #    * Components Tracy-Widow 

    return df_final

# Apply Bouchaud's clipping filter ----
#clipping_covariance_matrix(covariance_matrix, n)