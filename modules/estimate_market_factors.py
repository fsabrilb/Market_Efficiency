# -*- coding: utf-8 -*-
"""
Created on Thursday June 24 2024

@author: Laura Molero González
@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import numpy as np # type: ignore
import pandas as pd # type: ignore

from scipy.linalg import eigh # type: ignore

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
        Theoretical length of the time series such that q = N/n, where N is the
        number of financial time series (shares in a stock index)

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
    """Calculate probability of Tracy-Widom distribution given a z-score table

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
    random covariance matrix

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

# Estimate number of market factors from Tracy-Widom probability given statistical significance (alpha) ----
def get_market_factors(df_tracy_widom, eigen_values, n, alpha=0.01):
    """Compute the probability of the Tracy-Widom distribution assuming a
    Wishart distribution for the eigenvalues ​​of the non-noise part of the
    random covariance matrix

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

