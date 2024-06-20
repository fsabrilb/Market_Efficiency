# -*- coding: utf-8 -*-
"""
Created on Thursday June 24 2024

@author: Laura Molero González
@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
import misc_functions as mf

from functools import partial

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Binning the financial time series components for the entropy estimation ----
def binning_information_pairs(df, symbol_x, symbol_y, bins_x, bins_y, precision=12, normalized=True):
    """Estimate marginal and joint probabilities of two components financial
    time series:

    Args:
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe of financial time series with the following columns:
            - Adjusted closed value ("Adj Close")
            - Logarithmic return ("log_return")
            - Normalized logarithmic return ("normalized_log_return")
    symbol_x : str
        Ticker assigned in Yahoo finance for a component in financial time
        series (shares for stock indexes). First symbol named as x
    symbol_y : str
        Ticker assigned in Yahoo finance for a component in financial time
        series (shares for stock indexes). Second symbol named as y
    bins_x : int
        The bin specification for first ticker (symbol_x) as pd.cut()
    bins_y : int
        The bin specification for second ticker (symbol_y) as pd.cut()
    precision : int
        The precision at which to store and display the bins labels (default
        value 12)
    normalized : bool
        Boolean variable for selection of normalized log-return (default value
        True)
    
    Returns:
    ---------------------------------------------------------------------------
    df_1 : pandas DataFrame
        Dataframe with the marginal probabilities of financial time series
        associated with symbol_x (shares for stock indexes)
    df_2 : pandas DataFrame
        Dataframe with the marginal probabilities of financial time series
        associated with symbol_y (shares for stock indexes)
    df_3 : pandas DataFrame
        Dataframe with the joint probabilities of two financial time series
        associated with symbol_x and symbol_y (shares for stock indexes)
    """

    # Column selection
    if normalized == True:
        column_ = "normalized_log_return"
    else:
        column_ = "log_return"

    # Normalize variables for correct and comparable binningg
    df_binning = (
        df[df["symbol"].isin([symbol_x, symbol_y])][["date", "symbol", column_]]
            .pivot(index = "date", columns = "symbol", values = column_)
            .reset_index()
    )

    df_binning["rank_x"] = df_binning[symbol_x].max() - df_binning[symbol_x].min()
    df_binning["rank_y"] = df_binning[symbol_y].max() - df_binning[symbol_y].min()
    df_binning[symbol_x] = (df_binning[symbol_x] - df_binning[symbol_x].min()) / df_binning["rank_x"]
    df_binning[symbol_y] = (df_binning[symbol_y] - df_binning[symbol_y].min()) / df_binning["rank_y"]

    # Binning data
    df_binning["bin_x"] = pd.cut(
        df_binning[symbol_x],
        bins = np.linspace(0, 1, num = bins_x + 1),
        include_lowest = True,
        labels = ["".join(("category_", str(z + 1).zfill(int(np.log10(bins_x)) + 1))) for z in range(bins_x)],
        precision = precision
    )

    df_binning["bin_y"] = pd.cut(
        df_binning[symbol_y],
        bins = np.linspace(0, 1, num = bins_y + 1),
        include_lowest = True,
        labels = ["".join(("category_", str(z + 1).zfill(int(np.log10(bins_y)) + 1))) for z in range(bins_y)],
        precision = precision
    )

    # Histogram data (marginal probabilities and joint probabilities)
    df_1 = (
        df_binning[["bin_x"]]
            .value_counts(sort = False, normalize = True)
            .reset_index()
            .rename(columns = {"proportion" : "prob_x"})
            .sort_values(["bin_x"])
    )

    df_2 = (
        df_binning[["bin_y"]]
            .value_counts(sort = False, normalize = True)
            .reset_index()
            .rename(columns = {"proportion" : "prob_y"})
            .sort_values(["bin_y"])
    )

    df_3 = (
        df_binning[["bin_x", "bin_y"]]
            .value_counts(sort = False, normalize = True)
            .reset_index()
            .rename(columns = {"proportion" : "joint_xy"})
            .sort_values(["bin_x", "bin_y"])
    )

    return df_1, df_2, df_3

# Estimation of entropies (joint entropy, mutual information, shannon entropy) ----
def estimate_entropy_pairs(
    df,
    precision,
    normalized,
    log_path,
    log_filename,
    verbose,
    entropy_args_list
):
    """Estimate entropies of financial time series according to:
        symbol_x = entropy_args_list[0]
        symbol_y = entropy_args_list[1]
        bins_x   = entropy_args_list[2]
        bins_y   = entropy_args_list[3]

    Args:
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe of financial time series with the following columns:
            - Adjusted closed value ("Adj Close")
            - Logarithmic return ("log_return")
            - Normalized logarithmic return ("normalized_log_return")
    precision : int
        The precision at which to store and display the bins labels (default
        value 12)
    normalized : bool
        Boolean variable for selection of normalized log-return (default value
        True)
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_entropy")
    verbose : int
        Provides additional details as to what the computer is doing when
        entropy estimation is running (default value is 1)
    symbol_x : str
        Ticker assigned in Yahoo finance for a component in financial time
        series (shares for stock indexes). First symbol named as x
    symbol_y : str
        Ticker assigned in Yahoo finance for a component in financial time
        series (shares for stock indexes). Second symbol named as y
    bins_x : int
        The bin specification for first ticker (symbol_x) as pd.cut()
    bins_y : int
        The bin specification for second ticker (symbol_y) as pd.cut()
    
    Returns:
    ---------------------------------------------------------------------------
    df_final : pandas DataFrame
        Dataframe with the entropies (Shannon, mutual information, shared
        information) of financial time series associated with symbol_x and
        symbol_y(shares for stock indexes)
    """

    # Definition of Brownian motion parameters
    symbol_x = entropy_args_list[0]
    symbol_y = entropy_args_list[1]
    bins_x = entropy_args_list[2]
    bins_y = entropy_args_list[3]

    # Binning information
    df_1, df_2, df_3 = binning_information_pairs(
        df = df,
        symbol_x = symbol_x,
        symbol_y = symbol_y,
        bins_x = bins_x,
        bins_y = bins_y,
        precision = precision,
        normalized = normalized
    )

    # Entropy estimation (Shannon entropy and joint entropy)
    hx = mf.estimate_renyi_entropy(x = df_1["prob_x"], p = 1)
    hy = mf.estimate_renyi_entropy(x = df_2["prob_y"], p = 1)
    hxy = mf.estimate_renyi_entropy(x = df_3["joint_xy"], p = 1)
    
    x_ = df_1["prob_x"].values
    y_ = df_2["prob_y"].values
    joint_xy = df_3["joint_xy"].values

    # Entropy estimation (Mutual information)
    mi_xx = mf.estimate_mutual_information(x = x_, y = x_, joint_xy = joint_xy)
    mi_xy = mf.estimate_mutual_information(x = x_, y = y_, joint_xy = joint_xy)
    mi_yx = mf.estimate_mutual_information(x = y_, y = x_, joint_xy = joint_xy)
    mi_yy = mf.estimate_mutual_information(x = y_, y = y_, joint_xy = joint_xy)
    
    # Entropy estimation (Variance of information)
    si_xx = mf.estimate_shared_information_distance(x = x_, y = x_, joint_xy = joint_xy)
    si_xy = mf.estimate_shared_information_distance(x = x_, y = y_, joint_xy = joint_xy)
    si_yx = mf.estimate_shared_information_distance(x = y_, y = x_, joint_xy = joint_xy)
    si_yy = mf.estimate_shared_information_distance(x = y_, y = y_, joint_xy = joint_xy)

    # Final information
    df_final = pd.DataFrame(
        {
            "symbol_x" : [symbol_x, symbol_x, symbol_y, symbol_y],
            "symbol_y" : [symbol_x, symbol_y, symbol_x, symbol_y],
            "entropy_x" : [hx, hx, hx, hx],
            "entropy_y" : [hy, hy, hy, hy],
            "joint_entropy" : [hxy, hxy, hxy, hxy],
            "mutual_information" : [mi_xx, mi_xy, mi_yx, mi_yy],
            "shared_information" : [si_xx, si_xy, si_yx, si_yy]
        }
    )

    # Entropy estimation (Jaccard distance)
    df_final["modified_jaccard_distance"] = df_final["shared_information"] / df_final["joint_entropy"]

    # Function development
    if verbose >= 1:
        with open("{}/{}.txt".format(log_path, log_filename), "a") as file:
            file.write(
                "Entropy estimation: s_x={}, bins_x={}, s_y={}, bins_y={}, precision={}, normalized={}\n".format(
                    symbol_x,
                    bins_x,
                    symbol_y,
                    bins_y,
                    precision,
                    normalized
                )
            )

    return df_final

# Matrix construction with the mutual information metric ----
def estimate_entropy_matrix(
    df,
    min_bins,
    precision=12,
    normalized=True,
    log_path="../logs",
    log_filename="log_entropy",
    verbose=1,
    tqdm_bar=True
):
    """Estimation of multiple entropies according to:
        symbol_x    = entropy_args_list[k, 0]
        symbol_y    = entropy_args_list[k, 1]
        bins_x      = entropy_args_list[k, 2]
        bins_y      = entropy_args_list[k, 3]
    for k in {1, 2,..., n_pairs}, where n_pairs represent all possible pairs
    in components of financial time series

    Args
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe of financial time series with the following columns:
            - Adjusted closed value ("Adj Close")
            - Logarithmic return ("log_return")
            - Normalized logarithmic return ("normalized_log_return")
    min_bins : int
        Minimum number of bins accepted to estimate the entropies
    precision : int
        The precision at which to store and display the bins labels (default
        value 12)
    normalized : bool
        Boolean variable for selection of normalized log-return (default value
        True)
    log_path : string
        Local path for logs (default value is "../logs")
    log_filename : string
        Local filename for logs (default value is "log_entropy")
    verbose : int
        Provides additional details as to what the computer is doing when
        entropy estimation is running (default value is 1)
    tqdm_bar : bool
        Progress bar in parallel run (default value is True)

    Returns
    ---------------------------------------------------------------------------
    df_entropy : pandas DataFrame
        Dataframe with entropy matrix information of financial time series
    """

    # Auxiliary function for entropies estimation
    fun_local = partial(
        estimate_entropy_pairs,
        df,
        precision,
        normalized,
        log_path,
        log_filename,
        verbose
    )

    # Argument list construction (loop of parallelization)
    df_symbols = (
        df[["symbol"]]
            .sort_values(["symbol"])
            .value_counts(sort = False)
            .reset_index()
            .rename(columns = {"count" : "bin_value"})
            .reset_index()
    )
    df_symbols["bin_value"] = df_symbols["bin_value"].apply(lambda x: int(np.sqrt(x)))
    df_symbols = pd.DataFrame(
        {
            "index_x" :np.repeat(df_symbols["index"], df_symbols.shape[0]),
            "index_y" :np.tile(df_symbols["index"], df_symbols.shape[0]),
            "symbol_x" : np.repeat(df_symbols["symbol"], df_symbols.shape[0]),
            "symbol_y" : np.tile(df_symbols["symbol"], df_symbols.shape[0]),
            "bins_x" : np.repeat(df_symbols["bin_value"], df_symbols.shape[0]),
            "bins_y" : np.tile(df_symbols["bin_value"], df_symbols.shape[0])
        }
    )
    
    mask = (
        (df_symbols["index_x"] <= df_symbols["index_y"]) &
        (df_symbols["bins_x"] >= min_bins) &
        (df_symbols["bins_y"] >= min_bins)
    )
    df_symbols = df_symbols[["symbol_x", "symbol_y", "bins_x", "bins_y"]][mask]
    entropy_args_list = df_symbols.values.tolist()
    
    # Parallel loop for entropies estimation
    df_entropy = mf.parallel_run(fun = fun_local, arg_list = entropy_args_list, tqdm_bar = tqdm_bar)
    df_entropy = pd.concat(df_entropy)
    
    return df_entropy
