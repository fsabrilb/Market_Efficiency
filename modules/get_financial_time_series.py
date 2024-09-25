# -*- coding: utf-8 -*-
"""
Created on Thursday June 14 2024

@author: Laura Molero González
@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Load of financial time series ----
def load_financial_time_series(
    ticker_dict,
    initial_date="1900-01-01",
    final_date="2023-01-01",
    interval="1d"
):
    """Download and process multiple data from Yahoo finance:

    Args:
    ---------------------------------------------------------------------------
    ticker_dict : dict
        Dictionary of Yahoo finance tickers (items) and his name (values) for
        download
    initial_date: str
        Initial date for time series in ISO format (YYYY-MM-DD)
    final_date : str
        Final date for time series in ISO format (YYYY-MM-DD)
    interval : str
        Frequency between the reported data

    Returns:
    ---------------------------------------------------------------------------
    df_fts : pandas DataFrame
        Dataframe with downloaded financial time series
    """

    # Load and process information ----
    df_financial_time_series = []
    for ticker, ticker_name in ticker_dict.items():
        # Download data ----
        df_local = yf.download(
            tickers=ticker,
            start=initial_date,
            end=final_date,
            interval=interval,
            progress=False
        )

        if df_local.shape[0] > 0:
            df_local["symbol"] = ticker
            df_local["ticker_name"] = ticker_name

            # Generate date column and sort by symbol, ticker_name and date
            df_local["date"] = df_local.index
            df_local = df_local.sort_values(by=["symbol", "ticker_name", "date"])  # noqa: E501

            # Generate index column ----
            df_local["step"] = np.arange(df_local.shape[0]) - 1

            # Relocate date, symbol, ticker_name and step column ----
            df_local.insert(0, "date", df_local.pop("date"))
            df_local.insert(1, "symbol", df_local.pop("symbol"))
            df_local.insert(2, "ticker_name", df_local.pop("ticker_name"))
            df_local.insert(3, "step", df_local.pop("step"))

            # Estimate return with close price and profile time series ----
            old_count = df_local.shape[0]
            df_local["return"] = df_local["Adj Close"].diff(periods=1)
            df_local = df_local[(df_local["return"].notnull() & df_local["return"] != 0)]  # noqa: E501
            print("- Download {} with initial {} rows and {} rows after profiling".format(ticker, old_count, df_local.shape[0]))  # noqa: E501

            # Estimate log-return with close price ----
            df_local["log_return"] = np.log(df_local["Close"])
            df_local["log_return"] = df_local["log_return"].diff(periods=1)
            df_local["z_score_log_return"] = df_local[["log_return"]].apply(lambda x: (x - np.mean(x)) / np.std(x))  # noqa: E501

            # Replace NaN with zeros ----
            df_local["log_return"] = df_local["log_return"].fillna(0)

            # Final merge of data
            df_financial_time_series.append(df_local.fillna(0))
            print("- Processed {} : {}".format(ticker, ticker_name))

    df_financial_time_series = pd.concat(df_financial_time_series)

    return df_financial_time_series


# Covariance matrix estimation ----
def estimate_covariance_stock_index(df, column_="z_score_log_return"):
    """Estimate covariance matrix of financial time series:

    Args:
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe of financial time series with the following columns:
            - Adjusted closed value ("Adj Close")
            - Logarithmic return ("log_return")
            - Normalized logarithmic return through z-score transformation
            ("z_score_log_return")
            - No market components or residuals if Sharpe model is executed
            (see estimate_market_factors module, estimate_sharpe_model func)
    column_ : str
        Column of financial time series dataframe used to estimate covariance
        matrix (default value "z_score_log_return")

    Returns:
    ---------------------------------------------------------------------------
    cov_matrix : pandas DataFrame
        Dataframe with the covariances of different financial time series
        components (shares for stock indexes)
    """

    cov_matrix = (
        df[["date", "symbol", column_]]
        .pivot(index="date", columns="symbol", values=column_)
        .cov(min_periods=None, ddof=1, numeric_only=False)
        .reset_index(drop=True)
    )

    return cov_matrix


# Deployment for multiple markets ----
def process_financial_time_series(
    ticker_dict,
    initial_date="1900-01-01",
    final_date="2024-01-01",
    interval="1d"
):
    """Download and process multiple data from Yahoo finance:

    Args:
    ---------------------------------------------------------------------------
    ticker_dict : dict
        Dictionary of Yahoo finance tickers (items) and his name (values) for
        download
    initial_date: str
        Initial date for time series in ISO format (YYYY-MM-DD)
    final_date : str
        Final date for time series in ISO format (YYYY-MM-DD)
    interval : str
        Frequency between the reported data

    Returns:
    ---------------------------------------------------------------------------
    df_market : pandas DataFrame
        Dataframe with downloaded financial time series after process columns
    """

    # Download Yahoo Finance data
    print("----------------------------------- YAHOO FINANCE DATA -----------------------------------\n")  # noqa: E501
    df_market = load_financial_time_series(
        ticker_dict=ticker_dict,
        initial_date=initial_date,
        final_date=final_date,
        interval=interval
    )

    # Drop unnecessarily data
    dropped_columns = ["ticker_name", "step", "Open", "High", "Low", "Close", "Volume"]  # noqa: E501
    df_market = df_market.drop(columns=dropped_columns).reset_index().drop(columns=["Date"])  # noqa: E501

    return df_market
