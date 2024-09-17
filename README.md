# Market Efficiency and Asset Pricing Models: Insights from an Entropy-Based Approach
---

This paper investigates the link between market efficiency and the number of significant factors in asset pricing using an entropy-based approach, focusing on four developed stock markets and eight emerging stock markets from 2000 to 2024. Thus, it provides a deeper understanding of how market structure and information flow affect the number of significant factors influencing stock returns. By applying random matrix theory and a modified edge distribution test, the results show that developed markets are characterized by an eight-factor model for most of the period, with only one factor remaining when the market factor is removed. In contrast, emerging markets show only one significant factor throughout, even after excluding the market factor. It is noteworthy that the Greek market, although classified as emerging, behaves similarly to developed markets. These findings suggest that market efficiency is closely linked to the number of significant pricing factors, and entropy can serve as a powerful tool to capture these dynamics.

## File structure

The structure of the data repository consists of:

*   ***Input files:*** It corresponds to the input data sets for the study developed with the entropy matrix. There are two types of input files:
    *   **data_dictionary:** The data dictionary provides the tickers and stock names of a market so that it is downloaded from Yahoo Finance using a Python API known as ```yfinance```.
    *   **processed_data:** The processed data corresponds to the processing of different time series data where three types of time series have been calculated: returns, logarithmic returns (*log-returns*), and standard normalization of the logarithmic returns (*z-score log-returns*).

    It is important to mention that the data of this study does not contain raw data but only processed data. Indeed, only financial time series as stock markets extracted from YahooFinance are considered. In these time series, processed data is obtained as a data frame where the time series of returns, log-returns, and z-score log-returns are calculated for different stock markets. Nevertheless, the possibility of including other types of time series data is left for readers interested in this work.

*   ***Modules:*** It corresponds to the different modules developed in Python for the development of the study, namely:
    *   **estimate_entropy.py:** Module designed for the calculation of the entropy matrix on different time series and in the sliding time window. It is important to mention that the entropy matrix for the matrix of data on $p$ cross-sectional units observed over $n$ periods is constructed as the square symmetric matrix $\mathcal{E}\in\mathcal{M}(p,p;\mathbb{R})$ where its components are defined by the variance of information between two actions of the cross-sectional unit.

    *   **estimate_market_factors.py:** Module designed to calculate the number of significant components and the number of significant factors for the matrix of data on $p$ cross-sectional units observed over $n$ periods. It is important to mention that the number of components is estimated using Marchenko-Pastur law or Tracy-Widom test. Also, for the number of factors the Onatski R statistic, and Edge Distribution modified test are applied.

    *   **get_financial_time_series.py:** Module designed to process financial time series extracted from YahooFinance through the ```yfinance``` Python library and its ticker. Once the YahooFinance data is downloaded with this module, other functions estimate the returns with the daily closing prices, the logarithmic returns (*log-returns*), and standard normalization of the logarithmic returns (*z-score log-returns*). An example of the metadata output after this processing is mentioned in the following section (see [Metadata of the data sets](#metadata-of-the-data-sets)).

    *   **misc_functions.py:** Module designed to construct the Shannon entropy, Renyi entropy, Mutual Information, and Shared information distance for arbitrary vectors. Also, the function for the parallelization of processes is defined.

    *   **plot_market_factors.py:** Module designed to graph the temporal evolution of significant components and the number of significant factors for the matrix of data on $p$ cross-sectional units observed over $n$ periods. Additionally, other graphs are shown such as the evolution of the eigenvalues over time.

*   ***Logs (optional):*** It corresponds to an optional folder in which different log files are generated to know what is failing in any of the parallelized functions in the different modules of the data repository if any of these files suddenly stops working.

*   ***Output files:*** It corresponds to the folder with the output files after processing different data sets. For example, in this folder, the figures and tables for analysis will be by default. Some of these analyses are to show the estimation of the number of components using Marchenko-Pastur law and the number of factors using ED over a stock market, and the estimation of the temporal evolution of the eigenvalues of covariance and entropy matrices.

*   ***Scripts:*** It corresponds to different Jupyter notebooks where the study analyses were carried out and to emphasize some additional aspects, a section is dedicated to them later.

## Metadata of the data sets

The metadata of the different data sets that appear in this repository are organized by the ```.csv``` or ```.xlsx``` files placed in the input_files and output_files folders, namely:

|  **Folder** |      **Dataset**      | **Dataset type** |
|:-----------:|:---------------------:|:----------------:|
| input_files |   tickers_buxbd.csv   |     df_ticker    |
| input_files |   tickers_case30.csv  |     df_ticker    |
| input_files |    tickers_dji.csv    |     df_ticker    |
| input_files |    tickers_fchi.csv   |     df_ticker    |
| input_files |    tickers_gdat.csv   |     df_ticker    |
| input_files |   tickers_gdaxi.csv   |     df_ticker    |
| input_files |    tickers_gspc.csv   |     df_ticker    |
| input_files |    tickers_ibex.csv   |     df_ticker    |
| input_files | tickers_icolcapcl.csv |     df_ticker    |
| input_files |    tickers_ipsa.csv   |     df_ticker    |
| input_files |    tickers_jkse.csv   |     df_ticker    |
| input_files |    tickers_mxx.csv    |     df_ticker    |
| input_files |    tickers_nsei.csv   |     df_ticker    |
| input_files |      onatski.csv      |   df_quantiles   |
| input_files |    tracy_widom.csv    |   df_quantiles   |

The ```Dataset type``` column indicates the type of table that is available since some of these schemes are repeated between the different data sets considered (stock markets). Thus, the types of data sets available are:

*   ```df_ticker```: Corresponds to the data dictionary of multiple stock market time series (*fts*) data processed and hosted as processed data.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | ticker | string | Alphanumeric   character string | Alphanumeric code   that identifies a financial time series in YahooFinance |
    | ticker_name | string | Alphanumeric   character string | Name assigned to the   YahooFinance ticker |

*   ```df_quantiles```: Corresponds to the data set of quantiles used for the estimation of Tracy-Widom test and the Onatski R statistic.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | level | float | Positive integer   values | Level used to   estimate the Onatski R statistic |
    | z_score | float | Numeric values | z-score obtained   after centering and scaling of random variable used to obtain Tracy-Widom   distribution |
    | probability | float | Numeric values | Probability of   Tracy-Widom distribution |

*   ```df_fts```: Corresponds to the data set of multiple tickers in the same stock market(s) time series (*fts*) data processed and hosted as processed data.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | date | datetime | Dates in ISO 8601 format ("YYYY-mm-dd") | Date on which the data was extracted from YahooFinance |
    | symbol | string | Alphanumeric character string | Alphanumeric code that identifies a financial time series in YahooFinance |
    | Adj Close | float | Positive numeric values | Adjusted closing value of the price of a financial time series on a specific date |
    | return | float | Numeric values | Closing price returns of a financial time series between two consecutive dates |
    | log_return | float | Numeric values | Closing price logarithmic returns of a financial time series between two consecutive dates |
    | z_score_log_return | float | Numeric values | Normalization of the logarithmic returns of a financial time series between two consecutive dates |
    
*   ```df_rolling_window```: Corresponds to the number of components and factors in the covariance and entropy matrices of different financial time series components (shares for stock markets)

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | final_date | datetime | Dates in ISO 8601   format ("YYYY-mm-dd") | Date on which the   data was extracted from YahooFinance |
    | column_ | string | Alphanumeric   character string | String flag to   specify if the number of components and factors is estimated over log-returns   or residuals |
    | dropped_eigen_cov | int | Positive integer   values | Number of dropped   eigenvalues in the covariance matrix when Bouchaud filter is applied |
    | dropped_eigen_entropy | int | Positive integer   values | Number of dropped   eigenvalues in the entropy matrix when Bouchaud filter is applied |
    | alpha | float | Positive numeric   values | Level of statistical   significance for Tracy-Widom test |
    | n_components_cov | int | Positive integer   values | Number of significant   components for covariance matrix according to Tracy-Widom test |
    | n_components_cov_mp | int | Positive integer   values | Number of significant   components for covariance matrix according to Marchenko-Pastur law |
    | n_components_entropy | int | Positive integer   values | Number of significant   components for entropy matrix according to Tracy-Widom test |
    | n_components_entropy_mp | int | Positive integer   values | Number of significant   components for entropy matrix according to Marchenko-Pastur law |
    | level | int | Positive integer   values | Integer values for   Onatski test |
    | n_factors_cov | int | Positive integer   values | Number of market   factors for covariance matrix according to Onatski R statistic test |
    | n_factors_entropy | int | Positive integer   values | Number of market   factors for entropy matrix according to Onatski R statistic test |
    | edge_distribution_cov | int | Positive integer   values | Number of market   factors for covariance matrix according Edge Distribution test |
    | edge_distribution_entropy | int | Positive integer   values | Number of market   factors for entropy matrix according Edge Distribution test |
    | market_ticker | string | Alphanumeric   character string | Alphanumeric code   that identifies stock market in YahooFinance |
    | market_name | string | Alphanumeric   character string | Name of the stock   market in YahooFinance |
    | market_country | string | Alphanumeric   character string | Country of the stock   market |
    | market_type | string | Alphanumeric   character string | String associated to   developed or emerging market |

*   ```df_eigenvalues```: Corresponds to the eigenvalues for covariance and entropy matrices for different times and compares with Marchenko-Pastur law.

    | **Variable** | **Type** | **Allowed values** | **Description** |
    |:---:|:---:|:---:|:---:|
    | final_date | datetime | Dates in ISO 8601   format ("YYYY-mm-dd") | Date on which the   data was extracted from YahooFinance |
    | column_ | string | Alphanumeric   character string | String flag to   specify if the number of components and factors is estimated over log-returns   or residuals |
    | marchenko_pastur_lower_bound | float | Positive numeric   values | Lower bound estimated   with the Marchenko-Pastur law |
    | marchenko_pastur_upper_bound | float | Positive numeric   values | Upper bound estimated   with the Marchenko-Pastur law |
    | eigenvalues_id | int | Positive integer   values | ID of the eigenvalues   according to the increasing order |
    | eigenvalues_cov | float | Positive numeric   values | Eigenvalues of the   covariance matrix |
    | eigenvalues_entropy | float | Numeric values | Eigenvalues of the   entropy matrix |
    | market_ticker | string | Alphanumeric   character string | Alphanumeric code   that identifies stock market in YahooFinance |
    | market_name | string | Alphanumeric   character string | Name of the stock   market in YahooFinance |
    | market_country | string | Alphanumeric   character string | Country of the stock   market |
    | market_type | string | Alphanumeric   character string | String associated to   developed or emerging market |

## Scripts order

The set of codes developed for this data repository is divided into two parts specified below.

### Example of the estimation of entropy matrix

To see the use of the entropy matrix and its use in a stock market, we have the notebook ```estimate_covariance_matrix_entropy``` in such a way that it shows a data exploration for a developed market and it is observed how the eigenvalues ​​of the covariance and entropy matrix are differentiated.

### Rolling window analysis

To investigate the link between market efficiency and the number of significant factors in asset pricing using an entropy-based approach, a fixed-size moving window analysis is performed in which the spectrum (eigenvalues) of the covariance and entropy matrices are calculated and this is repeated every certain number of days. Finally, this information records the temporal evolution of the eigenvalues ​​of the two matrices, identifies the number of components and the number of significant factors at each time step. To do this, you have the following scripts:

1.  process_data_fts
2.  get_market_factors_rolling_window
3.  plot_market_factors_rolling_window

## Code/Software

All the information shown in this data repository is organized in the different folders mentioned and with all the files shown in the following public Github repository [[1]](#references).

To run the different notebooks in the ```scripts``` folder, it is recommended to use versios of ```requirements.txt``` file, namely:

- [X] matplotlib==3.8.4
- [X] numpy==1.26.4
- [X] pandas==2.2.1
- [X] scipy==1.13.0

## References

\[1] F. Abril, L. Molero. *Market_Efficiency*. Github repository. Available on: [https://github.com/fsabrilb/Market_Efficiency](https://github.com/fsabrilb/Market_Efficiency)
