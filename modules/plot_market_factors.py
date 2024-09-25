# -*- coding: utf-8 -*-
"""
Created on Thursday August 22 2024

@author: Laura Molero González
@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import re
import warnings
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.dates as mdates  # type: ignore
import matplotlib.ticker as mtick  # type: ignore

from datetime import date
from matplotlib import rcParams  # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)


# Plot evolution of number of factors and components ----
def plot_market_efficiency(
    df,
    width,
    height,
    fontsize_labels=13.5,
    fontsize_legend=11.5,
    n_cols=4,
    marker_size=4,
    n_x_breaks=10,
    n_y_breaks=10,
    fancy_legend=False,
    usetex=False,
    dpi=200,
    save_figures=True,
    output_path="../output_files",
    information_name="",
    input_generation_date="2024-04-22"
):
    """Plot number of components and factor for covariance and entropy matrices
    for different times and compares with Marchenko-Pastur law using Onatski
    test and Edge distirbution test

    Args:
    ---------------------------------------------------------------------------
    df : pandas DataFrame
        Dataframe with the covariances and entropies of different financial
        time series components (shares for stock indexes) with eight columns
        namely:
            - initial_date: Lower bound for temporal window
            - final_date: Upper bound for temporal window
            - column_: information taken for the calculation of the number of
            factors
            - dropped_eigen_cov: Dropped eigenvalues in covariance matrix after
            Bouchaud clipping filter
            - dropped_eigen_entropy: Dropped eigenvalues in entropy matrix
            after Bouchaud clipping filter
            - alpha: Level of statistical significance for Tracy-Widom test
            - n_components_cov: Number of significant components for covariance
            matrix according to Tracy-Widom test
            - n_components_cov_mp: Number of significant components for
            covariance matrix according to Marchenko-Pastur law
            - n_components_entropy: Number of significant components for
            entropy matrix according to Tracy-Widom test
            - n_components_entropy_mp: Number of significant components for
            entropy matrix according to Marchenko-Pastur law
            - level: Integer values for Onatski test
            - n_factors_cov: Number of market factors for covariance matrix
            according to Onatski test
            - n_factors_entropy: Number of market factors for entropy matrix
            according to Onatski test
            - edge_distribution_cov: Number of market factors for covariance
            matrix according Edge Distribution test
            - edge_distribution_entropy: Number of market factors for entropy
            matrix according Edge Distribution test
    width : int
        Width of final plot (default value 12)
    height : int
        Height of final plot (default value 28)
    fontsize_labels : float
        Font size in axis labels (default value 13.5)
    fontsize_legend : float
        Font size in legend (default value 11.5)
    n_cols : int
        Number of columns in legend related to plot (default value 4)
    marker_size : int
        Point size in market factors plot (default value 4)
    n_x_breaks : int
        Number of divisions in x-axis (default value 10)
    n_y_breaks : int
        Number of divisions in y-axis (default value 10)
    fancy_legend : bool
        Fancy legend output (default value False)
    usetex : bool
        Use LaTeX for renderized plots (default value False)
    dpi : int
        Dot per inch for output plot (default value 200)
    save_figures : bool
        Save figures flag (default value True)
    output_path : string
        Local path for outputs (default value is "../output_files")
    information_name : string
        Name of the output plot (default value "")
    input_generation_date : string
        Date of generation (control version) (default value "2024-08-21")

    Returns:
    ---------------------------------------------------------------------------
    No return for the function
    """

    # Initialize Plot data
    rcParams.update(
        {"font.family": "serif", "text.usetex": usetex, "pgf.rcfonts": False}
    )

    # Generation of plotted data and Plot 1: Covariance - Plot 2: Entropy
    levels_ = df["level"].unique()
    alphas_ = df["alpha"].unique()

    fig, ax = plt.subplots(len(levels_) * len(alphas_), 2)
    fig.set_size_inches(w=width, h=height)

    for i in range(len(levels_)):
        for j in range(len(alphas_)):
            k = j + i * len(alphas_)
            df_aux = df[((df["level"] == levels_[i]) & (df["alpha"] == alphas_[j]))]  # noqa: E501

            # Local data
            dates_j = df_aux["final_date"]
            n_cov = df_aux["n_components_cov"]
            f_cov = df_aux["n_factors_cov"]
            e_cov = df_aux["edge_distribution_cov"]
            n_entropy = df_aux["n_components_entropy"]
            f_entropy = df_aux["n_factors_entropy"]
            e_entropy = df_aux["edge_distribution_entropy"]
            time_labels = pd.date_range(start=dates_j.min(), end=dates_j.max(), periods=n_x_breaks).strftime("%Y-%m-%d")  # noqa: E501

            # Components
            ax[k, 0].plot(
                dates_j,
                n_cov,
                c="black",
                marker="o",
                ls="--",
                ms=marker_size,
                label=r"$n_{{CC}}^{{(c)}}(N_{{s}},\tau)$"
            )
            # ax[k, 1].plot(
            #     dates_j,
            #     n_entropy,
            #     c = "black",
            #     marker = "o",
            #     linestyle = "--",
            #     markersize = marker_size,
            #     label = r"$n_{CC}^{(e)}(N_{{s}},\tau)$"
            # )

            # Factors
            ax[k, 0].plot(
                dates_j,
                f_cov,
                c="firebrick",
                marker="D",
                ls="",
                ms=marker_size,
                label=r"$n_{{F}}^{{(c)}}(N_{{s}},\tau)$"
            )
            ax[k, 1].plot(
                dates_j,
                f_entropy,
                c="firebrick",
                marker="D",
                ls="",
                ms=marker_size,
                label=r"$n_{{F}}^{{(e)}}(N_{{s}},\tau)$"
            )

            # Edge Distribution
            ax[k, 0].plot(
                dates_j,
                e_cov,
                c="forestgreen",
                marker="v",
                ls="",
                ms=marker_size,
                label=r"$n_{{ED}}^{{(c)}}(N_{{s}},\tau)$"
            )
            ax[k, 1].plot(
                dates_j,
                e_entropy,
                c="forestgreen",
                marker="v",
                ls="",
                ms=marker_size,
                label=r"$n_{{ED}}^{{(e)}}(N_{{s}},\tau)$"
            )

            # Highlight regions with Factors > Components or Edge distribution > Components # noqa: E501
            ax[k, 0].fill_between(
                dates_j,
                0,
                f_cov,
                where=(n_cov < f_cov),
                alpha=0.14,
                facecolor="black",
                interpolate=True,
                label=r"$n_{{CC}}^{{(c)}}(N_{{s}},\tau)<n_{{F}}^{{(c)}}(N_{{s}},\tau)$"  # noqa: E501
            )
            ax[k, 0].fill_between(
                dates_j,
                0,
                e_cov,
                where=(n_cov < e_cov),
                alpha=0.14,
                facecolor="orange",
                interpolate=True,
                label=r"$n_{{CC}}^{{(c)}}(N_{{s}},\tau)<n_{{ED}}^{{(c)}}(N_{{s}},\tau)$"  # noqa: E501
            )
            ax[k, 1].fill_between(
                dates_j,
                0,
                f_entropy,
                where=(n_entropy < f_entropy),
                alpha=0.12,
                facecolor="black",
                interpolate=True,
                label=r"$n_{{CC}}^{{(e)}}(N_{{s}},\tau)<n_{{F}}^{{(e)}}(N_{{s}},\tau)$"  # noqa: E501
            )
            ax[k, 1].fill_between(
                dates_j,
                0,
                e_entropy,
                where=(n_entropy < e_entropy),
                alpha=0.15,
                facecolor="orange",
                interpolate=True,
                label=r"$n_{{CC}}^{{(e)}}(N_{{s}},\tau)<n_{{ED}}^{{(e)}}(N_{{s}},\tau)$"  # noqa: E501
            )

            # Axis formatter
            titles = ["Covariance", "Entropy"]
            y_max = [
                np.max([np.max(n_cov), np.max(f_cov), np.max(e_cov)]),
                np.max([np.max(n_entropy), np.max(f_entropy), np.max(e_entropy)])  # noqa: E501
            ]
            for p in [0, 1]:
                ax[k, p].tick_params(which="major", direction="in", top=True, right=True, labelsize=fontsize_labels, length=12)  # noqa: E501
                ax[k, p].tick_params(which="minor", direction="in", top=True, right=True, labelsize=fontsize_labels, length=6)  # noqa: E501
                ax[k, p].xaxis.set_major_locator(mtick.FixedLocator(time_labels))  # noqa: E501
                ax[k, p].xaxis.set_ticks(time_labels)
                ax[k, p].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # noqa: E501

                ax[k, p].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax[k, p].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))  # noqa: E501
                ax[k, p].set_xlabel("Date", fontsize=fontsize_labels)
                ax[k, p].set_ylabel("Number of components or factors", fontsize=fontsize_labels)  # noqa: E501
                ax[k, p].tick_params(axis="x", labelrotation=90)
                ax[k, p].set_xlim(date.fromisoformat(time_labels[0]), date.fromisoformat(time_labels[len(time_labels) - 1]))  # noqa: E501
                ax[k, p].set_ylim(0, y_max[p] + 1)
                ax[k, p].legend(
                    fancybox=fancy_legend,
                    shadow=False,
                    ncol=n_cols,
                    fontsize=fontsize_legend,
                    frameon=True
                )
                ax[k, p].set_title(
                    r"({}) {}, $L={}$, $\alpha={}\%$".format(chr(k + 65), titles[p], levels_[i], (1 - alphas_[j]) * 100),  # noqa: E501
                    loc="left",
                    y=1.005,
                    fontsize=fontsize_legend
                )

    # Save figures
    plt.show()
    fig.tight_layout()
    if save_figures:
        fig.savefig(
            "{}/{}_components_factors_{}.png".format(
                output_path,
                information_name,
                re.sub("-", "", input_generation_date)
            ),
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            transparent=False,
            pad_inches=0.03,
            dpi=dpi
        )
        plt.close()

    return 0


# Plot evolution of number of factors and components using only Entropy matrix and Edge distribution ----  # noqa: E501
def plot_market_efficiency_entropy(
    df_normalized,
    df_residuals,
    k_max,
    width_ratio,
    width,
    height,
    fontsize_labels=13.5,
    fontsize_legend=11.5,
    n_cols=4,
    marker_size=4,
    n_x_breaks=10,
    n_y_breaks=10,
    fancy_legend=False,
    usetex=False,
    dpi=200,
    save_figures=True,
    output_path="../output_files",
    information_name="",
    input_generation_date="2024-04-22"
):
    """Plot number of components and factor for entropy matrices for different
    times and compares with Marchenko-Pastur law using only Edge distribution

    Args:
    ---------------------------------------------------------------------------
    df_normalized : pandas DataFrame
        Dataframe with the covariances and entropies of different financial
        time series components (shares for stock indexes) with eight columns
        namely:
            - initial_date: Lower bound for temporal window
            - final_date: Upper bound for temporal window
            - column_: information taken for the calculation of the number of
            factors
            - dropped_eigen_cov: Dropped eigenvalues in covariance matrix after
            Bouchaud clipping filter
            - dropped_eigen_entropy: Dropped eigenvalues in entropy matrix
            after Bouchaud clipping filter
            - alpha: Level of statistical significance for Tracy-Widom test
            - n_components_cov: Number of significant components for covariance
            matrix according to Tracy-Widom test
            - n_components_cov_mp: Number of significant components for
            covariance matrix according to Marchenko-Pastur law
            - n_components_entropy: Number of significant components for
            entropy matrix according to Tracy-Widom test
            - n_components_entropy_mp: Number of significant components for
            entropy matrix according to Marchenko-Pastur law
            - level: Integer values for Onatski test
            - n_factors_cov: Number of market factors for covariance matrix
            according to Onatski test
            - n_factors_entropy: Number of market factors for entropy matrix
            according to Onatski test
            - edge_distribution_cov: Number of market factors for covariance
            matrix according Edge Distribution test
            - edge_distribution_entropy: Number of market factors for entropy
            matrix according Edge Distribution test
    df_residuals : pandas DataFrame
        Dataframe with the covariances and entropies of different financial
        time series components (shares for stock indexes) with residuals and
        same columns of df_normalized
    k_max : int
        Maximum number of factors considered in the Onatski test (default value
        is 8)
    width_ratio : int
        Aspect ratio between the evolution of Edge distribution and histogram
    width : int
        Width of final plot (default value 12)
    height : int
        Height of final plot (default value 28)
    fontsize_labels : float
        Font size in axis labels (default value 13.5)
    fontsize_legend : float
        Font size in legend (default value 11.5)
    n_cols : int
        Number of columns in legend related to plot (default value 4)
    marker_size : int
        Point size in market factors plot (default value 4)
    n_x_breaks : int
        Number of divisions in x-axis (default value 10)
    n_y_breaks : int
        Number of divisions in y-axis (default value 10)
    fancy_legend : bool
        Fancy legend output (default value False)
    usetex : bool
        Use LaTeX for renderized plots (default value False)
    dpi : int
        Dot per inch for output plot (default value 200)
    save_figures : bool
        Save figures flag (default value True)
    output_path : string
        Local path for outputs (default value is "../output_files")
    information_name : string
        Name of the output plot (default value "")
    input_generation_date : string
        Date of generation (control version) (default value "2024-08-21")

    Returns:
    ---------------------------------------------------------------------------
    No return for the function
    """

    # Initialize Plot data
    rcParams.update(
        {"font.family": "serif", "text.usetex": usetex, "pgf.rcfonts": False}
    )

    # Generation of plotted data and Plot 1: Covariance - Plot 2: Entropy
    cols = ["final_date", "n_components_entropy", "edge_distribution_entropy"]
    df_norm = df_normalized[cols].drop_duplicates()
    df_resi = df_residuals[cols].drop_duplicates()
    fig, ax = plt.subplots(1, 2, width_ratios=[width_ratio, 1])
    fig.set_size_inches(w=width, h=height)

    # Local data
    dates_norm = df_norm["final_date"]
    n_entropy_norm = df_norm["n_components_entropy"]
    e_entropy_norm = df_norm["edge_distribution_entropy"]

    dates_resi = df_resi["final_date"]
    n_entropy_resi = df_resi["n_components_entropy"]
    e_entropy_resi = df_resi["edge_distribution_entropy"]

    time_labels = pd.date_range(
        start=dates_norm.min(),
        end=dates_norm.max(),
        periods=n_x_breaks
    ).strftime("%Y-%m-%d")

    # Plot 1 - Edge Distribution for normalized returns and residuals
    ax[0].plot(
        dates_norm,
        e_entropy_norm,
        c="forestgreen",
        marker="v",
        linestyle="",
        markersize=marker_size,
        label=r"$n_{{ED}}^{{(e)}}(N_{s},\tau)$"
    )
    ax[0].plot(
        dates_resi,
        e_entropy_resi,
        c="firebrick",
        marker="^",
        linestyle="",
        markersize=marker_size,
        label=r"$\rho_{{ED}}^{{(e)}}(N_{s},\tau)$"
    )

    # Plot 1 - Highlight regions with Edge distribution normalized = residuals
    try:
        ax[0].fill_between(
            dates_resi,
            0,
            e_entropy_resi,
            where=(e_entropy_norm == e_entropy_resi),
            alpha=0.15,
            facecolor="black",
            interpolate=True,
            label=r"$n_{{ED}}^{{(e)}}(N_{{s}},\tau)=\rho_{{ED}}^{{(e)}}(N_{{s}},\tau)$"  # noqa: E501
        )
    except Exception:
        pass

    # Plot 2 - Histogram of Edge distirbution
    weights_norm = np.ones_like(e_entropy_norm) / float(len(e_entropy_norm))
    weights_resi = np.ones_like(e_entropy_resi) / float(len(e_entropy_resi))

    ax[1].hist(
        e_entropy_norm,
        bins=k_max,
        alpha=0.12,
        facecolor="green",
        edgecolor="forestgreen",
        weights=weights_norm,
        histtype="stepfilled",
        cumulative=False,
        label="Normalized returns",
        linewidth=4
    )
    ax[1].hist(
        e_entropy_resi,
        bins=k_max,
        alpha=0.12,
        facecolor="red",
        edgecolor="firebrick",
        weights=weights_resi,
        histtype="stepfilled",
        cumulative=False,
        label="Residuals",
        linewidth=4
    )

    # Axis formatter
    y_max = np.max([
        np.max(n_entropy_norm),
        np.max(e_entropy_norm),
        np.max(n_entropy_resi),
        np.max(e_entropy_resi)
    ])

    labels_x = ["Date", "Number of factors"]
    labels_y = ["Number of factors", "Percentage"]
    titles = ["Number of factors", "Percentage"]
    for p in [0, 1]:
        ax[p].tick_params(which="major", direction="in", top=True, right=True, labelsize=fontsize_labels, length=12)  # noqa: E501
        ax[p].tick_params(which="minor", direction="in", top=True, right=True, labelsize=fontsize_labels, length=6)  # noqa: E501
        ax[p].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
        ax[p].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
        ax[p].set_xlabel(labels_x[p], fontsize=fontsize_labels)
        ax[p].set_ylabel(labels_y[p], fontsize=fontsize_labels)
        ax[p].tick_params(axis="x", labelrotation=90)
        ax[p].set_title(
            r"({}) {}".format(chr(p + 65), titles[p]),
            loc="left",
            y=1.005,
            fontsize=fontsize_legend
        )

    ax[0].legend(
        fancybox=fancy_legend,
        shadow=False,
        ncol=n_cols,
        fontsize=fontsize_legend,
        frameon=True
    )
    ax[1].legend(
        fancybox=fancy_legend,
        shadow=False,
        ncol=n_cols,
        fontsize=fontsize_legend,
        frameon=True,
        bbox_to_anchor=(0.5, -0.05)
    )

    ax[0].xaxis.set_major_locator(mtick.FixedLocator(time_labels))
    ax[0].xaxis.set_ticks(time_labels)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax[1].xaxis.set_major_locator(mtick.FixedLocator(np.arange(0, k_max + 1, 1)))  # noqa: E501
    ax[1].xaxis.set_ticks(np.arange(0, k_max + 1, 1))
    # ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    ax[0].set_xlim(date.fromisoformat(time_labels[0]), date.fromisoformat(time_labels[len(time_labels) - 1]))  # noqa: E501
    ax[0].set_ylim(0, y_max + 1)
    ax[1].set_xlim(0, y_max + 1)

    # Save figures
    plt.show()
    fig.tight_layout()
    if save_figures:
        fig.savefig(
            "{}/{}_edge_distribution_{}.png".format(
                output_path,
                information_name,
                re.sub("-", "", input_generation_date)
            ),
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            transparent=False,
            pad_inches=0.03,
            dpi=dpi
        )
        plt.close()

    return 0
