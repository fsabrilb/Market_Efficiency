# -*- coding: utf-8 -*-
"""
Created on Thursday August 22 2024

@author: Laura Molero González
@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import re
import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.dates as mdates # type: ignore
import matplotlib.ticker as mtick # type: ignore

from datetime import date
from matplotlib import rcParams # type: ignore

# Global options ----
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)

# Plot eigenvalues evolution for covariance and entropy matrix ----
def plot_eigen_evolution(
    df_eigenvalues,
    width,
    height,
    fontsize_labels=13.5,
    fontsize_legend=11.5,
    n_cols_1=4,
    n_cols_2=4,
    n_x_breaks=10,
    fancy_legend=False,
    x_legend=1,
    y_legend=1,
    usetex=False,
    dpi=200,
    save_figures=True,
    output_path="../output_files",
    information_name="",
    input_generation_date="2024-04-22"
):
    """Plot eigenvalues for covariance and entropy matrices for different times
    and compares with Marchenko-Pastur law

    Args:
    ---------------------------------------------------------------------------
    df_eigenvalues : pandas DataFrame
        DataFrame with the eigenvalues evolution estimated for covariance and
        entropy matrices with the following columns:
            - initial_date: Lower bound for temporal window
            - final_date: Upper bound for temporal window
            - column_: information taken for the calculation of the number of
            factors
            - marchenko_pastur_upper_bound: Upper bound estimated with the
            Marchenko-Pastur law
            - eigenvalues_id: ID of the eigenvalues according to the increasing
            order
            - eigenvalues_cov: Eigenvalues of the covariance matrix
            - eigenvalues_entropy: Eigenvalues of the entropy matrix
    width : int
        Width of final plot (default value 12)
    height : int
        Height of final plot (default value 28)
    fontsize_labels : float
        Font size in axis labels (default value 13.5)
    fontsize_legend : float
        Font size in legend (default value 11.5)
    n_cols_1 : int
        Number of columns in legend related to plot 1 (default value 4)
    n_cols_2 : int
        Number of columns in legend related to plot 2 (default value 4)
    n_x_breaks : int
        Number of divisions in x-axis (default value 10)
    fancy_legend : bool
        Fancy legend output (default value False)
    x_legend : float
        X position of graph legend (default value 1)
    y_legend : float
        Y position of graph legend (default value 1)
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
    rcParams.update({"font.family": "serif", "text.usetex": usetex, "pgf.rcfonts": False})

    # Generation of plotted data
    df_cov = df_eigenvalues[df_eigenvalues["eigenvalues_id"] <= df_eigenvalues["n_components_cov_mp"]]
    dates_cov = pd.to_datetime(df_cov["final_date"].unique(), errors = "coerce")
    times_cov = pd.date_range(start = dates_cov.min(), end = dates_cov.max(), periods = n_x_breaks).strftime("%Y-%m-%d")
    dates_entropy = pd.to_datetime(df_eigenvalues["final_date"].unique(), errors = "coerce")
    times_entropy = pd.date_range(start = dates_entropy.min(), end = dates_entropy.max(), periods = n_x_breaks).strftime("%Y-%m-%d")
    
    # Plot 1 - Evolution of eigenvalues of Covariance Matrix (EEC)
    fig_1, ax_1 = plt.subplots(1, 1)
    fig_1.set_size_inches(w = width, h = height)
    markers = {0 : "o", 1 : "v", 2 : "*", 3 : "s", 4 : "p"}

    num_colors_cov = df_cov["eigenvalues_id"].max()
    colors_cov = np.linspace(0, 1, num_colors_cov)
    cmap_cov = plt.get_cmap("hsv", num_colors_cov)
    cmap_cov.set_under("black")

    # Plot 1 - EEC - Real Components (Over Marchenko-Pastur law)
    for id in sorted(df_cov["eigenvalues_id"].unique()):
        ax_1.plot(
            df_cov[df_cov["eigenvalues_id"] == id]["final_date"],
            df_cov[df_cov["eigenvalues_id"] == id]["eigenvalues_cov"],
            c = cmap_cov(colors_cov[id-1]),
            marker = markers[id % 5],
            linestyle = "dashed",
            linewidth = 0.5,
            markersize = 2,
            label = r"$\lambda_{{{}}}^{{(c)}}(t)$".format(id)
        )

    # Plot 1 - EEC - Random Components (Under Marchenko-Pastur law)
    for id in range(num_colors_cov + 1, df_eigenvalues["eigenvalues_id"].max() + 1, 1):
        ax_1.plot(
            df_eigenvalues[df_eigenvalues["eigenvalues_id"] == id]["final_date"],
            df_eigenvalues[df_eigenvalues["eigenvalues_id"] == id]["eigenvalues_cov"],
            c = "black",
            marker = "o",
            linestyle = "",
            markersize = 1.4
        )

    # Plot 1 - EEC - Eigenvalues regions bounds (minimum, maximum and Marchenko-Pastur law)
    x_ = df_cov["final_date"]
    y_min = df_cov["eigenvalues_cov_min"]
    y_mid = df_cov["marchenko_pastur_upper_bound"]
    y_max = df_cov["eigenvalues_cov_max"]

    ax_1.plot(x_, y_mid, c = "black", marker = "", linestyle = "--", linewidth = 1, label = r"$\lambda_{+}$")
    ax_1.fill_between(
        x_,
        y_mid,
        y_max,
        #where = ((y_max >= y_mid) & (y_mid >= y_min)),
        alpha = 0.05,
        facecolor = "black",
        interpolate = True,
        label = r"$[\lambda_{+},\lambda_{max}]$"
    )
    ax_1.fill_between(
        x_,
        y_min,
        y_mid,
        #where = ((y_max >= y_mid) & (y_mid >= y_min)),
        alpha = 0.12,
        facecolor = "red",
        interpolate = True,
        label = r"$[\lambda_{min},\lambda_{+}]$"
    )

    # Plot 1 - EEC - Axis formatter
    ax_1.tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
    ax_1.tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
    ax_1.xaxis.set_major_locator(mtick.FixedLocator(times_cov))
    ax_1.xaxis.set_ticks(times_cov)
    ax_1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_1.set_yscale("log", subs = [2, 3, 4, 5, 6, 7, 8, 9])
    ax_1.set_xlabel("Date", fontsize = fontsize_labels)        
    ax_1.set_ylabel(r"Covariance eigenvalues $\lambda_{k}^{(c)}(t)$", fontsize = fontsize_labels)
    ax_1.tick_params(axis = "x", labelrotation = 90)
    ax_1.set_xlim(date.fromisoformat(times_cov[0]), date.fromisoformat(times_cov[len(times_cov) - 1]))
    ax_1.legend(
        fancybox = fancy_legend,
        shadow = False,
        ncol = n_cols_1,
        fontsize = fontsize_legend,
        bbox_to_anchor = (x_legend, y_legend)
    )
    #ax_1.set_title(r"({}) Covariance".format(chr(65)), loc = "left", y = 1.005, fontsize = 12)

    # Plot 2 - Evolution of eigenvalues of Entropy Matrix (EEE)
    fig_2, ax_2 = plt.subplots(1, 1)
    fig_2.set_size_inches(w = width, h = height)

    num_colors_entropy = df_eigenvalues["eigenvalues_id"].max()
    colors_entropy = np.linspace(0, 1, num_colors_entropy)    
    cmap_entropy = plt.get_cmap("hsv", num_colors_entropy)

    # Plot 2 - EEE - Components
    for id in sorted(df_eigenvalues["eigenvalues_id"].unique()):
        ax_2.plot(
            df_eigenvalues[df_eigenvalues["eigenvalues_id"] == id]["final_date"],
            df_eigenvalues[df_eigenvalues["eigenvalues_id"] == id]["eigenvalues_entropy"],
            c = cmap_entropy(colors_entropy[id-1]),
            marker = markers[id % 5],
            linestyle = "dashed",
            linewidth = 0.5,
            markersize = 2,
            label = r"$\lambda_{{{}}}^{{(e)}}(t)$".format(id)
        )

    # Plot 2 - EEE - Eigenvalues regions bounds (minimum, maximum and Marchenko-Pastur law)
    x_ = df_eigenvalues["final_date"]
    y_min = df_eigenvalues["eigenvalues_entropy_min"]
    y_mid = df_eigenvalues["marchenko_pastur_upper_bound"]
    y_max = df_eigenvalues["eigenvalues_entropy_max"]

    ax_2.plot(x_, y_mid, c = "black", marker = "", linestyle = "-", linewidth = 1, label = r"$\lambda_{+}$")
    ax_2.fill_between(
        x_,
        y_mid,
        y_max,
        #where = ((y_max >= y_mid) & (y_mid >= y_min)),
        alpha = 0.05,
        facecolor = "black",
        interpolate = True,
        label = r"$[\lambda_{+},\lambda_{max}]$"
    )
    ax_2.fill_between(
        x_,
        y_min,
        y_mid,
        #where = ((y_max >= y_mid) & (y_mid >= y_min)),
        alpha = 0.12,
        facecolor = "red",
        interpolate = True,
        label = r"$[\lambda_{min},\lambda_{+}]$"
    )

    # Plot 2 - EEE - Axis formatter
    linear_threshold = np.power(10, int(np.log10(np.abs(np.min(y_min)))))
    ax_2.tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
    ax_2.tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
    ax_2.xaxis.set_major_locator(mtick.FixedLocator(times_entropy))
    ax_2.xaxis.set_ticks(times_entropy)
    ax_2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_2.set_yscale("symlog", linthresh = linear_threshold, subs = [2, 3, 4, 5, 6, 7, 8, 9])
    ax_2.set_xlabel("Date", fontsize = fontsize_labels)        
    ax_2.set_ylabel(r"Entropy eigenvalues $\lambda_{k}^{(e)}(t)$", fontsize = fontsize_labels)
    ax_2.tick_params(axis = "x", labelrotation = 90)
    ax_2.set_xlim(date.fromisoformat(times_entropy[0]), date.fromisoformat(times_entropy[len(times_entropy) - 1]))
    ax_2.legend(
        fancybox = fancy_legend,
        shadow = False,
        ncol = n_cols_2,
        fontsize = fontsize_legend,
        bbox_to_anchor = (x_legend, y_legend)
    )
    #ax_2.set_title(r"({}) Entropy".format(chr(66)), loc = "left", y = 1.005, fontsize = 12)

    # Save figures
    plt.show()
    fig_1.tight_layout()
    if save_figures:
        fig_1.savefig(
            "{}/{}_covariance_{}.png".format(
                output_path,
                information_name,
                re.sub("-", "", input_generation_date)
            ),
            bbox_inches = "tight",
            facecolor = fig_1.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        plt.close()

    plt.show()
    fig_2.tight_layout()
    if save_figures:
        fig_2.savefig(
            "{}/{}_entropy_{}.png".format(
                output_path,
                information_name,
                re.sub("-", "", input_generation_date)
            ),
            bbox_inches = "tight",
            facecolor = fig_2.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        plt.close()

    return 0

# Plot evolution of number of factors and components ----
# Plot number of factors and components ----
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
    for different times and compares with Marchenko-Pastur law

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
            - dropped_eigen_entropy: Dropped eigenvalues in entropy matrix after
            Bouchaud clipping filter
            - alpha: Level of statistical significance for Tracy-Widom test
            - n_components_cov: Number of significant components for covariance
            matrix according to Tracy-Widom test
            - n_components_cov_mp: Number of significant components for
            covariance matrix according to Marchenko-Pastur law
            - n_components_entropy: Number of significant components for entropy
            matrix according to Tracy-Widom test
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
    rcParams.update({"font.family": "serif", "text.usetex": usetex, "pgf.rcfonts": False})

    # Generation of plotted data and Plot 1: Covariance - Plot 2: Entropy
    levels_ = df["level"].unique()
    alphas_ = df["alpha"].unique()

    fig,ax = plt.subplots(len(levels_) * len(alphas_), 2)
    fig.set_size_inches(w = width, h = height)

    for i in range(len(levels_)):
        for j in range(len(alphas_)):
            k = j + i * len(alphas_)
            df_aux = df[((df["level"] == levels_[i]) & (df["alpha"] == alphas_[j]))]

            # Local data
            dates_j = df_aux["final_date"]
            n_cov = df_aux["n_components_cov"]
            f_cov = df_aux["n_factors_cov"]
            e_cov = df_aux["edge_distribution_cov"]
            n_entropy = df_aux["n_components_entropy"]
            f_entropy = df_aux["n_factors_entropy"]
            e_entropy = df_aux["edge_distribution_entropy"]
            time_labels = pd.date_range(start = dates_j.min(), end = dates_j.max(), periods = n_x_breaks).strftime("%Y-%m-%d")

            # Components
            ax[k, 0].plot(
                dates_j,
                n_cov,
                c = "black",
                marker = "o",
                linestyle = "--",
                markersize = marker_size,
                label = r"$n_{CC}^{(c)}(t)$"
            )
            #ax[k, 1].plot(
            #    dates_j,
            #    n_entropy,
            #    c = "black",
            #    marker = "o",
            #    linestyle = "--",
            #    markersize = marker_size,
            #    label = r"$n_{CC}^{(e)}(t)$"
            #)
            
            # Factors
            ax[k, 0].plot(
                dates_j,
                f_cov,
                c = "firebrick",
                marker = "D",
                linestyle = "",
                markersize = marker_size,
                label = r"$n_{F}^{(c)}(t)$"
            )
            ax[k, 1].plot(
                dates_j,
                f_entropy,
                c = "firebrick",
                marker = "D",
                linestyle = "",
                markersize = marker_size,
                label = r"$n_{F}^{(e)}(t)$"
            )

            # Edge Distribution
            ax[k, 0].plot(
                dates_j,
                e_cov,
                c = "forestgreen",
                marker = "v",
                linestyle = "",
                markersize = marker_size,
                label = r"$n_{ED}^{(c)}(t)$"
            )
            ax[k, 1].plot(
                dates_j,
                e_entropy,
                c = "forestgreen",
                marker = "v",
                linestyle = "",
                markersize = marker_size,
                label = r"$n_{ED}^{(e)}(t)$"
            )

            # Highlight regions with Factors > Components or Edge distribution > Components
            ax[k, 0].fill_between(
                dates_j,
                0,
                f_cov,
                where = (n_cov < f_cov),
                alpha = 0.14,
                facecolor = "black",
                interpolate = True,
                label = r"$n_{CC}^{(c)}(t)<n_{F}^{(c)}(t)$"
            )
            ax[k, 0].fill_between(
                dates_j,
                0,
                e_cov,
                where = (n_cov < e_cov),
                alpha = 0.14,
                facecolor = "orange",
                interpolate = True,
                label = r"$n_{CC}^{(c)}(t)<n_{ED}^{(c)}(t)$"
            )
            ax[k, 1].fill_between(
                dates_j,
                0,
                f_entropy,
                where = (n_entropy < f_entropy),
                alpha = 0.12,
                facecolor = "black",
                interpolate = True,
                label = r"$n_{CC}^{(e)}(t)<n_{F}^{(e)}(t)$"
            )
            ax[k, 1].fill_between(
                dates_j,
                0,
                e_entropy,
                where = (n_entropy < e_entropy),
                alpha = 0.15,
                facecolor = "orange",
                interpolate = True,
                label = r"$n_{CC}^{(e)}(t)<n_{ED}^{(e)}(t)$"
            )

            # Axis formatter
            titles = ["Covariance", "Entropy"]
            y_max = [
                np.max([np.max(n_cov), np.max(f_cov), np.max(e_cov)]),
                np.max([np.max(n_entropy), np.max(f_entropy), np.max(e_entropy)])
            ]
            for p in [0, 1]:
                ax[k, p].tick_params(which = "major", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 12)
                ax[k, p].tick_params(which = "minor", direction = "in", top = True, right = True, labelsize = fontsize_labels, length = 6)
                ax[k, p].xaxis.set_major_locator(mtick.FixedLocator(time_labels))
                ax[k, p].xaxis.set_ticks(time_labels)
                ax[k, p].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

                ax[k, p].yaxis.set_major_locator(mtick.MaxNLocator(n_y_breaks))
                ax[k, p].yaxis.set_minor_locator(mtick.MaxNLocator(5 * n_y_breaks))
                ax[k, p].set_xlabel("Date", fontsize = fontsize_labels)        
                ax[k, p].set_ylabel("Number of components or factors", fontsize = fontsize_labels)
                ax[k, p].tick_params(axis = "x", labelrotation = 90)
                ax[k, p].set_xlim(date.fromisoformat(time_labels[0]), date.fromisoformat(time_labels[len(time_labels) - 1]))
                ax[k, p].set_ylim(0, y_max[p] + 1)
                ax[k, p].legend(
                    fancybox = fancy_legend,
                    shadow = False,
                    ncol = n_cols,
                    fontsize = fontsize_legend,
                    frameon = True
                )
                ax[k, p].set_title(
                    r"({}) {}, $L={}$, $\alpha={}\%$".format(chr(k + 65), titles[p], levels_[i], (1 - alphas_[j]) * 100),
                    loc = "left",
                    y = 1.005,
                    fontsize = fontsize_legend
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
            bbox_inches = "tight",
            facecolor = fig.get_facecolor(),
            transparent = False,
            pad_inches = 0.03,
            dpi = dpi
        )
        plt.close()

    return 0
