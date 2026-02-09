from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distributions(
        df,
        group_name,
        value_name="value",
        title="Group Box Plot",
        y_lim=None,
        y_line=None,
        figsize=(14, 8),
):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Boxplot
    sns.boxplot(
        data=df,
        x=group_name,
        y=value_name,
        ax=axes[0]
    )
    axes[0].set_ylabel("Value")

    # Violinplot
    sns.violinplot(
        data=df,
        x=group_name,
        y=value_name,
        ax=axes[1],
        inner="box",
    )
    axes[1].set_ylabel("Value")

    fig.suptitle(title, y=0.98)

    if y_line is not None:
        for ax in axes:
            ax.axhline(y_line, color="black", linestyle="--", linewidth=1)

    if y_lim is not None:
        axes[0].set_ylim(y_lim)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_distributions_nested(
        df,
        group_name,
        dataset_name="Dataset",
        value_name="value",
        title="Group Box Plot",
        y_lim=None,
        y_line=None,
        figsize=(14, 8),
):
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    sns.boxplot(
        data=df,
        x=group_name,
        y=value_name,
        hue=dataset_name,
        ax=axes[0]
    )
    axes[0].set_ylabel("Value")

    sns.violinplot(
        data=df,
        x=group_name,
        y=value_name,
        hue=dataset_name,
        ax=axes[1],
        inner="box",
    )
    axes[1].set_ylabel("Value")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()

    fig.suptitle(title, y=0.98)

    fig.legend(
        handles,
        labels,
        title=dataset_name,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(labels),
        frameon=False
    )

    if y_line is not None:
        for ax in axes:
            ax.axhline(y_line, color="black", linestyle="--", linewidth=1)

    if y_lim is not None:
        axes[0].set_ylim(y_lim)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram_with_density(
        df,
        group_col,
        value_col,
        title="Histogram with Density",
        xlabel="Value",
        ylabel="Density",
        figsize=(10, 6)
):
    """
    Plots a single histogram with density (KDE) for all groups in the DataFrame.

    Parameters:
    - df: DataFrame in long format (columns: group_col, value_col).
    - group_col: Name of the column containing group labels.
    - value_col: Name of the column containing values.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - figsize: Figure size.
    """
    plt.figure(figsize=figsize)

    sns.histplot(
        data=df,
        x=value_col,
        hue=group_col,
        kde=True,
        element="step",
        stat="density",
        common_norm=False
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_histogram_with_density_by_subgroup(
        df,
        group_col,
        value_col,
        subgroup_col,
        title="Histogram with Density by Subgroup",
        xlabel="Value",
        ylabel="Density",
        figsize=(14, 10)
):
    """
    Plots 4 subplots of histograms with density (KDE) for each group in the DataFrame,
    split by the values of an additional subgroup variable. All subplots share the same x and y scales.

    Parameters:
    - df: DataFrame in long format (columns: group_col, value_col, subgroup_col).
    - group_col: Name of the column containing group labels.
    - value_col: Name of the column containing values.
    - subgroup_col: Name of the column containing subgroup labels.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - figsize: Figure size.
    """
    # Get unique subgroups
    subgroups = df[subgroup_col].unique()

    # Create subplots with shared x and y axes
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easy iteration

    # Plot for each subgroup
    for i, subgroup in enumerate(subgroups[:4]):  # Limit to 4 subplots
        ax = axes[i]
        subgroup_df = df[df[subgroup_col] == subgroup]

        sns.histplot(
            data=subgroup_df,
            x=value_col,
            hue=group_col,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax
        )

        ax.set_title(f"{subgroup_col}: {subgroup}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram_with_density_for_dataframes(
        dfs,
        group_col,
        value_col,
        titles,
        fig_title="Histograms with Density by DataFrame",  # Unique figure title
        xlabel="Value",
        ylabel="Density",
        figsize=(14, 10)
):
    """
    Plots histograms with density (KDE) for each DataFrame in the list.
    Each DataFrame is plotted in a separate subplot, with shared x and y scales.

    Parameters:
    - dfs: List of DataFrames (each with columns: group_col, value_col).
    - group_col: Name of the column containing group labels.
    - value_col: Name of the column containing values.
    - titles: List of titles for each subplot (must match the number of DataFrames).
    - fig_title: Title for the entire figure.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - figsize: Figure size.
    """
    assert len(dfs) == len(titles), "Number of DataFrames and titles must match."

    n_dfs = len(dfs)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easy iteration

    for i, (df, title) in enumerate(zip(dfs[:4], titles[:4])):  # Limit to 4 subplots
        ax = axes[i]
        sns.histplot(
            data=df,
            x=value_col,
            hue=group_col,
            kde=True,
            element="step",
            stat="density",
            common_norm=False,
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(fig_title, y=1.02)  # Add the unique figure title
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

def plot_violin_for_dataframes(
        dfs,
        group_col,
        value_col,
        titles,
        xlabel="Value",
        ylabel="Density",
        figsize=(14, 10)
):
    """
    Plots violin plots for each DataFrame in the list.
    Each DataFrame is plotted in a separate subplot, with shared x and y scales.

    Parameters:
    - dfs: List of DataFrames (each with columns: group_col, value_col).
    - group_col: Name of the column containing group labels.
    - value_col: Name of the column containing values.
    - titles: List of titles for each subplot (must match the number of DataFrames).
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - figsize: Figure size.
    """
    assert len(dfs) == len(titles), "Number of DataFrames and titles must match."

    n_dfs = len(dfs)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten for easy iteration

    for i, (df, title) in enumerate(zip(dfs[:4], titles[:4])):  # Limit to 4 subplots
        ax = axes[i]
        sns.violinplot(
            data=df,
            x=group_col,
            y=value_col,
            ax=ax,
            inner="quartile",
            cut=0  # Extend the density plot to the extremes
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
