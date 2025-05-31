# Imports.
import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import re

###------------------------------------------###
###                                          ###
### Function definitions for NFL QB analysis ###
###                                          ###
###------------------------------------------###

#-----------------------#
# Import NFLVerse data. #
#-----------------------#

def import_nflverse_data(path: str, start_year: int, file_prefix: str):
    """
    Takes a path to any of the NFLFastR GitHub pages and outputs a Polars lazyframe.

    Args:
        path (str): The web address of the NFLVerse data.
        start_year (str): The first year for which you want to pull data
        file_prefix (str): The file type that you wish to import--CSV, parquet, etc.
        
    Returns:
        polars.lazyframe.frame.LazyFrame: A Polars LazyFrame containing the data.
    """

    # Set the range for our loop dynamically so we can continue pulling data after the beginning of January without adjusting the list
    max_year_range = list(range(start_year, datetime.datetime.now().year + 1))

    if datetime.datetime.now().year == max_year_range[-1]:
        years = list(range(start_year, datetime.datetime.now().year))
    else:
        years = list(range(start_year, datetime.datetime.now().year + 1))

    # Initialize list from which we will eventually concatenate the data
    data_list = []

    file_type = file_prefix
    for year in years:
            data_year = pl.scan_parquet(path + f'{file_type}_{year}.parquet')
            data_list.append(data_year)

    data_concatenated = pl.concat(data_list)

    return data_concatenated

#------------------------------------#
# Generate lagged EMAs for features. #
#------------------------------------#

def compute_ema(df: pl.lazyframe.frame.LazyFrame, columns: list, group: list, spans: list):
    """
    Computes the EWMA for specified columns, EMA lags, and differences between observed and EMA lagged data across a range of spans.

    Args:
        df (polars.lazyframe.frame.LazyFrame): A Polars LazyFrame containing the data.
        columns (list): A list containing the columns to be operated on.
        group (list): A list of columns to be grouped by.
        spans (list): A list of integers to compute the EWMAs over.

    Returns:
        polars.lazyframe.frame.LazyFrame: The original Polars LazyFrame with added columns.
    """
    for span in spans:
        for col in columns:

            df = (
                df
                .with_columns(
                    # Calculate the EMAs
                    pl.col(col).ewm_mean(span=span).over(group).alias(f'{col}_ema{span}')
                )
                .with_columns(
                    # Lag them
                    pl.col(f'{col}_ema{span}').shift(1).over(group).alias(f'{col}_ema{span}_lagged')
                )
                .with_columns(
                    # Calculate the difference
                    (pl.col(col) - (pl.col(f'{col}_ema{span}_lagged'))).alias(f'{col}_ema{span}_diff')
                )
            )
    
    return df

#--------------------------------------------------#
# Generate scatterplots for feature visualization. #
#--------------------------------------------------#

def plot_nfl_scatter(df: pd.DataFrame, features: list, n_cols: int, n_rows: int, target_var: str):
    """
    Generates scatterplots with associated means for a supplied list of columns in a Pandas dataframe.

    Args:
        df (pd.DataFrame): A Pandas dataframe containing data to be plotted.
        features (list): A list containing the column names in the dataframe to be plotted.
        n_cols (int): The number of columns containing scatterplots.
        n_rows (int): The number of rows containing scatterplots.
        target_var (str): The target variable.

    Returns:
        A set of scatterplots.
    """
    
    plt.figure(figsize=(12, n_rows * 5))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.stripplot(x=target_var, y=feature, data=df, jitter=True, alpha=0.25)

        # Compute mean of feature for each pass touchdown count 
        mean_values = df.groupby(target_var)[feature].mean()
        
        for pass_td_count, mean_val in mean_values.items():
            sns.pointplot(x=target_var, y=feature, data=df, color='red', markers='D', linestyles='-', errorbar=None)
    
        cleaned_feature_name = re.sub(r'(_ema2_lagged|_ema12_lagged|offense_|defense_)', '', feature)
        
        plt.title(cleaned_feature_name, fontsize=9)
        plt.xlabel('Number of Pass Touchdowns', fontsize=9)
        plt.ylabel(cleaned_feature_name, fontsize=9)
    
    plt.tight_layout()
    plt.show()

#----------------------------------#
# Generate residual deviance plot. #
#----------------------------------#

def compute_plot_deviance_residuals(y_true: np.ndarray , y_pred: np.ndarray):
    """
    Generates residual deviance plot given vectors of observed counts and each predicted mu.

    Args:
        y_true (np.ndarray): A numpy array with true target values.
        y_pred (np.ndarray): A numpy array with predicted mean target values.

    Returns:
        A residual deviance plot.
    """
    
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(y_true == 0, 0, y_true * np.log(y_true / y_pred))
    deviance = 2 * (term - (y_true - y_pred))
    deviance = np.maximum(deviance, 0)
    residuals = np.sign(y_true - y_pred) * np.sqrt(deviance)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Mean Number of Passing Touchdowns')
    plt.ylabel('Deviance Residual')
    plt.title('Deviance Residuals for QB Passing Touchdowns in a Game')
    plt.show()

#----------------------------#
# Generate calibration plot. #
#----------------------------#

def plot_td_calibration(df: pd.DataFrame, n_bins=5):
    """
    Generates a calibration plot given a dataframe containing series of observed counts and each predicted mu.

    Args:
        df (pd.DataFrame): A Pandas dataframe containing y_true and y_pred.
        bins (int): The number of bins to split the predicted means into. Defaults to 5.

    Returns:
        A calibration curve.
    """
    df_cpy = df.copy()    
    df_cpy['bin'] = pd.qcut(df_cpy['y_pred'], q=n_bins, duplicates='drop')
    
    summary = (df_cpy.groupby('bin', observed=True).agg(mean_pred=('y_pred', 'mean'), mean_true=('y_true', 'mean'), se_true=('y_true', lambda x: x.std()/np.sqrt(len(x))), count=('y_true', 'size')).reset_index())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar( summary['mean_pred'], summary['mean_true'], yerr=summary['se_true'], fmt='o-', label='Observed mean ± SE')
    ax.plot(summary['mean_pred'], summary['mean_pred'], '--', label='Perfect calibration')
    
    for _, row in summary.iterrows():
        ax.annotate(int(row['count']), (row['mean_pred'], row['mean_true'] + row['se_true'] + 0.02), textcoords="offset points", xytext=(0, 0), ha='center')
    
    ax.set_xlabel('Predicted Mean Pass TDs')
    ax.set_ylabel('Observed Mean Pass TDs')
    ax.set_title('Calibration Plot for QB Passing TDs With Bin Counts')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.show()

#-----------------------------------------------------------#
# Generate plot of prediction interval for given team/game. #
#-----------------------------------------------------------#

def plot_prediction_interval(df: pd.DataFrame, team: str, season: int, week: int):
    """
    Generates plot with prediction interval for a given team-season-game.

    Args:
        df (pd.DataFrame): A dataframe containing prediction intervals and other specified columns below.
        team (str): A string containing the team abbreviation.
        season (int): The season year.
        week (int): The week number.
    """

    df_pred_int_plot = (
        df
        .filter((pl.col('team') == team) & (pl.col('season') == season) & (pl.col('week') == week))
        .to_pandas()
    )

    # Terminate if no data is available for that team-season-week combination
    if df_pred_int_plot.empty:
        print("No data available for the given team, season, and week.")
        return

    # Grab the data
    row = df_pred_int_plot.iloc[0]
    y_boot = row['y_pred_bootstrap']
    true_val = int(row['y_true'])

    # Construct bins so our labels are centered
    bins = np.arange(0, 7 + 1) - 0.5
    counts, bin_edges = np.histogram(y_boot, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Here is the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, counts, width=1.0, alpha=0.6, color='g', align='center')
    plt.xticks(bin_centers, labels=np.arange(0, 7, dtype=int))
    plt.xlim(-0.5, 6.5) # this clips the visual output to a reasonable range. The extreme tail values are not worth showing.

    # This overlays y_true
    obs_height = counts[true_val] if 0 <= true_val < len(counts) else 0
    plt.scatter([true_val], [obs_height + 0.02], marker='*', s=200, color='r', label='Observed count')

    plt.title(f"Pass TD Simulations for {row['player_display_name']}, Week {week} of {season}")
    plt.xlabel("Number of TD Passes")
    plt.ylabel("Proportion")
    plt.legend()
    plt.show()