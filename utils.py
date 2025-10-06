import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from adjustText import adjust_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import statsmodels.api as sm


def calculate_age(row):
    if pd.isna(row['birth_date']) or pd.isna(row['season']):
        return None
    try:
        # July 1 of that season year
        season_year = int(row['season'])
        season_date = datetime(season_year, 7, 1)
        age = (season_date - row['birth_date']).days / 365.25
        return round(age, 2)
    except:
        return None
    

def make_comparison_df(df, league1, league2, prefix1, prefix2):
    """Helper function to create comparison dataset between two leagues."""

    # Define columns to include
    cols_to_use = [
        'internal_box_plus_minus', 'true_shooting_percentage', 'three_point_attempt_rate',
        'turnover_percentage', 'usage_percentage', 'total_rebounding_percentage',
        'offensive_rebounding_percentage', 'defensive_rebounding_percentage',
        'assist_percentage', 'steal_percentage', 'block_percentage'
    ]
    df1 = df[df['league'] == league1][['full_name', 'season', 'age'] + cols_to_use].copy()
    df2 = df[df['league'] == league2][['full_name', 'season', 'age'] + cols_to_use].copy()

    # Rename columns with prefixes
    df1 = df1.rename(columns={col: f"{prefix1}_{col}" for col in cols_to_use})
    df2 = df2.rename(columns={col: f"{prefix2}_{col}" for col in cols_to_use})

    # Drop 'age' from df2 to avoid duplicate column conflict
    df2 = df2.drop(columns=['age'])

    # Merge on full_name and season
    merged = df1.merge(df2, on=['full_name', 'season'], how='inner')

    return merged


def compare_league_correlations(df, prefix1, prefix2):
    """
    Compute correlations and average ratios (or differences for IBPM) between two sets of league stats.

    Parameters:
        df (pd.DataFrame): The merged comparison dataframe (e.g., eurocup_vs_euroleague)
        prefix1 (str): The prefix for the lower league (e.g., 'EC', 'LA', 'ACB')
        prefix2 (str): The prefix for the EuroLeague columns (e.g., 'EL')

    Returns:
        pd.DataFrame: Summary table of correlations and average ratios/differences.
    """
    metrics = [
        'internal_box_plus_minus', 'true_shooting_percentage', 'three_point_attempt_rate',
        'turnover_percentage', 'usage_percentage', 'total_rebounding_percentage',
        'offensive_rebounding_percentage', 'defensive_rebounding_percentage',
        'assist_percentage', 'steal_percentage', 'block_percentage'
    ]

    results = []

    for metric in metrics:
        col1 = f"{prefix1}_{metric}"
        col2 = f"{prefix2}_{metric}"

        # Clean and filter out problematic values
        temp = df[[col1, col2]].replace([np.inf, -np.inf], np.nan).dropna()
        temp = temp[(temp[col1] != 0) & (temp[col2] != 0)]

        if len(temp) > 0:
            corr = temp[col1].corr(temp[col2])
            
            if metric == 'internal_box_plus_minus':
                # Difference for IBPM instead of ratio
                ratio_or_diff = (temp[col2] - temp[col1]).mean()
            else:
                ratio_or_diff = (temp[col2] / temp[col1]).mean()
        else:
            corr = np.nan
            ratio_or_diff = np.nan

        results.append({
            'metric': metric,
            'correlation': corr,
            'average_change_EL_to_lower_league': ratio_or_diff
        })

    summary = pd.DataFrame(results).sort_values(by='correlation', ascending=False)
    return summary


def plot_league_correlation(summary_df, league_name):
    """
    Scatter plot of correlation vs average change for a league comparison.
    Labels are adjusted to avoid overlap.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='correlation',
        y='average_change_EL_to_lower_league',
        data=summary_df,
        s=100,
        color='dodgerblue'
    )

    # Create text objects for each point
    texts = [
        plt.text(row['correlation'], row['average_change_EL_to_lower_league'], row['metric'], fontsize=9)
        for i, row in summary_df.iterrows()
    ]

    # Adjust text to minimize overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    plt.title(f"{league_name} - Correlation vs Average Change", fontsize=14)
    plt.xlabel("Correlation", fontsize=12)
    plt.ylabel("Average Change (EL / Lower League or EL - Lower for IBPM)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



def get_latest_pre_nba_row(group):
    """
    For each player group:
    - If earliest_nba_season exists, return the row with the latest season < earliest_nba_season
      (if any). If none exist, return NaN.
    - If earliest_nba_season is missing (NaN), return the row with the latest season overall.
    """
    earliest_nba = group['earliest_nba_season'].iloc[0]  # or 'earliest_nba_season' if that's your column name

    if pd.notna(earliest_nba):
        # Filter for seasons before earliest NBA season
        before_nba = group[group['season'] < earliest_nba]
        if not before_nba.empty:
            return before_nba.loc[[before_nba['season'].idxmax()]]
        else:
            return pd.DataFrame(columns=group.columns)  # no valid pre-NBA season
    else:
        # No NBA career → return latest season
        return group.loc[[group['season'].idxmax()]]


def aggregate_latest_pre_nba(df):
    """
    Groups the given league DataFrame by 'full_name' and applies the logic above.
    """
    result = (
        df.groupby('full_name', group_keys=False)
          .apply(get_latest_pre_nba_row)
          .reset_index(drop=True)
    )

    # Rename the column safely if it exists
    if 'season' in result.columns:
        result = result.rename(columns={'season': 'latest_season_before_nba'})
    
    return result


def add_jump_column(df):
    """
    Adds a 'jump' column:
    - True if latest_season_before_nba < earliest_nba_season
    - False otherwise (including if earliest_nba_season is NaN)
    """
    df['jump'] = (
        (df['latest_season_before_nba'] < df['earliest_nba_season'])
        & df['earliest_nba_season'].notna()
    )
    return df


def summarize_jump_info(df, league_name):
    print(f"--- {league_name} ---")
    print("Jump value counts:")
    print(df['jump'].value_counts(dropna=False))
    
    true_pct = (df['jump'].sum() / len(df)) * 100
    print(f"\nPercentage of Players that Made the NBA: {true_pct:.2f}%")
    
    print("\nRows where earliest_nba_season is empty:")
    print(df['earliest_nba_season'].isna().sum())
    print("\n")


# Function to compute overlap stats
def league_to_el_overlap(df, league_name, agg_el_df):

    # Get the set of EuroLeague player names for reference
    el_players = set(agg_el_df['full_name'])

    # Players who made the NBA ("jump" == True)
    jump_players = df[df['jump'] == True]['full_name']
    total_jump = len(jump_players)
    
    # Of those, how many are also in EuroLeague?
    overlap = len(set(jump_players) & el_players)
    
    # Calculate percentage
    pct_overlap = (overlap / total_jump * 100) if total_jump > 0 else 0
    
    print(f"{league_name}:")
    print(f"  Total players with jump = True: {total_jump}")
    print(f"  Players also in EuroLeague: {overlap}")
    print(f"  Percentage of overlap: {pct_overlap:.2f}%\n")


def run_lasso_logistic_with_pvalues(df, league_name):
    """
    Runs LASSO logistic regression to predict 'jump', selects significant features,
    and refits a statsmodels Logit model for p-values.
    """

    feature_cols = [
        'starts', 'minutes', 'usage_percentage', 'true_shooting_percentage',
        'three_point_attempt_rate', 'free_throw_rate', 'offensive_rebounding_percentage',
        'defensive_rebounding_percentage', 'assist_percentage', 'steal_percentage',
        'block_percentage', 'turnover_percentage', 'internal_box_plus_minus', 'age'
    ]

    # Drop rows with missing data
    df = df.dropna(subset=['jump'] + feature_cols)
    df = df.copy()

    # Convert to numeric to avoid object dtype issues
    for col in feature_cols + ['jump']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['jump'])
    X = df[feature_cols]
    y = df['jump'].astype(int)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Step 1: LASSO Logistic Regression (feature selection)
    model_lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    model_lasso.fit(X_train, y_train)

    y_pred = model_lasso.predict(X_test)
    print(f"\n--- {league_name} ---")
    print(classification_report(y_test, y_pred, digits=3))

    coefs = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model_lasso.coef_[0]
    })
    selected = coefs[coefs['coefficient'] != 0]['feature'].tolist()

    if not selected:
        print("\n⚠️ No features selected by LASSO. Skipping p-value step.")
        return model_lasso, None

    print(f"\nSelected features from LASSO: {selected}")

    # Step 2: Fit statsmodels logistic regression for p-values
    X_selected = df[selected].apply(pd.to_numeric, errors='coerce')
    X_selected = sm.add_constant(X_selected)

    # Double-check for object dtype (shouldn’t happen now)
    if any(X_selected.dtypes == 'object'):
        print("⚠️ Non-numeric columns remain in X_selected. Converting to float.")
        X_selected = X_selected.astype(float)

    logit_model = sm.Logit(y, X_selected)
    result = logit_model.fit(disp=False)

    print("\nStatsmodels Logistic Regression Summary:")
    print(result.summary())

    summary_df = pd.DataFrame({
        'feature': result.params.index,
        'coef': result.params.values,
        'p_value': result.pvalues.values
    }).sort_values(by='p_value')

    print("\nFeature Significance (sorted by p-value):")
    print(summary_df)

    return model_lasso, summary_df

def plot_stat_distribution_across_leagues(
    agg_el_df_without_2021, agg_ec_df_without_2021, agg_acb_df_without_2021, agg_la_df_without_2021, leagues_without_2021, stat_col
):
    """
    Plots the distribution of a statistic across four leagues,
    split by whether the player made the NBA jump (jump==True vs False).

    Parameters:
        agg_el_df, agg_ec_df, agg_acb_df, agg_la_df: DataFrames for each league
        stat_col (str): The column to visualize
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (league_name, df) in enumerate(leagues_without_2021.items()):
        ax = axes[i]

        # Drop missing values for the statistic
        df_plot = df.dropna(subset=[stat_col, 'jump'])

        # Plot the distribution split by jump
        sns.histplot(
            data=df_plot,
            x=stat_col,
            hue='jump',
            multiple='stack',
            palette={True: 'green', False: 'red'},
            bins=15,
            alpha=0.7,
            ax=ax
        )

        ax.set_title(f"{league_name}: {stat_col} distribution by NBA jump")
        ax.set_xlabel(stat_col)
        ax.set_ylabel("Count")
        ax.legend(title="Jumped to NBA", labels=["True", "False"])

    plt.tight_layout()
    plt.show()


def run_lasso_linear_with_pvalues(df, league_name):
    """
    Runs LASSO linear regression to predict avg_nba_ibpm, selects significant features,
    and refits an OLS model for p-values.
    """

    feature_cols = [
        'starts', 'minutes', 'usage_percentage', 'true_shooting_percentage',
        'three_point_attempt_rate', 'free_throw_rate', 'offensive_rebounding_percentage',
        'defensive_rebounding_percentage', 'assist_percentage', 'steal_percentage',
        'block_percentage', 'turnover_percentage', 'internal_box_plus_minus', 'age'
    ]

    # Drop rows with missing data
    df = df.dropna(subset=['avg_nba_ibpm'] + feature_cols).copy()

    # Convert to numeric
    for col in feature_cols + ['avg_nba_ibpm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['avg_nba_ibpm'])
    X = df[feature_cols]
    y = df['avg_nba_ibpm']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # Step 1: LASSO feature selection
    lasso = LassoCV(cv=min(5, len(X_train)//5), random_state=42)
    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"\n--- {league_name} ---")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Best alpha (regularization strength): {lasso.alpha_:.5f}")

    # Get selected features
    coefs = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lasso.coef_
    })
    selected = coefs[coefs['coefficient'] != 0]['feature'].tolist()

    if not selected:
        print("\n⚠️ No features selected by LASSO. Skipping p-value step.")
        return lasso, None

    print(f"\nSelected features from LASSO: {selected}")

    # Step 2: OLS Regression for p-values
    X_selected = sm.add_constant(df[selected])
    ols_model = sm.OLS(y, X_selected).fit()

    print("\nStatsmodels OLS Regression Summary:")
    print(ols_model.summary())

    summary_df = pd.DataFrame({
        'feature': ols_model.params.index,
        'coef': ols_model.params.values,
        'p_value': ols_model.pvalues.values
    }).sort_values(by='p_value')

    print("\nFeature Significance (sorted by p-value):")
    display(summary_df)  # cleaner than print for notebooks

    # ✅ Return silently (won’t auto-print in Jupyter)
    return lasso, summary_df


def step1_filter_players(df, league, max_age=26, min_minutes=300, min_ibpm=0):
    """
    Filters a single league DataFrame based on:
      - age <= max_age
      - minutes >= min_minutes
      - internal_box_plus_minus > min_ibpm
    Returns the filtered DataFrame and prints the number of rows.
    """
    filtered_df = df[
        (df['age'] <= max_age) &
        (df['minutes'] >= min_minutes) &
        (df['internal_box_plus_minus'] > min_ibpm)
    ]
    print(f"Filtered dataset of {league} has {len(filtered_df)} rows")
    return filtered_df


def step2_filter_players(df, league, min_free_throw_rate=0.2, min_steal_percentage=2, min_true_shooting_percentage=0.5):
    """
    Further filters the Step 1 dataset based on:
      - free_throw_rate > min_free_throw_rate
      - steal_percentage > min_steal_percentage
    Returns the filtered DataFrame and prints the number of rows.
    """
    filtered_df = df[
        (df['free_throw_rate'] > min_free_throw_rate) &
        (df['steal_percentage'] > min_steal_percentage) &
        (df['true_shooting_percentage'] > min_true_shooting_percentage)
    ]
    print(f"Step 2 filtered dataset of {league} has {len(filtered_df)} rows")
    return filtered_df