import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Calculate KDE by category
def calculate_kdes(df):
    kdes = {}
    ranges = {}
    # Get categories values
    categories = df.iloc[:,-1].unique()
    # Caculate KDE for each category
    for category in categories:
        # Filter df by category
        filtered_df = df[df.iloc[:,-1] == category]
        # Get attributes values
        values = filtered_df.iloc[:,:-1].to_numpy()
        # Calculate KDE
        kde = stats.gaussian_kde(values.T)
        # Get min
        min_value = np.min(values)
        # Get max
        max_value = np.max(values)
        # Add range to ranges list
        ranges[category] = (min_value,max_value)
        # Add KDE to KDEs dictionary
        kdes[category] = kde
    return kdes, ranges

# Estimate instance densities using KDEs
def estimate_density(df, kdes):
    # Define new df
    new_df = []
    # Get categories
    categories = df.iloc[:,-1].unique()
    # Estimate instance density for each category instances
    for category in categories:
        # Filter dataset by category
        filtered_df = df[df.iloc[:,-1] == category]
        # Get category kde
        kde = kdes[category]
        # Estimate density for each instance in filtered dataset
        for index, row in df.iterrows():
            # Instance without category to numpy
            instance = row.iloc[:-1].to_numpy()
            # Estimate density
            density = kde(instance)[0]
            # Instance with category to list
            instance = row.to_list()
            # Add density to instance
            instance.append(density) 
            # Add instance to new df
            new_df.append(instance)
    # Define new dataset columns
    columns = df.columns.to_list()
    columns.append("density")
    # Create new df from new df instances
    new_df = pd.DataFrame(data=new_df, columns=columns)
    return new_df

# Plot univariate dataset with multiplet categories using KDE
def plot_univariate(df, kdes, ranges=False):
    # Define figure
    fig, ax = plt.subplots()
    # Plot by category
    for category, kde in kdes.items():
        # Get distribution values
        values = df[df.iloc[:,1] == category].iloc[:,0]
        # If there ara ranges
        if ranges:
            # There are ranges
            # Get min and max values from ranges
            min_value = ranges[category][0]
            max_value = ranges[category][1]
        else:
            # There are no ranges
            # Get min and max values from values
            min_value = values.min()
            max_value = values.max()
        # Generate support
        support = np.linspace(min_value, max_value, 1000)
        # Get KDE density 
        density = kde(support)
        # Plot distribution
        ax.fill_between(support, density)

# Build ranking by density
def build_ranking(df):
    return df.sort_values(by=['density'], ascending=False)

# Gets threshold first intances from ranking
def prune(ranking, threshold):
    # Prune ranking by threshold
    ranking = ranking.iloc[:threshold,:]
    # Remove density column
    ranking = ranking.iloc[:,:-1]
    return ranking