import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Calculate KDE by category
def calculate_kdes(df):
    kdes = {}
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
        # TMP
        # min = np.min(values)
        # max = np.max(values)
        # support = np.linspace(min, max, 1000)
        # density = kde(support)
        # plt.fill_between(support, density)
        # Add KDE to KDEs dictionary
        kdes[category] = kde
    return kdes

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
def plot_univariate(df, kdes):
    # Define figure
    fig, ax = plt.subplots()
    # Plot by category
    for category in kdes:
        # Get distribution values
        values = df[df.iloc[:,1] == category].iloc[:,0]
        # Get min and max values
        min = values.min()
        max = values.max()
        # Generate support
        support = np.linspace(min, max, 1000)
        # Get category KDE
        kde = kdes[category]
        # Get KDE density 
        density = kde(support)
        # Plot distribution
        ax.fill_between(support, density)