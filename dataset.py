from pandas import DataFrame
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate a random dataset
def generate_dataset(n_samples, n_features, n_categories, seed):
    # Generate data with X,y format
    X, y = make_blobs(n_samples=n_samples, centers=n_categories, n_features=n_features, random_state=seed)
    # Create dictionary
    df = { i: X[:, i] for i in range(X.shape[1]) }
    df["category"] = y
    # Create dataframe
    df = DataFrame(df)
    return df.sort_values(by=["category"])

# Get attributes ranges
def attributes_ranges(df):
    # Define attributes intervals structure
    attributes_intervals = {} 
    # Remove category column from dataset
    df = df.iloc[:,:-1]
    # Get attributes interval for each attribute
    for attribute in df.columns:
        # Get min value
        min_value = df[attribute].min()
        # Get max value
        max_value = df[attribute].max()
        # Make interval
        interval = (min_value, max_value)
        # Add interval to attribute intervals
        attributes_intervals[attribute] = interval
    return attributes_intervals

# Plot an histogram from 1D data (group by categories)
def histogram(title, df):
    # Define figure
    fig, ax = plt.subplots()
    # Set chart title
    ax.set_title(title)
    # Get categories values
    categories = df.iloc[:,-1].unique()
    # Plot hist by category
    for category in categories:
        # Get distribution values
        values = df[df.iloc[:,1] == category].iloc[:,0]
        # Plot distribution
        ax.hist(values)

# Plot a scatter chart from 2D data (group by cateogries)
def scatter(title, df):
    # Define canvas
    fig, ax = plt.subplots()
    # Set title
    ax.set_title(title)
    # Define scatter chart
    plt.scatter(df.iloc[:,0], df.iloc[:,1], c = df.iloc[:,2])
    # Plot it
    plt.show()
    