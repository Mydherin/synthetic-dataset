from pandas import DataFrame
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate a random dataset with 1 feature
def generate_univariate_dataset(n_samples, n_categories, seed):
    # Define feature number
    n_features = 1
    # Generate data with X,y format
    X, y = make_blobs(n_samples=n_samples, centers=n_categories, n_features=n_features, random_state=seed)
    # Create dataframe
    df = DataFrame(dict(x=X[:,0], category=y))
    return df

# Plot an histogram showing univariate dataset distribution
def plot_univariate(df):
    # Define figure
    fig, ax = plt.subplots()
    # Get categories values
    categories = df.iloc[:,-1].unique()
    # Plot hist by category
    for category in categories:
        # Get distribution values
        values = df[df.iloc[:,1] == category].iloc[:,0]
        # Plot distribution
        ax.hist(values)

# Plot an histogram from distribution
def plot_distribution(distribution):
    # Define figure
    fig, ax = plt.subplots()
    # Plot distribution
    ax.hist(distribution)