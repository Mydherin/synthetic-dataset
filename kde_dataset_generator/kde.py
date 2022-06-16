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

# Plot univariate dataset using KDE
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