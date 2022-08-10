from pandas import DataFrame
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
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

# Stratified Cross validation for any classifier
def scv(df, clf = GaussianNB(), folds=10):
    # Get X
    X = df.iloc[:,:-1]
    # Get y
    y = df.iloc[:,-1]
    # Define kind of cross validation
    cv = StratifiedKFold(n_splits=folds)
    # Do cross validation
    scores = cross_val_score(clf, X, y, cv=cv)
    # Get scores mean
    mean = scores.mean()
    # Get scores std
    std = scores.std()

    return mean, std

# Holdout for any classifier
def holdout(df, train_df = DataFrame(), clf = GaussianNB()):
    # Define X_train, y_train, X_test and y_test
    X_train, y_train, X_test, y_test = (None, None, None, None)
    # If there is a train df
    if not train_df.empty:
        # Get X_train
        X_train = train_df.iloc[:,:-1]
        # Get y_train
        y_train = train_df.iloc[:,-1]
        # Get X_test
        X_test = df.iloc[:,:-1]
        # Get y_test
        y_test = df.iloc[:,-1]
    else:
        # TODO: Implement this block
        pass
    # Train the model
    clf.fit(X_train, y_train)
    # Make predictions
    y_pred = clf.predict(X_test)
    # Calculate accuracy
    accuracy = (y_test == y_pred).sum()/len(y_test)
    
    return accuracy


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
    