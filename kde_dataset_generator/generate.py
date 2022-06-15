from pandas import DataFrame
from sklearn.datasets import make_blobs

# Generate a random dataset with 1 feature
def generate_test_dataset(n_samples, n_categories, seed):
    # Define feature number
    n_features = 1
    # Generate data with X,y format
    X, y = make_blobs(n_samples=300, centers=2, n_features=n_features, random_state=1)
    # Create dataframe
    df = DataFrame(dict(x=X[:,0], category=y))
    return df
