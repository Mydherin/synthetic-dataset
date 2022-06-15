from scipy import stats

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
        # Add KDE to KDEs dictionary
        kdes[category] = kde
    return kdes
