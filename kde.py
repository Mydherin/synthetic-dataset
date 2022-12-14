import pandas as pd
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Calculate KDE for each category
def kdes(df):
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

# Generate the support for univariate dataset 
def univariate_support(df, granularity):
    # Define new df
    new_df = []
    # Get categories
    categories = df.iloc[:,-1].unique()
    # Generate support for each category
    for category in categories:
        # Filter dataset by category
        filtered_df = df[df.iloc[:,-1] == category]
        # Get min value
        min_value = filtered_df.iloc[:,0].min()
        # Get max value
        max_value = filtered_df.iloc[:,0].max()
        # Build the support
        support = np.linspace(min_value, max_value, granularity)
        support = list(zip(support, np.full(granularity, category)))
        # Add support to new df
        new_df.extend(support)
    # Build new df
    new_df = pd.DataFrame(data=new_df, columns=df.columns)
    return new_df 

# Generate new dataset from a multivariate dataset.
# This distribution of this dataset is a rectangle, this means that the density is the same for each instance.
def multivariate_dataset(df, granularity):
    # Define new df
    new_df = []
    # Get categories
    categories = df.iloc[:,-1].unique()
    # Define n_attributes
    n_attributes = len(df.columns) - 1
    # Get instances for each category
    for category in categories:
        # Filter dataset by category
        filtered_df = df[df.iloc[:,-1] == category]
        # Define attributes supports
        attributes_supports = []
        # Get support for each attribute
        for attribute in range(n_attributes):
            # Get min value
            min_value = filtered_df.iloc[:,attribute].min()
            # Get max value
            max_value = filtered_df.iloc[:,attribute].max()
            # Build the support
            support = np.linspace(min_value, max_value, granularity)
            # Add support to attributes supports 
            attributes_supports.append(support)
        # Generate combinations between all attributes supports
        ## Generate meshgrid
        meshgrid = np.meshgrid(*attributes_supports)
        ## Reshape meshgrid as vectors (one per dimension)
        meshgrid = [support.ravel() for support in meshgrid]
        ## Stack all vectors and transpose them
        combinations = np.vstack(meshgrid).T
        ## Assign category correctly to each combination
        for combination in combinations:
            # Parse numpy combination to list
            combination = combination.tolist()
            # Assing category
            combination.append(category)
            # Add instance to new_df
            new_df.append(combination)
    # Create new df
    new_df = pd.DataFrame(new_df, columns=df.columns)
    return new_df

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
        for index, row in filtered_df.iterrows():
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

# Generate new dataser adjusting instance representation.
def adjust_representation(densities):
    # Define new instances
    new_instances = []
    # Get categories
    categories = densities.iloc[:,-2].unique()
    # Estimate instance density for each category instances
    for category in categories:
        # Filter dataset by category
        filtered_densities = densities[densities.iloc[:,-2] == category]
        # Get min density value
        min_density = filtered_densities.iloc[:,-1].min()
        # Generate proportional number (density based) of instances for each instance
        for index, row in filtered_densities.iterrows():
            # Get instance density
            density = row["density"]
            # Estimate amount of duplicate instances
            n_duplicate_instances = round(density/min_density)
            # Generate duplicate instances
            duplicate_instances = [row.iloc[:-1].to_list() for i in range(n_duplicate_instances)]
            # Add duplicate instances to new dataframe
            new_instances.extend(duplicate_instances)
    # Define new columns
    columns = densities.columns.to_list()
    del columns[-1]
    # Define new df
    new_df = pd.DataFrame(data=new_instances, columns=columns)
    return new_df

# Generate new dataser adjusting instance representation.
def multivariate_adjust_representation(densities):
    # Define new instances
    new_instances = []
    # Get categories
    categories = densities.iloc[:,-2].unique()
    # Estimate instance density for each category instances
    for category in categories:
        # Filter dataset by category
        filtered_densities = densities[densities.iloc[:,-2] == category]
        # Sort by density 
        filtered_densities = filtered_densities.sort_values(by=['density'], ascending=False)
        # Define min density
        PERCENT = 0.5
        reference_instance = round(len(filtered_densities) * PERCENT)
        min_density = filtered_densities.iloc[reference_instance, -1]
        # Generate proportional number (density based) of instances for each instance
        for index, row in filtered_densities.iterrows():
            # Get instance density
            density = row["density"]
            # Estimate amount of duplicate instances
            n_duplicate_instances = round(density/min_density)
            # Generate duplicate instances
            duplicate_instances = [row for i in range(n_duplicate_instances)]
            # Add duplicate instances to new dataframe
            new_instances.extend(duplicate_instances)
    # Define new df
    new_df = pd.DataFrame(data=new_instances, columns=densities.columns)
    new_df = new_df.drop(["density"], axis=1)
    return new_df

# Genrate new dataframe randomly using KDEs
def random_generation(n_instances, kdes, attribute_intervals, columns, seed):
    # Set random seed
    random.seed(seed)
    # Define new df
    new_df = {}
    # Define finish
    finish = {}
    # Initialize finish and new_df
    for category in kdes:
        finish[category] = False
        new_df[category] = []
    # Generate and label instances until reaching the target amount for each category
    while not all(list(finish.values())):
        # Define instance
        instance = []
        # Build new instance generating an attribute value for each attribute
        for attribute in attribute_intervals:
            # Generate random value between min and max value
            new_value = random.uniform(attribute_intervals[attribute][0],attribute_intervals[attribute][1])
            # Add value to instance
            instance.append(new_value)
        # Define max density
        max_density = -1
        # Define new category
        new_category = None
        # Get max density between densities which have been estimate for each category
        for category in kdes:
            # Estimate density
            density = kdes[category](instance)
            # If density is the biggest density
            if density > max_density:
                # It is the biggest density
                # Change max denstity to current denstiy
                max_density = density
                # Change new category to current category
                new_category = category
        # If category is not full
        if len(new_df[new_category]) < n_instances:
            # If category is not full
            # Add category to instance
            instance.append(new_category)
            # Add instance new_df category
            new_df[new_category].append(instance)
        else:
            # If category is full
            # Set category is full in finish
            finish[new_category] = True
    # Wrap new df
    new_df_building = []
    for category in kdes:
        new_df_building.extend(new_df[category])
    new_df = new_df_building
    # Define final new df
    new_df = pd.DataFrame(data=new_df, columns=columns)
    return new_df

# Plot a 2D chart from 1D data (group by categories)
def plot_2d(title, df, kdes):
    # Define figure
    fig, ax = plt.subplots()
    # Set chart title
    ax.set_title(title)
    # Plot by category
    for category, kde in kdes.items():
        # Get distribution values
        values = df[df.iloc[:,1] == category].iloc[:,0]
        # Get min and max values from values
        min_value = values.min()
        max_value = values.max()
        # Generate support
        support = np.linspace(min_value, max_value, 1000)
        # Get KDE density 
        density = kde(support)
        # Plot distribution
        ax.fill_between(support, density)

# Plot a 3D chart from 2D data
def plot_3d(title, df):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    
    # Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10

    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kde = stats.gaussian_kde(values)
    f = np.reshape(kde(positions).T, xx.shape)
    
    # PLot 3D chart
    fig, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    w = ax.plot_wireframe(xx, yy, f)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title(title)

