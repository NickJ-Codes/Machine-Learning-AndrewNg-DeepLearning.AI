from sklearn.datasets import load_iris
import pandas as pd

# Create a dataframe with features
iris = load_iris()
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

#add target column
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

#View the first few rows
print("First 5 rows:")
print(df.head())

# Full summary of the dataset
print("\nSummary statistics:")
print(df.describe())

# get info about the dataframe
print("\nDataset info")
print(df.info())

# se the idistribution of epecies
print("\nDistribution of species:")
print(df['species'].value_counts())