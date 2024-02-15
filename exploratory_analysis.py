# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px

# Load the dataset
df = pd.read_csv('test_indicium_pricing.csv')

# Initial Exploration
print(df.head())
print(df.info())

# Descriptive Statistics
print(df.describe())

# Graphical Visualizations
plt.hist(df['price'], bins=30, color='blue', alpha=0.7)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x='room_type', y='price', data=df)
plt.show()

sns.scatterplot(x='longitude', y='latitude', hue='price', data=df)
plt.show()

# Select only numeric columns for correlation matrix calculation
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_columns].corr()

# Visualize the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Textual Analysis in Location Names (NLP)

# Geospatial Map
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
fig = px.scatter_mapbox(gdf, lat='latitude', lon='longitude', color='price',
                        hover_name='neighborhood', hover_data=['price'], zoom=10)
fig.update_layout(mapbox_style='open-street-map')
fig.show()

# Answer to Specific Questions
# Address the specific challenge questions based on the insights obtained.

# Save Results
# Save visualizations and intermediate results in a document or notebook.
