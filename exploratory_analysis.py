# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px

# Load the dataset
df = pd.read_csv('teste_indicium_precificacao.csv')

# Initial Exploration
print(df.head())
print(df.info())

df.last()

# Descriptive Statistics
print(df.describe())

# Graphical Visualizations
plt.hist(df['price'], bins=30, color='blue', alpha=0.7)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Histogram Plot for 'price'
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Boxplot for 'room_type' in relation to 'price'
plt.figure(figsize=(12, 8))
sns.boxplot(x='room_type', y='price', data=df, palette='Set3')
plt.title('Price Distribution by Room Type')
plt.xlabel('Room Type')
plt.ylabel('Price')
plt.show()

# Scatterplot for 'longitude' and 'latitude' with 'price' encoded by color
plt.figure(figsize=(14, 10))
sns.scatterplot(x='longitude', y='latitude', hue='price', data=df, palette='viridis', size='price', sizes=(10, 200))
plt.title('Geospatial Distribution of Prices')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
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
                        hover_name='bairro_group', hover_data=['price'], zoom=10)
fig.update_layout(mapbox_style='open-street-map')
fig.show()

# Answer to Specific Questions
# Address the specific challenge questions based on the insights obtained.

# Initial Analysis:
# 1. The histogram shows the distribution of prices, indicating a predominance of lower values.
# 2. The boxplot highlights differences in prices between different room types, with some notable discrepancies.
# 3. The geospatial scatterplot reveals patterns in the distribution of prices in different regions of New York.

# business hypotheses:
# - Specific locations (longitude and latitude) influence prices.
# - Certain room types have a significant price variation.
# - Price distribution may follow geographical or demand patterns in the city.

# These observations guide your pricing strategy, focusing on strategic regions or adjusting prices based on room type.
