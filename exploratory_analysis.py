# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import plotly.express as px

# Carregar o conjunto de dados
df = pd.read_csv('teste_indicium_precificacao.csv')

# Exploração Inicial
print(df.head())
print(df.info())

# Estatísticas Descritivas
print(df.describe())

# Visualizações Gráficas
plt.hist(df['price'], bins=30, color='blue', alpha=0.7)
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.show()

sns.boxplot(x='room_type', y='price', data=df)
plt.show()

sns.scatterplot(x='longitude', y='latitude', hue='price', data=df)
plt.show()

# Selecionar apenas colunas numéricas para o cálculo da matriz de correlação
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numeric_columns].corr()

# Visualizar a matriz de correlação
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Análise Textual nos Nomes dos Locais (NLP)

# Mapa Geoespacial
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
fig = px.scatter_mapbox(gdf, lat='latitude', lon='longitude', color='price',
                        hover_name='bairro', hover_data=['price'], zoom=10)
fig.update_layout(mapbox_style='open-street-map')
fig.show()

# Resposta às Perguntas Específicas
# Abordar as perguntas específicas do desafio com base nos insights obtidos.

# Salvar Resultados
# Salvar as visualizações e resultados intermediários em um documento ou notebook.
