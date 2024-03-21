import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
arquivo = pd.read_csv("unidade2/dados.csv")

# Selecionar apenas as colunas numéricas
dados_numericos = arquivo.select_dtypes(include=['float64', 'int64'])

# Preencher valores ausentes com a média
dados_sem_nan = dados_numericos.dropna()
dados_sem_nan['Continent'] = arquivo['Continent']

# Lista para armazenar os DataFrames divididos por continente e década
dataframes_por_continente_e_decada = []

# Calcular a média da expectativa de vida global em cada década
anos_inicio = list(range(1950, 2010, 10))  # Início dos intervalos de 10 anos
anos_fim = list(range(1960, 2020, 10))     # Fim dos intervalos de 10 anos

for inicio, fim in zip(anos_inicio, anos_fim):
    for continente in dados_sem_nan['Continent'].unique():
        dataframe_decada_continente = dados_sem_nan[(dados_sem_nan['Year'] >= inicio) & (dados_sem_nan['Year'] < fim) & (dados_sem_nan['Continent'] == continente)]
        if not dataframe_decada_continente.empty:
            dataframes_por_continente_e_decada.append(dataframe_decada_continente)

# Lista para armazenar os coeficientes de regressão para cada subconjunto
coeficientes_vida_por_subconjunto = []
coeficientes_pib_por_subconjunto = []
mse_por_subconjunto = []

# Iterar sobre cada subconjunto de dados por continente e década
for dataframe in dataframes_por_continente_e_decada:
    # Separar as features (variáveis independentes) e o target (variável dependente) para expectativa de vida
    X_vida = dataframe[['Year', 'GDP per capita', 'Population (historical estimates)']]
    y_vida = dataframe['Life expectancy']
    # Separar as features (variáveis independentes) e o target (variável dependente) para PIB
    X_pib = dataframe[['Year', 'GDP per capita', 'Population (historical estimates)']]
    y_pib = dataframe['GDP per capita']

    # Dividir os dados em conjunto de treinamento e teste para expectativa de vida
    X_train_vida, X_test_vida, y_vida_train, y_vida_test = train_test_split(X_vida, y_vida, test_size=0.2, random_state=42)
    # Dividir os dados em conjunto de treinamento e teste para PIB
    X_train_pib, X_test_pib, y_pib_train, y_pib_test = train_test_split(X_pib, y_pib, test_size=0.2, random_state=42)

    # Treinar o modelo de regressão linear para expectativa de vida
    model_vida = LinearRegression()
    model_vida.fit(X_train_vida, y_vida_train)
    coeficientes_vida_por_subconjunto.append({
        'Continent': dataframe['Continent'].iloc[0],
        'Decade': dataframe['Year'].iloc[0],
        'Coeficientes': model_vida.coef_
    })

    # Treinar o modelo de regressão linear para PIB
    model_pib = LinearRegression()
    model_pib.fit(X_train_pib, y_pib_train)
    coeficientes_pib_por_subconjunto.append({
        'Continent': dataframe['Continent'].iloc[0],
        'Decade': dataframe['Year'].iloc[0],
        'Coeficientes': model_pib.coef_
    })

    # Fazer previsões no conjunto de teste para expectativa de vida
    y_pred_vida = model_vida.predict(X_test_vida)
    # Fazer previsões no conjunto de teste para PIB
    y_pred_pib = model_pib.predict(X_test_pib)

    # Calcular o MSE para o subconjunto atual para expectativa de vida
    mse_vida = mean_squared_error(y_vida_test, y_pred_vida)
    # Calcular o MSE para o subconjunto atual para PIB
    mse_pib = mean_squared_error(y_pib_test, y_pred_pib)
    mse_por_subconjunto.append({'Continent': dataframe['Continent'].iloc[0], 'Decade': dataframe['Year'].iloc[0], 'MSE_vida': mse_vida, 'MSE_pib': mse_pib})

    # Plotar os gráficos para expectativa de vida
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_vida['Year'], y_vida_train, color='blue', label='Expectativa de Vida (Treinamento)')
    plt.scatter(X_test_vida['Year'], y_vida_test, color='red', label='Expectativa de Vida (Teste)')
    plt.plot(X_test_vida['Year'], model_vida.predict(X_test_vida), color='green', linewidth=2, label='Regressão Linear (Vida)')
    plt.xlabel('Ano')
    plt.ylabel('Expectativa de Vida')
    plt.title(f'Regressão Linear para {dataframe["Continent"].iloc[0]} - década de {dataframe["Year"].iloc[0]}s')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dataframe["Continent"].iloc[0]}_decada_{dataframe["Year"].iloc[0]}s_regression_vida.png')
    plt.close()

    # Plotar os gráficos para PIB
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_pib['Year'], y_pib_train, color='blue', label='PIB (Treinamento)')
    plt.scatter(X_test_pib['Year'], y_pib_test, color='red', label='PIB (Teste)')
    plt.plot(X_test_pib['Year'], model_pib.predict(X_test_pib), color='green', linewidth=2, label='Regressão Linear (PIB)')
    plt.xlabel('Ano')
    plt.ylabel('PIB per capita')
    plt.title(f'Regressão Linear para {dataframe["Continent"].iloc[0]} - década de {dataframe["Year"].iloc[0]}s')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{dataframe["Continent"].iloc[0]}_decada_{dataframe["Year"].iloc[0]}s_regression_pib.png')
    plt.close()

# Imprimir os coeficientes de regressão para expectativa de vida para cada subconjunto
for coeficientes in coeficientes_vida_por_subconjunto:
    print(f"Coeficientes de regressão para {coeficientes['Continent']} - década de {coeficientes['Decade']}s (Expectiva de vida)")

# Imprimir os coeficientes de regressão para expectativa de vida para cada subconjunto
for coeficientes in coeficientes_vida_por_subconjunto:
    print(f"Coeficientes de regressão para {coeficientes['Continent']} - década de {coeficientes['Decade']}s (Expectiva de vida)")

# Imprimir os MSEs para cada subconjunto
for mse_info in mse_por_subconjunto:
    print(f"MSE para {mse_info['Continent']} - década de {mse_info['Decade']}s (Expectiva de vida): {mse_info['MSE_vida']}")

    
