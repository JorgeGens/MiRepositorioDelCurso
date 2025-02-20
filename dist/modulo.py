# Instalamos las librerias necesarias que vamos necesitando
# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install seaborn
# !pip install psycopg2
# !pip install sqlalchemy

# Importo las librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chequeo que se han instalado bien
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("matplotlib:", plt.matplotlib.__version__)

# Carga de datos.  Accedo a los archivos en GitHub y copio la ruta raw
ruta1 = "https://raw.githubusercontent.com/ricardoahumada/DataExpert/refs/heads/main/etapa2/data/Caracteristicas_Equipos.csv"
ruta2 = "https://raw.githubusercontent.com/ricardoahumada/DataExpert/refs/heads/main/etapa2/data/Historicos_Ordenes.csv"
ruta3 = "https://raw.githubusercontent.com/ricardoahumada/DataExpert/refs/heads/main/etapa2/data/Registros_Condiciones.csv"

# Guardo los valores en estas variables
caracteristicas_equipos = pd.read_csv(ruta1)
historicos_ordenes = pd.read_csv(ruta2)
registros_condiciones = pd.read_csv(ruta3)

# Al traerme los datos los toma como strings. Paso los valores de fecha a datetime
# y si por lo que sea no se puede para algún valor, hago que valga Nan en lugar de que genere une error, con errors='coerce'
historicos_ordenes['Fecha'] = pd.to_datetime(historicos_ordenes['Fecha'], errors='coerce')
registros_condiciones['Fecha'] = pd.to_datetime(registros_condiciones['Fecha'], errors='coerce')

# ELimino los valores nulos:
historicos_ordenes.dropna(subset=['Costo_Mantenimiento'], inplace=True)
registros_condiciones.dropna(subset=['Horas_Operativas'], inplace=True)

# Eliminación de Duplicados
historicos_ordenes.drop_duplicates(inplace=True)
registros_condiciones.drop_duplicates(inplace=True)
caracteristicas_equipos.drop_duplicates(inplace=True)

# Detección de Outliers en Columnas Clave. valores atípicos en Costo_Mantenimiento y Horas_Operativas usando el método de Rango Intercuartil (IQR)
def detectar_outliers_iqr(df, columna):
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    return outliers

# Aplicar la detección en columnas clave
outliers_costo = detectar_outliers_iqr(historicos_ordenes, 'Costo_Mantenimiento')
outliers_horas = detectar_outliers_iqr(registros_condiciones, 'Horas_Operativas')

print("Outliers en Costo de Mantenimiento:", outliers_costo.shape[0])
print("Outliers en Horas Operativas:", outliers_horas.shape[0])

# Unimos los tres datasets para obtener una vista consolidada:

# Fusionamos condiciones operativas con características de equipos
merged_df = pd.merge(registros_condiciones, caracteristicas_equipos, on='ID_Equipo', how='left')

# Fusionamos con órdenes de mantenimiento solo por ID_Equipo (sin fecha exacta para capturar más datos)
df_final = pd.merge(merged_df, historicos_ordenes, on='ID_Equipo', how='left')

# El dataset limpio y transformado se guarda en formato CSV para futuros análisis.
df_final.to_csv("dataset_final.csv", index=False)

# Filtrar valores fuera del rango esperado
valores_erroneos = df_final[
    (df_final['Temperatura_C']  < 0 )  | (df_final['Temperatura_C']  > 700 )  |
    (df_final['Vibracion_mm_s'] < 0 )  | (df_final['Vibracion_mm_s'] > 100 )
]

print(valores_erroneos)

# Distribución de Fallos y Tipos de Mantenimiento
import seaborn as sns
import matplotlib.pyplot as plt

# Contar el número de mantenimientos por tipo
tipo_mantenimiento = historicos_ordenes['Tipo_Mantenimiento'].value_counts()

# Visualización con gráfico de barras
plt.figure(figsize=(8,5))
sns.barplot(x=tipo_mantenimiento.index, y=tipo_mantenimiento.values)
plt.xlabel("Tipo de Mantenimiento")
plt.ylabel("Cantidad de Fallos")
plt.title("Distribución de Fallos por Tipo de Mantenimiento")
plt.show()

# Calcular la correlación entre variables
correlacion = registros_condiciones[['Temperatura_C', 'Vibracion_mm_s', 'Horas_Operativas']].corr()
print(correlacion)

# Visualización de la correlación con un mapa de calor
plt.figure(figsize=(8,6))
sns.heatmap(correlacion, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor de Correlaciones")
plt.show()

# Relación entre temperatura y horas operativas
plt.figure(figsize=(8,5))
sns.scatterplot(x=registros_condiciones['Temperatura_C'], y=registros_condiciones['Horas_Operativas'], alpha=0.5)
plt.xlabel("Temperatura (°C)")
plt.ylabel("Horas Operativas")
plt.title("Relación entre Temperatura y Horas Operativas")
plt.show()

# Relación entre vibración y horas operativas
plt.figure(figsize=(8,5))
sns.scatterplot(x=registros_condiciones['Vibracion_mm_s'], y=registros_condiciones['Horas_Operativas'], alpha=0.5, color='r')
plt.xlabel("Vibración (mm/s)")
plt.ylabel("Horas Operativas")
plt.title("Relación entre Vibración y Horas Operativas")
plt.show()

# Creo dos variables más a partir de los datos de origen para luego calcular la vida útil estimada: horas operativas y recomendación de revisión
# df_final['Vida_Util_Estimada'] = df_final['Horas_Operativas'] / df_final['Horas_Recomendadas_Revision']

# Ordenar datos por equipo y fecha para calcular la diferencia temporal
df_final = df_final.sort_values(by=['ID_Equipo', 'Fecha_x'])
df_final['Tiempo_Hasta_Fallo'] = df_final.groupby('ID_Equipo')['Fecha_x'].diff().dt.days
df_final['Tiempo_Hasta_Fallo'].fillna(df_final['Tiempo_Hasta_Fallo'].median(), inplace=True)

# Guardar en Neon
import psycopg2
from sqlalchemy import create_engine

# Configuración de la conexión a PostgreSQL
# Esto mejor guardarlo en un .env
usuario = 'TestGens_owner'
contrasena = 'npg_mucgnV5OkD4t'
host = 'ep-fragrant-sun-a9a0jdor-pooler.gwc.azure.neon.tech'
puerto = '5432'
db = 'TestGens'

engine = create_engine(f'postgresql://{usuario}:{contrasena}@{host}:{puerto}/{db}')

# Guardar el dataset en PostgreSQL
df_final.to_sql("mantenimiento_equipos", engine, if_exists='replace', index=False)
