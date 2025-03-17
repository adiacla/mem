import requests
import pandas as pd
from datetime import datetime, timedelta

# URL de la API
url = "https://www.simem.co/backend-files/api/PublicData"

# Parámetros iniciales para obtener datos recientes
params = {"datasetid": "ae3f23"}

# Realizar la solicitud GET a la API
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    
    if "result" in data and "records" in data["result"] and len(data["result"]["records"]) > 0:
        # Convertir los registros en DataFrame
        records = data["result"]["records"]
        df = pd.DataFrame(records)

        # Convertir la columna "Fecha" a datetime
        df["Fecha"] = pd.to_datetime(df["Fecha"])

        # Definir el rango de fechas (hoy y un mes atrás)
        hoy = datetime.today()
        inicio_mes_anterior = hoy - timedelta(days=30)

        # Filtrar los datos desde hace 30 días hasta hoy
        df_filtrado = df[(df["Fecha"] >= inicio_mes_anterior) & (df["Fecha"] <= hoy)]

        # Pivotear la tabla para que "CodigoVariable" sean columnas y "Valor" su contenido
        df_pivot = df_filtrado.pivot(index="Fecha", columns="CodigoVariable", values="Valor")

        # Resetear el nombre del índice
        df_pivot.columns.name = None

        # Mostrar el DataFrame transformado
        print(df_pivot)

        # Guardar en un archivo CSV
        df_pivot.to_csv("precios_escasez_ultimos_30_dias.csv", index=True)
        
    else:
        print("No hay datos disponibles en la API.")

else:
    print(f"Error en la solicitud: {response.status_code}")
