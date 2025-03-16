import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from datetime import datetime, timedelta, date
from workalendar.america import Colombia
from objetoAPI import request_data

# Cargar el modelo y el scaler
model = tf.keras.models.load_model("pbolsa.keras")  # Usar el modelo correcto
scaler = joblib.load("scaler.pkl")  # Asegúrate de tener el scaler guardado

# Obtener datos históricos desde la API
df_pbolsa = request_data(
    "PrecBolsNaci",
    "Sistema",
    date(2022, 1, 1),
    date.today()
)
names = [hour for hour in df_pbolsa.columns][2:26]
df_pbolsa['Daily_Average'] = df_pbolsa[names].mean(axis=1)

# Configurar la aplicación
st.title("Predicción de Precio de Bolsa de Energía")
st.subheader("Smart Region Lab")
st.image("https://estaticos.elcolombiano.com/binrepository/780x565/0c0/0d0/none/11101/BDII/mcuenta-de-servicios-4-39770279-20220406195018_44779049_20240402164929.jpg")

# Sidebar para entradas del usuario
st.sidebar.header("Parámetros de entrada")
dia = st.sidebar.number_input("Día", min_value=1, max_value=31, value=15)
mes = st.sidebar.number_input("Mes", min_value=1, max_value=12, value=3)
año = st.sidebar.number_input("Año", min_value=2000, max_value=2030, value=2024)
precio_oil = st.sidebar.number_input("Precio del Petróleo", value=85.0)
precio_escasez = st.sidebar.number_input("Precio de escasez", value=800.0)
demanda_real = st.sidebar.slider("Demanda real", min_value=5e6, max_value=15e6, value=9.1e6)
capacidad_embalse = st.sidebar.slider("Capacidad embalse", min_value=1e10, max_value=2e10, value=1.75e10)
irradiacion = st.sidebar.slider("Irradiación", min_value=100.0, max_value=1000.0, value=420.0)
temperatura = st.sidebar.slider("Temperatura", min_value=0.0, max_value=50.0, value=28.0)

# Determinar si es día festivo
cal = Colombia()
fecha = datetime(año, mes, dia)
es_festivo = cal.is_holiday(fecha)

# Mostrar valores de entrada
st.markdown("### Estos valores se van a usar para predecir")
st.write(f"- **Día a predecir:** {fecha.strftime('%Y-%m-%d')}")
st.write(f"- **Día laboral o festivo:** {'Festivo' if es_festivo else 'Laboral'}")
st.write(f"- **Precio del Petróleo:** {precio_oil}")
st.write(f"- **Precio de escasez:** {precio_escasez}")
st.write(f"- **Demanda real:** {demanda_real}")
st.write(f"- **Capacidad embalse:** {capacidad_embalse}")
st.write(f"- **Irradiación:** {irradiacion}")
st.write(f"- **Temperatura:** {temperatura}")

# Crear muestra para predicción
nueva_muestra = pd.DataFrame({
    "Holiday": [es_festivo],
    "Business_Day": [not es_festivo],
    "Precio_Oil": [precio_oil],
    "Precio_bolsa": [0],
    "Precio_escasez": [precio_escasez],
    "Demanda_real": [demanda_real],
    "Capacidad_embalse": [capacidad_embalse],
    "Irradiacion": [irradiacion],
    "Temperatura": [temperatura],
    "Month_sin": [np.sin(2 * np.pi * mes / 12)],
    "Month_cos": [np.cos(2 * np.pi * mes / 12)],
    "Day_sin": [np.sin(2 * np.pi * dia / 31)],
    "Day_cos": [np.cos(2 * np.pi * dia / 31)]
})

# Asegurar que las columnas coincidan con el entrenamiento
columnas=['Holiday', 'Business_Day', 'Precio_Oil', 'Precio_bolsa',
          'Precio_escasez', 'Demanda_real', 'Capacidad_embalse', 'Irradiacion',
          'Temperatura', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']

df_nueva_muestra = nueva_muestra[columnas]

# Normalizar los datos
df_nueva_muestra_scaled = pd.DataFrame(scaler.transform(df_nueva_muestra), columns=columnas)

# Reestructurar la muestra para que tenga la forma esperada por el modelo
X_nueva_muestra = np.expand_dims(df_nueva_muestra_scaled.values, axis=0)  # (1, num_features)

# Hacer la predicción
precio_predicho_normalizado = model.predict(X_nueva_muestra)

# Crear un array temporal para desnormalizar correctamente
temp_array = np.zeros((1, len(columnas)))
temp_array[:, columnas.index("Precio_bolsa")] = precio_predicho_normalizado

# Desnormalizar la predicción
precio_predicho = scaler.inverse_transform(temp_array)[:, columnas.index("Precio_bolsa")]

# Mostrar el resultado de la predicción
st.markdown("### Predicción")
st.metric(label="📈 Predicción del Precio de Bolsa", value=f"{precio_predicho[0]:.2f}")

# Pie de página
st.markdown("---")
st.markdown("Derechos de Autor © Alfredo Diaz Claro")
