
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib 
import tensorflow as tf
from datetime import datetime, timedelta, date
from workalendar.america import Colombia
from pydataxm import *
#pip install scikit-learn

# Cargar el modelo y el scaler
model = tf.keras.models.load_model("pbolsa.keras")  # Usar el modelo correcto
scaler = joblib.load("scaler.pkl")  # Asegúrate de tener el scaler guardado


# Configurar la aplicación
st.title("Predicción de Precio de Bolsa de Energía")
st.subheader("Smart Region Lab")
st.image("https://estaticos.elcolombiano.com/binrepository/780x565/0c0/0d0/none/11101/BDII/mcuenta-de-servicios-4-39770279-20220406195018_44779049_20240402164929.jpg")


# ===========================
# 🔹 2️⃣ Cargar los Últimos 30 Días de Datos Normalizados
# ===========================
df_scaled = pd.read_csv("df_scaled.csv", index_col=0)

# ===========================
# 🔹 4️⃣ Entrada de Datos del Usuario
# ===========================
st.sidebar.header("Parámetros de entrada")
dia = st.sidebar.number_input("Día", min_value=1, max_value=31, value=17)
mes = st.sidebar.number_input("Mes", min_value=1, max_value=12, value=3)
año = st.sidebar.number_input("Año", min_value=2000, max_value=2030, value=2025)
precio_oil = st.sidebar.number_input("Precio del Petróleo", value=66.55)
precio_escasez = st.sidebar.number_input("Precio de escasez", value=770.55)
demanda_real = st.sidebar.slider("Demanda real", min_value=1e6, max_value=10e6, value=1915612.0, step=1.0)
capacidad_embalse = st.sidebar.slider("Capacidad embalse", min_value=1e9, max_value=2e10, value=17185800635.0, step=1.0)
irradiacion = st.sidebar.slider("Irradiación", min_value=100.0, max_value=1000.0, value=408.05, step=1.0)
temperatura = st.sidebar.slider("Temperatura", min_value=0.0, max_value=50.0, value=25.0, step=0.1)

# ===========================
# 🔹 5️⃣ Determinar si es Día Festivo
# ===========================
cal = Colombia()
fecha = datetime(año, mes, dia)
es_festivo = cal.is_holiday(fecha)

# ===========================
# 🔹 6️⃣ Crear la Nueva Muestra para Mañana
# ===========================
nueva_muestra = pd.DataFrame({
    "Holiday": [es_festivo],
    "Business_Day": [not es_festivo],
    "Precio_Oil": [precio_oil],
    "Precio_bolsa": [244],  # Valor a predecir
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

# ===========================
# 🔹 7️⃣ Normalizar la Nueva Muestra
# ===========================
nueva_muestra_scaled = scaler.transform(nueva_muestra)
nueva_muestra_scaled_df = pd.DataFrame(nueva_muestra_scaled, columns=df_scaled.columns)

# ===========================
# 🔹 8️⃣ Usar los Últimos 30 Días + Nueva Muestra
# ===========================
seq_length = 30
last_30_days = df_scaled.iloc[-seq_length:].copy()  # Tomar los últimos 30 días
last_30_days = pd.concat([last_30_days, nueva_muestra_scaled_df], ignore_index=True)  # Agregar nueva muestra
print(last_30_days)
# ===========================
# 🔹 9️⃣ Formatear los Datos para el Modelo
# ===========================
X_input = np.array(last_30_days[-seq_length:]).reshape(1, seq_length, df_scaled.shape[1])

# ===========================
# 🔹 🔟 Hacer la Predicción
# ===========================
precio_predicho_normalizado = model.predict(X_input)
precio_predicho_normalizado
# ===========================
# 🔹 1️⃣1️⃣ Desnormalizar el Resultado
# ===========================

# Crear un array temporal para desnormalizar correctamente
temp_array = np.zeros((1, df_scaled.shape[1]))  # Un array con ceros del mismo tamaño que los datos
temp_array[:, df_scaled.columns.get_loc("Precio_bolsa")] = precio_predicho_normalizado  # Solo rellenamos la columna a predecir

# Aplicar inverse_transform sobre toda la estructura
precio_predicho = scaler.inverse_transform(temp_array)[:, df_scaled.columns.get_loc("Precio_bolsa")]

# ===========================
# 🔹 1️⃣2️⃣ Mostrar el Resultado en Streamlit
# ===========================
st.markdown("### Predicción")
st.metric(label="📈 Predicción del Precio de Bolsa", value=f"{precio_predicho[0]:.2f}")



# ===========================
# 🔹 Pie de Página
# ===========================
st.markdown("---")
st.markdown("Derechos de Autor © Alfredo Diaz Claro")
