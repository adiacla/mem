from docx import Document
import requests
import json
import pandas as pd

document_text = ""
api_text = ""


def cargar_documento(archivo):
    """Load and process a .docx document"""
    global document_text
    doc = Document(archivo.name)
    document_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return [
        {
            "role": "assistant",
            "content": "📄 Documento cargado exitosamente. ¿Qué desea preguntar?",
        }
    ]


def obtener_datos_api(url):
    """Retrieve and process API data"""
    global api_text
    try:
        response = requests.get(url)
        print(f"📡 API solicitada: {url}")
        print(f"📡 Código de respuesta: {response.status_code}")

        if response.status_code == 200:
            api_data = response.json()

            if "result" in api_data and "records" in api_data["result"]:
                records = api_data["result"]["records"]

                df = pd.DataFrame(records)

                # Convertir la columna "Fecha" a datetime
                df["Fecha"] = pd.to_datetime(df["Fecha"])

                # Pivotear la tabla: "CodigoVariable" serán las columnas, "Valor" el contenido
                df_pivot = df.pivot(
                    index="Fecha", columns="CodigoVariable", values="Valor"
                ).reset_index()

                # Cambiar el formato de la fecha a "DD-MM-YYYY"
                df_pivot["Fecha"] = df_pivot["Fecha"].dt.strftime("%d-%m-%Y")

                df_pivot.index.name = "index"

                # Convertir el DataFrame a texto legible
                pivot_text = df_pivot.to_string()

                name = api_data["result"].get("name", "Sin nombre")
                description = (
                    api_data["result"]
                    .get("metadata", {})
                    .get("description", "Sin descripción")
                )

                api_text = (
                    f"🔹 **{name}**\n"
                    f"📄 {description}\n\n"
                    f"🌐 **Información cargada desde la API:** {url}\n\n"
                    f"📊 **Conjuntos de dato del API:**\n```\n{pivot_text}\n```"
                )

                print(f"✅ API cargada: {api_text}")
                return [
                    {
                        "role": "assistant",
                        "content": "📄 API cargada exitosamente. ¿Qué desea preguntar?",
                    }
                ]
            else:
                print("❌ Error: No se encontraron registros en la API.")
                return [
                    {
                        "role": "assistant",
                        "content": "❌ Error: No se encontraron registros en la API.",
                    }
                ]
        else:
            print(f"❌ Error en la API: {response.status_code}")
            return [
                {
                    "role": "assistant",
                    "content": f"❌ Error en la API: {response.status_code}",
                }
            ]
    except Exception as e:
        print(f"❌ Error al conectar con la API: {str(e)}")
        return [
            {
                "role": "assistant",
                "content": f"❌ Error al conectar con la API: {str(e)}",
            }
        ]


def get_document_text():
    """Return the global document_text variable"""
    global document_text
    return document_text


def get_api_text():
    """Return the global api_text variable"""
    global api_text
    return api_text
