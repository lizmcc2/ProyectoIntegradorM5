# =========================
# LIBRERÍAS
# =========================
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ft_engineering import preparar_datos_inferencia


# =========================
# INICIALIZAR API
# =========================
app = FastAPI(
    title="API de predicción de pago a tiempo",
    description="Desplegar un modelo de ML para predecir si un cliente pagará a tiempo o no",
    version="1.0.0"
)


# =========================
# ENDPOINT RAÍZ
# =========================
@app.get("/")
def home():
    return {"mensaje": "API de predicción funcionando 🚀"}


# =========================
# RUTA ROBUSTA DEL MODELO
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/modelo.pkl")


# =========================
# CARGAR MODELO
# =========================
try:
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    print("✅ Modelo cargado exitosamente")

except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    modelo = None


# =========================
# ESQUEMA DE ENTRADA
# =========================
class InputData(BaseModel):
    data: list


# =========================
# ENDPOINT DE SALUDO
# =========================
@app.get("/saludo")
def saludo():
    return {"mensaje": "Hola, esta API está corriendo correctamente"}


# =========================
# ENDPOINT DE PREDICCIÓN
# =========================
@app.post("/predict")
def predict(input_data: InputData):

    if modelo is None:
        return {"error": "El modelo no está cargado"}

    try:
        # Convertir JSON a DataFrame
        df = pd.DataFrame(input_data.data)

        # SOLO INFERENCIA (sin target)
        X, _, _ = preparar_datos_inferencia(df)

        # Predicción
        pred = modelo.predict(X)

        return {"predictions": pred.tolist()}

    except Exception as e:
        return {"error": str(e)}


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)
