# librerias
import pandas as pd
import numpy as np
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

#inicialización de la aplicación
app = FastAPI(
    title= "API de predicción de pago a tiempo",
    description= "Desplegar un modelo de machine learning para predecir si un cliente pagara a tiempo o no"
    version= "1.0.0"
)

# Cargamos el modelo entrenado (.pkl, .joblib)
try:
    # Cargamos el modelo desde el archivo model.pkl
    with open ("../models/models.pkl"."rb") as f:
        modelo = pickle.load(f)
        
        print("Modelo cargado exitosamente")
    
except Exception as e:  
    print (f"error al cargar modelo: {e}")
    modelo=None
    

# Crear endpoint de saludo
@app.get("/saludo")

def saludo():
    return{"mensaje": "Hola, esta API esta corriendo correctamente..."}

# Crear un endpoint para hacer predicciones 
@app.post("/predict")

def predict_batch(input_data: dict):
    if modelo is None:
        return "El modelo no pudo ser cargado. Revisa los log del servidor"
    
    try:
        
        # Modelo cargado y listo para hacer predicciones
        return "El modelo esta cargado y listo para hacer predicciones"
        
    except: Exception as e:
        return f"Error al hacer las predicciones (e)"
    
# Cargar el script
if__name__= "__main__"
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=true)