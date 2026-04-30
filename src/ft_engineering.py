

# librerias
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import   OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# =========================
# LIMPIEZA DE DATOS
# =========================
def limpiar_datos(df):
    """
    Aplica las reglas de limpieza definidas en el EDA
    """

    # Tipos de datos
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Validaciones básicas
    df = df[df["salario_cliente"] > 0]

    # Estandarizar texto
    df["tipo_credito"] = df["tipo_credito"].fillna("").astype(str).str.lower().str.strip()
    df["tipo_laboral"] = df["tipo_laboral"].fillna("").astype(str).str.lower().str.strip()
    
    # Tratamiento nulos
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col) # convertir "  " o " N/A " en "" o "N/A"
    valores_nulos = ["", " ", "NA", "N/A", "null", "NULL", "nan"] # unificación nulos
    df = df.replace(valores_nulos, np.nan)

    # Variable objetivo (ajustar según tus datos)
    df["Pago_atiempo"] = df["Pago_atiempo"].astype(int)

    return df


# =========================
# FEATURE ENGINEERING
# =========================
def crear_features(df):
    """
    Creación de nuevas variables
    """

    # Features de fecha
    if "fecha_prestamo" in df.columns:
        df["anio_prestamo"] = df["fecha_prestamo"].dt.year
        df["mes_prestamo"] = df["fecha_prestamo"].dt.month

    # Ratio deuda / ingreso (nivel de endeudamiento)
    if {"saldo_total", "salario_cliente"}.issubset(df.columns):
        df["ratio_deuda_ingreso"] = df["saldo_total"] / df["salario_cliente"]
        df["ratio_deuda_ingreso"] = df["ratio_deuda_ingreso"].replace([np.inf, -np.inf], np.nan)

    # Capacidad de pago 
    if {"cuota_pactada", "salario_cliente"}.issubset(df.columns):
        df["ratio_cuota_ingreso"] = df["cuota_pactada"] / df["salario_cliente"]
        df["ratio_cuota_ingreso"] = df["ratio_cuota_ingreso"].replace([np.inf, -np.inf], np.nan)

    # Deuda total
    if {"saldo_principal", "saldo_mora"}.issubset(df.columns):
        df["deuda_total"] = df["saldo_principal"] + df["saldo_mora"]

    # Total créditos reportados
    cols_creditos = [
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal"
    ]
    if set(cols_creditos).issubset(df.columns):
        df["total_creditos"] = df[cols_creditos].sum(axis=1)

    # Intensidad crediticia
    if {"cant_creditosvigentes", "huella_consulta"}.issubset(df.columns):
        df["intensidad_crediticia"] = (
            df["cant_creditosvigentes"] + df["huella_consulta"]
        )

    return df


# =========================
# PREPARACIÓN FINAL
# =========================
def preparar_datos_modelo(df):

    # 🔥 Manejo para entrenamiento vs producción
    if "Pago_atiempo" in df.columns:
        y = df["Pago_atiempo"]
        X = df.drop(columns=["Pago_atiempo"])
    else:
        y = None
        X = df.copy()

    # Separar variables numéricas y categóricas
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, num_cols, cat_cols


def preparar_datos_inferencia(df):

    # No hay variable objetivo en producción
    X = df.copy()

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return X, num_cols, cat_cols


# =========================
# EJECUCIÓN DEL FLUJO
# =========================

if __name__ == "__main__":

    df = cargar_datos()

    df_clean = limpiar_datos(df)
    df_clean = crear_features(df_clean)

    X, y, num_cols, cat_cols = preparar_datos_modelo(df_clean)
    
    print("Script ejecutado correctamente")

