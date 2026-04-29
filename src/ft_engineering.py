

# librerias
import pandas as pd
import numpy as np
from cargar_datos import cargar_datos
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import   OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Cargamos los datos
df= cargar_datos()

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

    # Paso 1: separar target y features
    X = df.drop("Pago_atiempo", axis=1) # Variables predictoras
    y = df["Pago_atiempo"]  # Variable objetivo

    # Paso 2: eliminar columnas no útiles
    X = X.drop(columns=["fecha_prestamo"], errors="ignore")
    X = X.drop(columns=["saldo_mora_codeudor"], errors="ignore")
    X = X.drop(columns=["tendencia_ingresos"], errors="ignore")
    X = X.drop(columns=["saldo_mora"], errors="ignore")
    X = X.drop(columns=["saldo_total"], errors="ignore")
    X = X.drop(columns=["saldo_principal"], errors="ignore")
    X = X.drop(columns=["puntaje"], errors="ignore")
    X = X.drop(columns=["puntaje_datacredito"], errors="ignore")
    X = X.drop(columns=["promedio_ingresos_datacredito"], errors="ignore")
    X = X.drop(columns=["cuota_pactada"], errors="ignore")
    X = X.drop(columns=["ratio_cuota_ingreso"], errors="ignore")
    X = X.drop(columns=["ratio_deuda_ingreso"], errors="ignore")
    X = X.drop(columns=["deuda_total"], errors="ignore")
    X = X.drop(columns=["creditos_sectorFinanciero"], errors="ignore")
    X = X.drop(columns=["creditos_sectorCooperativo"], errors="ignore")
    X = X.drop(columns=["creditos_sectorReal"], errors="ignore")
    X = X.drop(columns=["total_creditos"], errors="ignore")
    X = X.drop(columns=["intensidad_crediticia"], errors="ignore")
    
    # Paso 3: tipos de datos definidos desde el EDA
    num_cols = [
        'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente',
        'total_otros_prestamos', 'cuota_pactada', 'puntaje', 'puntaje_datacredito',
        'cant_creditosvigentes', 'huella_consulta', 'saldo_mora', 'saldo_total', 
        'saldo_principal', 'creditos_sectorFinanciero', 
        'creditos_sectorCooperativo', 'creditos_sectorReal', 'promedio_ingresos_datacredito'
    ]

    cat_cols = [
        'tipo_credito', 'tipo_laboral'
    ]

    # Validar columnas existentes
    num_cols = [col for col in num_cols if col in X.columns]
    cat_cols = [col for col in cat_cols if col in X.columns]

    return X, y, num_cols, cat_cols


# =========================
# EJECUCIÓN DEL FLUJO
# =========================

df_clean = limpiar_datos(df)
df_clean = crear_features(df_clean)

X, y, num_cols, cat_cols = preparar_datos_modelo(df_clean)

# =========================
# PIPELINE PRINCIPAL
# =========================

# Train/Test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# pipeline para transformar variables numericas 
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline para transformar variables categóricas
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# combinar los transformadores en ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# Pipeline completo + modelo
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Entrenamiento
pipeline.fit(X_train, y_train)

# Predicción
y_pred = pipeline.predict(X_test)

# Evaluación
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
