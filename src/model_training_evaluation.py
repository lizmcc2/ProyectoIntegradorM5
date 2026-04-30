# librerias

import pandas as pd
import matplotlib.pyplot as plt
from cargar_datos import cargar_datos
from ft_engineering import limpiar_datos, crear_features, preparar_datos_modelo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# Cargar datos
df = cargar_datos()
df = limpiar_datos(df)
df = crear_features(df)

X, y, num_cols, cat_cols = preparar_datos_modelo(df)


# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# PREPROCESSOR Preprocessor
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# FUNCIONES AUXILIARES
# =========================

def build_model(model, preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def summarize_classification(nombre, threshold, y_test, y_pred, y_proba=None):

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

    resultados = {
        "Modelo": nombre,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall_0": recall_score(y_test, y_pred, pos_label=0),
        "Recall_1": recall_score(y_test, y_pred, pos_label=1),
        "F1_0": f1_score(y_test, y_pred, pos_label=0),
        "F1_1": f1_score(y_test, y_pred, pos_label=1),
    }

    if y_proba is not None:
        resultados["ROC_AUC"] = roc_auc_score(y_test, y_proba)

    return resultados

# MODELOS

modelos = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier()
}

# Entrenamiento y Evaluación

thresholds = [0.5, 0.4, 0.3, 0.2]

resultados = []

pipelines_entrenados = {}

for nombre, modelo in modelos.items():

    print("\n" + "="*50)
    print(f"MODELO: {nombre}")
    print("="*50)

    pipeline = build_model(modelo, preprocessor)

    # Entrenar
    pipeline.fit(X_train, y_train)
    
    pipelines_entrenados[nombre] = pipeline

    # Probabilidades
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # ROC-AUC
        roc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {roc:.4f}")

        # Evaluar diferentes thresholds
        for t in thresholds:
            print("\n" + "-"*40)
            print(f"THRESHOLD = {t}")
            print("-"*40)

            y_pred = (y_proba >= t).astype(int)
            
            resumen = summarize_classification(
                nombre,
                t,
                y_test,
                y_pred,
                y_proba
            )

            resultados.append(resumen)

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            print("\nMatriz de Confusión:")
            print(confusion_matrix(y_test, y_pred))

    except:
        print("Este modelo no soporta predict_proba")

df_resultados = pd.DataFrame(resultados)

# SELECCIÓN DEL MEJOR MODELO

df_best_model = df_resultados.groupby("Modelo")["Recall_0"].mean().sort_values(ascending=False)

print("\n" + "="*50)
print("MEJOR MODELO SELECCIONADO")
print("="*50)
print(df_best_model)

best_model_name = df_best_model.index[0]

print(f"\nMejor modelo: {best_model_name}")

print("\n" + "="*50)
print("TABLA RESUMEN FINAL")
print("="*50)
print(df_resultados)

df_plot = df_resultados.sort_values("Recall_0", ascending=False).drop_duplicates("Modelo")
df_plot = df_plot.set_index("Modelo")[["Recall_0", "Recall_1", "ROC_AUC"]]

df_plot.plot(kind="bar")

plt.title("Comparación de Modelos")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(True)

plt.show()

print("\nMejores configuraciones por modelo:")
print(df_plot)

import pickle
import os

# Crear carpeta models si no existe
os.makedirs("models", exist_ok=True)

# Obtener el mejor pipeline
best_pipeline = pipelines_entrenados[best_model_name]

# Guardar modelo
with open("models/modelo.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print("\nModelo guardado en models/modelo.pkl")