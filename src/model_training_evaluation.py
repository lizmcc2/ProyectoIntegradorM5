# librerias

import pandas as pd
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

for nombre, modelo in modelos.items():

    print("\n" + "="*50)
    print(f"MODELO: {nombre}")
    print("="*50)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", modelo)
    ])

    # Entrenar
    pipeline.fit(X_train, y_train)

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

            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            print("\nMatriz de Confusión:")
            print(confusion_matrix(y_test, y_pred))

    except:
        print("Este modelo no soporta predict_proba")