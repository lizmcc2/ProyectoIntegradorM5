
# librerias
import pandas as pd
from cargar_datos import cargar_datos
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import   OneHotEncoder
from sklearn.model_selection import train_test_split

# cargar datos
df= cargar_datos()

print(df.head())
print(df.columns)
print(df.describe())

# 1. identificar las features y el target