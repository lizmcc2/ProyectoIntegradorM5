import os 
import pandas as pd

def cargar_datos():
    
    # Ruta absoluta del directorio donde se encuentra el archivo (src)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Subir al nivel de donde se encuentra la carpeta de la base de datos 
    ruta_proyecto = os.path.dirname(ruta_actual)
    
    # Construir la ruta completa al CVS
    ruta_csv = os.path.join(ruta_proyecto, "Base_de_datos.csv")
    
    #Leemos los datos y la imprimimos
    df = pd.read_csv(ruta_csv)
    print(df)
    return df

if __name__ == "__main__":
    
    # Si el scrip se ejecuta directamente, carga los datos y muestra las primeras filas
    
    datos = cargar_datos()
    print (datos.head())
    print (datos.columns)

