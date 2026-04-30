# PROYECTO RIESGO CREDITICIO

Una empresa financiera requiere predecir el comportamiento de Riesgo Crediticio de los nuevos usuarios, usando información histórica de créditos con el fin de aprender patrones y predecir comportamientos futuros.

---

# 📊 Avance 1: Informe de Procesamiento y Limpieza de Datos (EDA)

Se realizó un Análisis Exploratorio de Datos (EDA) enfocado en la limpieza, estandarización y caracterización de un dataset financiero.

---

## 1. Exploración e Identificación Inicial

Se cargó el dataset original y se identificó la estructura de las variables. Se detectó que, aunque el dataset reportaba pocos nulos iniciales, existían valores inconsistentes representados como texto ("N/A", "null", " ").

---

## 2. Unificación de Valores Nulos

Se procedió a estandarizar todos los valores faltantes utilizando la librería NumPy (`np.nan`).

- **Acción:** Se reemplazaron las cadenas vacías y etiquetas de texto nulo en todas las columnas.  
- **Resultado:** Se identificaron volúmenes significativos de nulos en variables clave como `promedio_ingresos_datacredito` (2,930 nulos) y `tendencia_ingresos` (2,932 nulos).

---

## 3. Conversión de Tipos de Datos

Se ajustaron los formatos técnicos para permitir cálculos matemáticos y optimizar la memoria:

- **Temporales:** `fecha_prestamo` se convirtió a formato `datetime64`.  
- **Numéricas:** Las variables financieras (saldos, ingresos, puntajes) se forzaron a tipo numérico.  
- **Categóricas:** Se transformaron etiquetas de texto a tipo `category` para facilitar la agrupación.

---

## 4. Tratamiento de Nulos (Imputación)

Para no perder el volumen de datos (especialmente el ~27% en ingresos), se aplicaron técnicas de imputación:

- **Variables Numéricas:** Se utilizó la **Mediana** (ej. `puntaje_datacredito`, saldos, ingresos) para evitar el sesgo de valores extremos.  
- **Variables Categóricas:** Se utilizó la **Moda** para la variable `tendencia_ingresos`.  
- **Ajuste Final:** Una vez eliminados los nulos, las columnas financieras se convirtieron a tipo Entero (`int64`) para una representación exacta.

---

## 5. Depuración de Variables (Eliminación)

Se realizó un análisis de relevancia donde se determinó la eliminación de:

- `saldo_mora_codeudor`: Identificada como irrelevante para el objetivo de predicción actual.

---

## 6. Caracterización Final de las Variables

El dataset quedó conformado por 22 columnas clasificadas de la siguiente manera:

- **Dicotómica:** `Pago_atiempo` (objetivo).  
- **Nominales (Politómicas):** `tipo_credito`, `tipo_laboral`.  
- **Ordinal:** `tendencia_ingresos`.  
- **Numéricas Discretas:** `cant_creditosvigentes`, `huella_consulta`, `puntaje`.  
- **Numéricas Continuas:** `salario_cliente`, `saldo_total`, `promedio_ingresos_datacredito`.

---

# Análisis Univariable: Variables Numéricas

Se evidencia que en alguna de las variables:

- **Salario Cliente (Dispersión Extrema):** Existe una diferencia entre el 50% (3.000.000) y el máximo (220.000.000). La asimetría es alta (43.77), lo que significa que hay pocos clientes con salarios gigantes que desvían el promedio.  
- **Saldos en Mora:** El `saldo_mora` y `saldo_mora_codeudor`, el 75% de los datos es 0. Significa que la distribución está muy concentrada en el mínimo.  
- **Edad del Cliente:** El promedio es 43 años, con un rango que va desde los 19 hasta los 123 años. El valor de 123 años podría ser un error de digitación.  
- **Kurtosis Elevada:** Las variables `promedio_ingresos_datacredito` (44.9) y `huella_consulta` (39.9) tienen una curtosis muy alta. Es decir, los datos están muy "puntiagudos" o concentrados en un solo valor.

---

## Histograma de las variables que presentan una distribución normal
![hTGM](https://i.postimg.cc/Dw5mWggM/Histograma-de-las-variables.png)



- `edad_cliente`: Tiene una distribución multimodal (varios picos) entre los 20 y 60 años. Hay un pequeño grupo aislado en los 120 años, lo que es un outlier extremo.  
- `puntaje`: Presenta una distribución sesgada a la derecha. Casi todos los clientes están concentrados en el valor máximo (cerca de 100).  
- `puntaje_datacredito`: Es la que mejor se comporta. Tiene una distribución normal (gaussiana) casi perfecta, excepto por el pequeño pico en 0.  
- `capital_prestado` y `cant_creditosvigentes`: Tienen asimetría positiva.  
- `plazo_meses`: Distribución discreta con picos en valores estándar (6, 12, 24, 36 y 60 meses).

---

## Análisis de Boxplots

![BXP](https://i.postimg.cc/8P001QQs/Analisis-de-Boxplots.png)


- `edad_cliente`: Se evidencian outliers de 121-123 años.  
- `puntaje_datacredito`: Muestra outliers en ambos extremos, especialmente en puntajes bajos.  
- `capital_prestado`: Gran cantidad de outliers superiores (hasta 40 millones).  
- `huella_consulta`: Algunos clientes presentan más de 25 consultas.

---

# Variables con Asimetría y Kurtosis

Debido a la alta asimetría de la variable salario (Skewness > 40), se recomienda una transformación logarítmica para estabilizar la varianza y mejorar el desempeño del modelo.

![KTS](https://i.postimg.cc/4yFBNt0W/asimetria-(skewness)-y-kurtosis.png)

---
Gráficas Izquierda: Los datos son prácticamente ilegibles. No es posible ver patrones.
Gráficas Derecha: los datos siguen una distribución unimodal (con un pico principal).

Reporte de Reducción de Asimetría con logaritmo:


- `cuota_pactada`: Original 3.79 → Log 0.40  
- `saldo_total`: Original 20.32 → Log -1.63  
- `saldo_principal`: Original 5.15 → Log -1.67  
- `creditos_sectorFinanciero`: Original 2.70 → Log -0.08  
- `creditos_sectorCooperativo`: Original 4.22 → Log 2.30  
- `creditos_sectorReal`: Original 3.16 → Log 0.63  
- `promedio_ingresos_datacredito`: Original 5.08 → Log -3.91  
- `salario_cliente`: Original 43.78 → Log -4.89  
- `total_otros_prestamos`: Original 38.46 → Log -3.61  

Se destaca que variables como `cuota_pactada` (0.40) y `creditos_sectorFinanciero` (-0.08) ahora tienen una distribución casi perfecta.

---

# Análisis Univariable: Variables Categóricas

## 1. Distribución de Pago_atiempo

•	fuerte desbalance: 95.3% (10,252 clientes) cumplen con sus obligaciones, frente a un pequeño 4.7% (511 clientes) en mora.

•	El algoritmo tendrá que ser muy preciso para aprender a detectar ese 4.7% entre tanta gente que sí paga.
 

**Nota:** Existe un fuerte desbalance de clases.

---

## 2. Distribución de tipo_credito

•	El crédito tipo "4" es el producto estrella con el 72.0% de la participación.

•	Los tipos 7 y 68 son prácticamente testimoniales (0.0%). Esto sugiere que podrías agrupar las categorías minoritarias en un solo grupo llamado "Otros" en el futuro para evitar que el modelo se confunda con datos tan escasos.


---

## 3. Distribución de tipo_laboral

•	La mayoría de los clientes son mpleados (62.8%), pero el grupo de Independientes (37.2%) es bastante sólido y representativo.

•	Esta es la variable con la distribución más equilibrada, lo que la hace perfecta para el análisis bivariable.
 

---

# Análisis Bivariable

## Puntaje vs Pago

![PY](https://i.postimg.cc/mrNj1KqS/Puntaje-datacredito-vs-pagoa-atiempo.png)

- Diferencia de medianas: 24 puntos  
- Mayor puntaje → mayor probabilidad de pago  
- Puntaje 0 es una anomalía importante  

---

## Tipo Laboral vs Pago
![PK](https://i.postimg.cc/DyJcPSHV/Tipo-Laboral-vs-Pago-a-Tiempo.png)

- Independientes: 5.5% mora  
- Empleados: 4.3% mora  

---

# Inclusión de Nuevas Variables

![GH](https://i.postimg.cc/9QytsvW3/Inclusion-de-nuevas-variables.png)

- **Deuda Total:** Mayor en clientes cumplidos  
- **Uso del Crédito:** Alta presencia de outliers  

---

# Análisis Multivariable
![AM](https://i.postimg.cc/26944Yx8/Analisis-Multivariable.png)

Hallazgos Estratégicos (Análisis Multivariable Final)

1.	La Variable Predictora Estrella: puntaje
la variable puntaje muestra una correlación de 0.92 con Pago_atiempo. Indicando que es extremadamente preciso para predecir quién paga a tiempo.

2.	Éxito de la Ingeniería de Características: deuda_total_actual_log
Tu nueva variable deuda_total_actual_log muestra una correlación muy fuerte (0.86) con total_others_prestamos_log. confirmando que la deuda externa es el componente principal del endeudamiento total de los clientes. 

3.	Relación Estructural: cuota_pactada_log vs capital_prestado_log
Mantienen una correlación de 0.77. Al aplicar logaritmos a ambas, la relación se volvió más lineal, lo cual es optimo para trabajar con algoritmos como la Regresión Logística.

4.	Independencia del puntaje_datacredito
Tiene correlaciones muy bajas (cercanas a 0.05 o -0.14) con las variables de saldo y salario. significa que el puntaje de datacredito aporta información nueva y diferente que no se puede deducir simplemente mirando cuánto gana o cuánto debe el cliente.

5.	Redundancia Detectada (Multicolinealidad)
saldo_total_log y saldo_principal_log tienen 0.82.
cant_creditosvigentes y creditos_sectorFinanciero tienen 0.79.
Al estar tan relacionadas, podrías considerar usar solo una de cada pareja para evitar que el modelo se "sobreajuste" o se confunda con datos redundantes.

---

# Conclusiones Generales

## 1. Calidad de Datos

- Eliminación de nulos  
- Transformaciones logarítmicas clave  

## 2. Factores de Riesgo

- Independientes = mayor riesgo  
- Ingresos no determinan mora  

## 3. Comportamiento Estructural

- Puntaje = variable central  
- Dataset desbalanceado (95.3% vs 4.7%)

---

# Recomendaciones

1.	Selección de Variables (Feature Selection): Utilizar preferiblemente las versiones logarítmicas de las variables financieras y mantener las nuevas variables de ingeniería (ratio_uso_credito), ya que presentan relaciones más lineales y separables.

2.	Gestión de Redundancia: Dado que el saldo_total y el saldo_principal tienen una correlación de 0.82, se recomienda utilizar solo uno de ellos (o su ratio) para evitar ruidos de multicolinealidad.

3.	Tratamiento de Outliers: Se recomienda tratar de forma especial los registros con edad superior a 100 años identificados en el EDA, ya que actúan como ruido estadístico que no representa la realidad biológica de la cartera.

4.	Balanceo de Clases: Es imperativo aplicar técnicas de remuestreo (como SMOTE) o ajustar los pesos de las clases en el algoritmo para compensar la minoría de datos en mora (4.7%).
  


---

# 📊 AVANCE 2: Ingeniería de Características y Modelos Supervisados

## ⚙️ Feature Engineering (Ingeniería de Características)

En esta etapa del proyecto se realizó un proceso de ingeniería de características con el objetivo de mejorar la calidad de los datos y construir variables más representativas para el modelo de clasificación.

Este proceso se dividió en tres componentes principales:

### 1. Limpieza de Datos

Se aplicaron transformaciones del EDA para asegurar la consistencia y calidad del dataset:

- Conversión de tipos de datos: La variable fecha_prestamo fue convertida a formato datetime para permitir la extracción de información temporal.  
- Eliminación de duplicados: Se eliminaron registros duplicados para evitar sesgos en el entrenamiento del modelo.  
- Validaciones básicas: Se filtraron registros con salario_cliente <= 0, ya que no representan casos válidos para análisis financiero.  

Estandarización de variables categóricas Se aplicó:

- Conversión a minúsculas  
- Eliminación de espacios en blanco  
- Manejo de valores nulos  

Tratamiento de valores nulos: Se unificaron diferentes representaciones de valores faltantes ("NA", "null", " ", etc.) a NaN.

Variable objetivo: La variable Pago_atiempo fue convertida a formato numérico (0 y 1) para su uso en modelos de clasificación.

---

### 2. Creación de Nuevas Variables

Se generaron nuevas variables derivadas con el objetivo de capturar mejor el comportamiento financiero de los clientes:

**Variables temporales:** A partir de fecha_prestamo: anio_prestamo, mes_prestamo  
Estas variables permiten capturar posibles patrones estacionales.

**Indicadores financieros:** Se construyeron métricas claves utilizadas comúnmente en análisis de riesgo crediticio:

🔹 Ratio deuda / ingreso  
ratio_deuda_ingreso = saldo_total / salario_cliente  
Mide el nivel de endeudamiento del cliente.

🔹 Ratio cuota / ingreso  
ratio_cuota_ingreso = cuota_pactada / salario_cliente  
Representa la capacidad de pago mensual del cliente.

🔹 Deuda total  
deuda_total = saldo_principal + saldo_mora  
Indica la carga total de deuda del cliente.

🔹 Total de créditos  
total_creditos = suma de créditos en diferentes sectores  
Mide la exposición crediticia total del cliente.

🔹 Intensidad crediticia  
intensidad_crediticia = cant_creditosvigentes + huella_consulta  
Refleja el nivel de actividad financiera reciente del cliente.

Manejo de valores extremos: Se controlaron valores infinitos generados por divisiones, reemplazándolos por NaN.

---

### 3. Selección y Eliminación de Variables

Una parte clave del proceso fue la eliminación de variables, basada en criterios técnicos:

Variables eliminadasSe eliminaron variables como:

- puntaje, puntaje_datacredito  
- saldo_total, saldo_mora, saldo_principal  
- promedio_ingresos_datacredito  
- cuota_pactada  
- Variables derivadas como:  
  - ratio_deuda_ingreso  
  - ratio_cuota_ingreso  
  - deuda_total  
  - total_creditos  
  - intensidad_crediticia  

Estas variables fueron eliminadas por:

**Posible data leakage**

- Algunas variables contienen información demasiado directa sobre el resultado (Pago_atiempo)  
- Esto genera modelos artificialmente perfectos  

**Sobreajuste (overfitting)**

- El modelo aprende relaciones no generalizables  

**Redundancia**

- Variables altamente correlacionadas entre sí  

---

### 4. Variables Finales del Modelo

Después del proceso de selección, el modelo se entrena con:

**Variables numéricas:**

- capital_prestado  
- plazo_meses  
- edad_cliente  
- salario_cliente  
- total_otros_prestamos  
- cant_creditosvigentes  
- huella_consulta  

**Variables categóricas:**

- tipo_credito  
- tipo_laboral  

**Variables temporales:**

- anio_prestamo  
- mes_prestamo  

---

Finalmente, es importante mencionar que se realizó un ejercicio comparativo de dos enfoques:

1. Manteniendo todas las variables originales  
2. Eliminando variables con posible fuga de información  

En el primer caso, los modelos alcanzaron métricas perfectas (Accuracy y ROC-AUC de 1.00), lo cual indica la presencia de data leakage, ya que variables como el puntaje crediticio y saldos de mora contienen información directamente relacionada con el resultado del pago.

En el segundo caso, al eliminar las variables mencionadas, el rendimiento del modelo disminuyó significativamente, evidenciando un escenario más realista, pero también exponiendo un fuerte desbalance de clases que afecta la capacidad del modelo para detectar clientes morosos.

Se prioriza el modelo construido sin dichas variables, ya que representa mejor un escenario real de predicción, aunque implique un menor rendimiento.

---

## Conclusión

El proceso de feature engineering fue clave para:

- Mejorar la calidad de los datos  
- Evitar data leakage  
- Reducir el sobreajuste  
- Construir un conjunto de variables más realista  

Sin embargo, esta depuración también hizo evidente un desafío importante; la dificultad del modelo para identificar la clase minoritaria (clientes en riesgo), lo cual está directamente relacionado con el desbalance del dataset.

---

# ⚙️ Modelado Supervisado y Evaluación de Modelos

En esta etapa del proyecto se implementaron distintos modelos de clasificación supervisada con el objetivo de predecir la variable objetivo Pago_atiempo, la cual indica si un cliente cumple o no con sus obligaciones crediticias.

---

## Desbalance de la Variable Objetivo

Uno de los principales hallazgos del análisis exploratorio fue el fuerte desbalance en la variable objetivo:

- Clase 1 (Pago a tiempo): 95.25%  
- Clase 0 (No paga a tiempo): 4.75%  

Este desbalance implica que un modelo puede alcanzar una alta precisión global simplemente prediciendo siempre la clase mayoritaria, sin realmente aprender patrones útiles para identificar clientes en riesgo.

Por esta razón, la métrica de accuracy no es suficiente, y se prioriza el análisis de métricas como:

- Recall (especialmente de la clase 0)  
- Precision  
- F1-score  
- ROC-AUC  

---

## Pipeline de Modelado

Se implementó un flujo de trabajo utilizando Pipelines de Scikit-learn, lo cual permite integrar de forma ordenada y reproducible:

1. Imputación de valores faltantes  
2. Escalado de variables numéricas  
3. Codificación de variables categóricas (One-Hot Encoding)  
4. Entrenamiento del modelo  

Esto garantiza que todas las transformaciones se apliquen correctamente tanto en entrenamiento como en predicción, evitando data leakage.

---

## Modelos Evaluados

Se entrenaron los siguientes modelos:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  

Adicionalmente, en el caso de Logistic Regression se utilizó el parámetro:

class_weight="balanced"

para mitigar el efecto del desbalance de clases.

---

## Ajuste del Threshold (Umbral de Decisión)

Por defecto, los modelos clasifican usando un threshold de 0.5 sobre la probabilidad predicha. Sin embargo, dado el desbalance del dataset, se evaluaron distintos valores de threshold (0.5, 0.4, 0.3, 0.2) con el objetivo de mejorar la detección de la clase minoritaria.

Reducir el threshold permite:

- Aumentar el recall de la clase 0 (detectar más clientes en riesgo)  
- A costa de disminuir la precisión y la accuracy  

Este ajuste es clave en problemas de riesgo crediticio, donde es más costoso no detectar un cliente riesgoso que generar falsos positivos.

---

## Resultados y Análisis

### Evaluación de Modelos Supervisados

Se entrenaron tres modelos supervisados: Logistic Regression, Random Forest y Gradient Boosting, utilizando pipelines para garantizar la correcta transformación de variables numéricas y categóricas.

Los resultados evidencian un fuerte desbalance en la variable objetivo, donde la clase mayoritaria (clientes que pagan a tiempo) representa aproximadamente el 95% de los datos. Esto impacta significativamente el desempeño de los modelos, especialmente en la capacidad de identificar la clase minoritaria (clientes que no pagan a tiempo).

Aunque todos los modelos presentan métricas de accuracy superiores al 95%, esta métrica resulta engañosa en contextos desbalanceados. En particular:

- Random Forest no logra identificar ningún caso de la clase 0 (recall = 0.00)  
- Gradient Boosting presenta un desempeño marginal en la clase minoritaria (recall ≈ 0.01)  
- Logistic Regression logra el mejor equilibrio, alcanzando un recall cercano a 0.56 para la clase 0  

Adicionalmente, se evaluó el impacto del threshold de decisión sobre las predicciones del modelo. Se observó que:

- Umbrales más altos (0.5) favorecen la detección de la clase minoritaria  
- Umbrales más bajos (0.2 - 0.3) favorecen la clase mayoritaria, reduciendo significativamente la capacidad de detectar clientes en riesgo  

![cm](https://i.postimg.cc/1zmBwsqd/comparacion-models.png)


En este contexto, Logistic Regression se selecciona como el modelo más adecuado, ya que es el único capaz de identificar de manera significativa la clase minoritaria, lo cual es crítico en un problema de riesgo crediticio.

---

## Conclusión

El principal desafío radica en el alto desbalance de la variable objetivo, lo cual dificulta que los modelos identifiquen correctamente la clase minoritaria.

Entre los modelos evaluados:

- Logistic Regression fue el más adecuado para este caso  
- Especialmente al combinar:  
  - class_weight="balanced"  
  - Ajuste del threshold  

Esto permitió mejorar la capacidad del modelo para detectar clientes con riesgo de incumplimiento, lo cual es crítico en un contexto de negocio.

---

## Interpretación desde el Negocio

En el contexto crediticio:

- Clase 0 → Cliente que NO paga a tiempo (riesgo)  
- Clase 1 → Cliente que sí paga a tiempo  

El objetivo del modelo es reducir el riesgo de otorgar crédito a clientes que no pagarán.

Por ello, se prioriza:

- Maximizar el recall de la clase 0  
- Incluso si esto implica un aumento en falsos positivos  

# 📊 Avance 3: Monitoreo y Detección de Data Drift

El objetivo de este avance es implementar un sistema de monitoreo proactivo para el modelo de riesgo crediticio. La detección de Data Drift (desplazamiento de datos) es crítica para garantizar que las predicciones del modelo sigan siendo confiables a medida que el perfil de los solicitantes cambia con el tiempo.

---

## 2. Metodología de Monitoreo

Se ha desarrollado un tablero interactivo en Streamlit que compara dos estados de la información:

- Dataset de Referencia (Baseline): Datos históricos utilizados durante la fase de entrenamiento y validación.  
- Dataset de Monitoreo (Actual): Datos capturados en tiempo real (simulados mediante logs) que ingresan al modelo para predicción.  

Para asegurar la consistencia, ambos datasets atraviesan el mismo pipeline de ft_engineering (limpieza, imputación y creación de variables) antes de ser comparados.

---

## 3. Métricas Estadísticas Implementadas

Se seleccionaron métricas específicas según la naturaleza de cada variable:

### A. Variables Numéricas

Para variables como capital_prestado o plazo_meses, se aplican:

- Kolmogorov-Smirnov (KS Test): Evalúa si las distribuciones acumuladas de ambos datasets son iguales. Un p-value < 0.05 rechaza la hipótesis de igualdad, confirmando el drift.  

- Population Stability Index (PSI): Mide la estabilidad de la población.  
  - $PSI < 0.1$: Cambio insignificante.  
  - $0.1 \leq PSI < 0.25$: Cambio moderado (Alerta preventiva).  
  - $PSI \geq 0.25$: Cambio significativo (Requiere acción).  

- Jensen-Shannon Divergence: Mide la similitud entre distribuciones de probabilidad, proporcionando una métrica de distancia normalizada entre 0 y 1.

### B. Variables Categóricas

- Chi-Cuadrado: Se utiliza para variables como tipo_credito o tipo_laboral para determinar si las frecuencias de las categorías han cambiado significativamente.

---

## 4. Análisis de Visualización y Alertas

El sistema implementa una lógica de semáforo (Traffic Light System) para facilitar la toma de decisiones:

1. Estado Estable (Verde): La distribución se mantiene constante.  
2. Advertencia (Amarillo): Se detectan cambios ligeros que sugieren observación cercana.  
3. Drift Detectado (Rojo): Desviación estadística significativa que invalida las asunciones del modelo original.  

Se incluye un análisis temporal mediante gráficos de tendencia para observar la evolución del PSI global, permitiendo identificar si el drift es una anomalía puntual o una tendencia estructural del mercado.

---

## 5. Resultados y Recomendaciones Técnicas

- Metricas y Visualización
![da](https://i.postimg.cc/4NHSxf6j/monitoreo-data-drift-1.png)

- Analisis temporal
![de](https://i.postimg.cc/7Lws9SpN/monitoreo-data-drift-2.png)

- Recomendaciones
![dt](https://i.postimg.cc/QtM6GhNM/monitoreo-data-drift-3.png)

Acciones recomendadas según el estado detectado:

- Re-entrenamiento (Retraining): Si más del 20% de las variables clave muestran drift crítico.  
- Ajuste de Thresholds: Revisar los puntos de corte de decisión si el perfil de riesgo de la población se ha desplazado pero la relación con el target sigue siendo similar.  
- Auditoría de Ingesta: Verificar si el drift es causado por cambios en la calidad de los datos de entrada.  