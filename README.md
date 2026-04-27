# ProyectoIntegradorM5
Proyecto Integrador del Modulo 5




En el EDA se identifico los siguiente proporción en valores nulos, de los cuales:

- las variables tendencia_ingresos y promedio_ingresos_datacredito presentan aproximadamente un 27% de valores faltantes, lo que indica una posible baja disponibilidad o calidad de esta información por lo que no se usara para los analisis.

- Las  variables como puntaje_datacredito, saldo_mora y saldo_total presentan porcentajes de nulos inferiores al 2%, lo cual no representa un problema significativo para el análisis.

- La variable saldo_mora_codeudor presenta un nivel intermedio de valores faltantes (~5.5%), lo que requiere evaluación adicional antes de decidir su tratamiento.


| Columna                         | % Nulos | Interpretación |
| ------------------------------- | ------- | -------------- |
| `tendencia_ingresos`            | ~27.24% | ⚠️ Alto        |
| `promedio_ingresos_datacredito` | ~27.22% | ⚠️ Alto        |
| `saldo_mora_codeudor`           | ~5.48%  | 🟡 Medio       |
| `saldo_principal`               | ~3.76%  | 🟡 Bajo        |
| `saldo_total`                   | ~1.45%  | 🟢 Muy bajo    |
| `saldo_mora`                    | ~1.45%  | 🟢 Muy bajo    |
| `puntaje_datacredito`           | ~0.05%  | 🟢 Mínimo      |


# Caracterización de los Datos

El dataset contiene variables de tipo numérico, categórico y temporal, cada una con características específicas relevantes para el análisis.

Las variables numéricas corresponden principalmente a valores financieros y cuantitativos, como montos de crédito, saldos, ingresos y puntajes, lo que permite realizar análisis estadísticos y modelado predictivo.

Se identificaron variables categóricas de distintos tipos:

- Nominales: como tipo_laboral y tendencia_ingresos, que representan categorías sin un orden inherente.
- Ordinales: como tendencia_ingresos, con orden lógico.
- Dicotómicas: con múltiples categorías como pago_atiempo, que representa una variable binaria clave para el análisis, posiblemente utilizada como variable objetivo.
- Politómicas: como tipo_credito, que aunque está codificada numéricamente, representa categorías discretas (multiples).

Tambien se identificó variables de conteo relacionadas con la cantidad de créditos por sector económico, así como una variable temporal (fecha_prestamo), que puede ser utilizada para análisis de tendencias en el tiempo.

# Limpieza de los Datos

La variable puntaje presentaba inconsistencias en el formato decimal, utilizando coma como separador. Se realizó una transformación para estandarizar el formato a punto decimal y posteriormente se convirtió a tipo numérico, garantizando su correcta interpretación para el análisis.

Se generó una tabla resumen que consolida la información de cada variable, incluyendo su tipo de dato, tipo general y subtipo.

Esta clasificación permite identificar correctamente variables numéricas, categóricas, booleanas y temporales, así como distinguir entre variables nominales, ordinales y dicotómicas, facilitando su tratamiento en el análisis exploratorio y modelado.

# Tratamiento de los nulos

-- Nulos menores al 5%

Se imputaron valores nulos en variables numéricas con baja proporción (<6%) utilizando la mediana, debido a su robustez frente a valores extremos. Se redondeó el valor para mantener consistencia con el tipo entero de las variables.

-- Nulos del 27%

Se identificaron variables con alto porcentaje de valores nulos (~27%).  
Para estas variables se aplicó una estrategia combinada:

- Se crearon variables indicadoras de ausencia de información.
- Para variables categóricas se imputó la categoría "Desconocido".
- Para variables numéricas se utilizó la mediana, preservando la distribución original.
- Se conservaron los indicadores como variables explicativas adicionales.

Esta estrategia permite capturar el posible efecto informativo de los valores faltantes.

Se crearon variables indicadoras de valores nulos para evaluar su posible aporte analítico. 
Sin embargo, tras la revisión del objetivo del análisis, se decidió no incluirlas en el dataset final 
con el fin de mantener un modelo más simple e interpretable.


# obserbación del balanceo
El 95.25% de tus clientes pagan a tiempo, mientras que solo un 4.75% (511 personas) no lo hacen.
