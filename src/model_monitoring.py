# librerias

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from scipy.stats import ks_2samp, chi2_contingency

# Importamos TUS funciones de los archivos previos
from cargar_datos import cargar_datos
from ft_engineering import limpiar_datos, crear_features, preparar_datos_modelo

# Configuración de página
st.set_page_config(page_title="Monitoreo de Data Drift", layout="wide")

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================
DATASET_PATH = "./Base_de_datos.csv" 
MONITOR_LOG = "./Base_de_datos.csv" # En producción sería la data de hoy

@st.cache_data
def load_all_data():
    # Carga y procesamiento 
    df_raw = pd.read_csv(DATASET_PATH)
    df_full = crear_features(limpiar_datos(df_raw))
    X, y, num_cols, cat_cols = preparar_datos_modelo(df_full)
    
    # Dataset Referencia (Entrenamiento)
    X_ref = X.copy()
    
    # Dataset Monitoreo (Simulación de datos actuales con ruido)
    X_mon = X.sample(frac=0.5, random_state=1).copy()
    X_mon[num_cols] = X_mon[num_cols] * np.random.uniform(0.85, 1.15, size=X_mon[num_cols].shape)
    
    # Tabla con pronósticos
    X_mon['Pronóstico'] = (np.random.rand(len(X_mon)) > 0.15).astype(int)
    
    return X_ref, X_mon, num_cols, cat_cols

X_ref, X_mon, num_cols, cat_cols = load_all_data()

# Inicializamos p_val 
p_val = 1.0 

# ==========================================
# 2. FUNCIONES DE MÉTRICAS (KS, PSI, JS)
# ==========================================
def calcular_psi(expected, actual, buckets=10):
    def scale_data(data, bins):
        return np.histogram(data, bins=bins)[0] / len(data)
    min_val, max_val = min(expected.min(), actual.min()), max(expected.max(), actual.max())
    breakpoints = np.linspace(min_val, max_val, buckets + 1)
    exp_percents = np.clip(scale_data(expected, breakpoints), 1e-6, None)
    act_percents = np.clip(scale_data(actual, breakpoints), 1e-6, None)
    return np.sum((exp_percents - act_percents) * np.log(exp_percents / act_percents))

def jensen_shannon_div(p, q):
    from scipy.spatial.distance import jensenshannon
    p_hist = np.histogram(p, bins=20, density=True)[0]
    q_hist = np.histogram(q, bins=20, density=True)[0]
    return jensenshannon(p_hist, q_hist)

# ==========================================
# 3. INTERFAZ DE STREAMLIT (TABS Y ALERTAS)
# ==========================================
st.title("🚀 Tercer Avance: Monitoreo Avanzado de Data Drift")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Métricas y Visualización", "📈 Análisis Temporal", "💡 Recomendaciones"])

with tab1:
    st.subheader("Detección de cambios por variable")
    col_feat = st.selectbox("Selecciona una variable para monitorear:", list(X_ref.columns))
    
    c1, c2, c3, c4 = st.columns(4)
    
    if col_feat in num_cols:
        ks_stat, p_val = ks_2samp(X_ref[col_feat], X_mon[col_feat])
        psi_val = calcular_psi(X_ref[col_feat], X_mon[col_feat])
        js_div = jensen_shannon_div(X_ref[col_feat], X_mon[col_feat])
        
        c1.metric("KS Test (p-value)", f"{p_val:.4f}")
        c2.metric("PSI", f"{psi_val:.4f}")
        c3.metric("JS Divergence", f"{js_div:.4f}")
        
        # Lógica de Semáforo (Indicadores visuales)
        if p_val < 0.05:
            c4.error("🔴 Drift Detectado")
        elif p_val < 0.1:
            c4.warning("🟡 Riesgo Moderado")
        else:
            c4.success("🟢 Estable")
            
    elif col_feat in cat_cols:
        contingency = pd.crosstab(X_ref[col_feat], X_mon[col_feat])
        _, p_val, _, _ = chi2_contingency(contingency)
        c1.metric("Chi-Square (p-value)", f"{p_val:.4f}")
        c4.error("🔴 Drift Detectado" if p_val < 0.05 else "🟢 Estable")

    # Gráfico de comparación (Histograma + Violín)
    fig = px.histogram(X_ref, x=col_feat, nbins=30, title=f"Distribución Histórica vs Actual: {col_feat}", 
                       marginal="violin", barmode='overlay', color_discrete_sequence=['blue'])
        # DataFrame combinado
    df_comparativo = pd.concat([
        X_ref[[col_feat]].assign(Dataset='Referencia'),
        X_mon[[col_feat]].assign(Dataset='Monitoreo')
    ])
    
    fig = px.histogram(
        df_comparativo, 
        x=col_feat, 
        color='Dataset', 
        barmode='overlay',
        marginal="violin", 
        title=f"Distribución Histórica vs Actual: {col_feat}",
        color_discrete_map={'Referencia': 'blue', 'Monitoreo': 'red'}
    )
    
    st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("Evolución del Drift en el tiempo")
    # Simulación de tendencia temporal
    fechas = pd.date_range(end=pd.Timestamp.now(), periods=15, freq='D')
    trend_data = pd.DataFrame({
        'Fecha': fechas,
        'PSI_Global': np.random.uniform(0.05, 0.25, size=15)
    })
    fig_trend = px.line(trend_data, x='Fecha', y='PSI_Global', title="Comportamiento del PSI Global (15 días)")
    fig_trend.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Umbral de Alerta")
    st.plotly_chart(fig_trend, width='stretch')

with tab3:
    st.subheader("Mensajes Automáticos y Acciones")
    
    if p_val < 0.05:
        st.error("🚨 **ALERTA CRÍTICA:** Se han detectado desviaciones que pueden comprometer la precisión.")
        st.markdown("""
        **Sugerencias de acción:**
        *   **Retraining:** Programar re-entrenamiento inmediato con datos de la última ventana temporal.
        *   **Revisión de Variables:** Inspeccionar la variable seleccionada; el cambio poblacional es significativo.
        *   **Ajuste de Umbrales:** Validar si el modelo requiere un ajuste en los umbrales de decisión (thresholds).
        """)
    else:
        st.success("✅ **SITUACIÓN NORMAL:** No se detectan anomalías críticas en el desempeño poblacional.")

    st.markdown("---")
    st.subheader("📋 Tabla de Monitoreo (Datos + Pronósticos)")
    columnas_ordenadas = ['Pronóstico'] + [col for col in X_mon.columns if col != 'Pronóstico']
    st.dataframe(X_mon[columnas_ordenadas].head(20), width='stretch')
    st.dataframe(X_mon.head(20), width='stretch')
