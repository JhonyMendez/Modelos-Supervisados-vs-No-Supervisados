import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, silhouette_score, davies_bouldin_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="ML App - COVID-19 Pakistan", layout="wide")

# Título principal
st.title("🦠 Aplicación de Machine Learning - COVID-19 Pakistan")
st.markdown("**Modelos:** Gaussian Naive Bayes (Supervisado) + K-Means (No Supervisado)")

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Pakistan_COVID19.csv')
        # Limpiar datos: eliminar filas con valores nulos
        df = df.dropna()
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'Pakistan_COVID19.csv'. Por favor, colócalo en la misma carpeta que app.py")
        return None

df = load_data()

if df is None:
    st.stop()

# Preparar datos
@st.cache_data
def prepare_data(df):
    # Codificar la columna Province (variable categórica)
    le = LabelEncoder()
    df_processed = df.copy()
    df_processed['Province_Encoded'] = le.fit_transform(df_processed['Province'])
    
    # Características numéricas
    numeric_features = ['New_Cases', 'Recoveries', 'Deaths', 'Vaccinations', 'Hospitalized', 'Tests_Conducted']
    
    # Variable objetivo (target) será Province_Encoded
    X = df_processed[numeric_features].values
    y = df_processed['Province_Encoded'].values
    
    return X, y, le, numeric_features, df_processed

X, y, label_encoder, feature_names, df_processed = prepare_data(df)
province_names = label_encoder.classes_

# Sidebar para navegación
st.sidebar.title("🎯 Navegación")
mode = st.sidebar.radio("Selecciona el Modo:", 
                        ["📊 Exploración de Datos", 
                         "🎓 Modelo Supervisado (Gaussian NB)", 
                         "🔍 Modelo No Supervisado (K-Means)",
                         "💾 Zona de Exportación"])

# =============================================
# MODO 1: EXPLORACIÓN DE DATOS
# =============================================
if mode == "📊 Exploración de Datos":
    st.header("📊 Exploración del Dataset COVID-19 Pakistan")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", len(df))
    with col2:
        st.metric("Provincias", len(province_names))
    with col3:
        st.metric("Variables Numéricas", len(feature_names))
    with col4:
        st.metric("Total Casos", f"{df['New_Cases'].sum():,.0f}")
    
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(15))
    
    st.subheader("Información de las Columnas")
    col_info = pd.DataFrame({
        'Columna': df.columns,
        'Tipo': df.dtypes.values,
        'Valores No Nulos': df.count().values,
        'Valores Únicos': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info)
    
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df[feature_names].describe())
    
    # Visualizaciones
    st.subheader("📈 Visualizaciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de casos por provincia
        fig, ax = plt.subplots(figsize=(10, 6))
        province_cases = df.groupby('Province')['New_Cases'].sum().sort_values(ascending=False)
        province_cases.plot(kind='bar', ax=ax, color='#FF6B6B')
        ax.set_ylabel('Total de Casos Nuevos')
        ax.set_xlabel('Provincia')
        ax.set_title('Casos de COVID-19 por Provincia')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Distribución de registros por provincia
        fig, ax = plt.subplots(figsize=(10, 6))
        df['Province'].value_counts().plot(kind='bar', ax=ax, color='#4ECDC4')
        ax.set_ylabel('Número de Registros')
        ax.set_xlabel('Provincia')
        ax.set_title('Distribución de Registros por Provincia')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Matriz de correlación
    st.subheader("🔗 Matriz de Correlación")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df[feature_names].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=ax, square=True)
    ax.set_title('Correlación entre Variables')
    st.pyplot(fig)

# =============================================
# MODO 2: MODELO SUPERVISADO (GAUSSIAN NAIVE BAYES)
# =============================================
elif mode == "🎓 Modelo Supervisado (Gaussian NB)":
    st.header("🎓 Modelo Supervisado: Gaussian Naive Bayes")
    
    st.markdown("""
    **Objetivo:** Predecir la **Provincia** basándose en las métricas de COVID-19.
    
    **Gaussian Naive Bayes** es un clasificador probabilístico que asume que las características 
    siguen una distribución normal (gaussiana). Es rápido y efectivo para clasificación multiclase.
    """)
    
    # Parámetros
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 40, 20) / 100
    with col2:
        random_state = st.number_input("Semilla aleatoria", value=42, min_value=0)
    
    # Normalización opcional
    use_scaling = st.checkbox("Normalizar datos (StandardScaler)", value=True, 
                             help="Recomendado cuando las variables tienen diferentes escalas")
    
    if st.button("🚀 Entrenar Modelo Gaussian Naive Bayes", type="primary"):
        # Preparar datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalizar si es necesario
        if use_scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state['scaler'] = scaler
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            st.session_state['scaler'] = None
        
        # Entrenar modelo
        model_supervised = GaussianNB()
        model_supervised.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = model_supervised.predict(X_test_scaled)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Guardar en session_state
        st.session_state['model_supervised'] = model_supervised
        st.session_state['X_train'] = X_train_scaled
        st.session_state['X_test'] = X_test_scaled
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['metrics_supervised'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        # Mostrar métricas
        st.success("✅ Modelo entrenado exitosamente!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Accuracy", f"{accuracy:.4f}")
        col2.metric("🎯 Precision", f"{precision:.4f}")
        col3.metric("🎯 Recall", f"{recall:.4f}")
        col4.metric("🎯 F1-Score", f"{f1:.4f}")
        
        # Matriz de confusión
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=province_names, yticklabels=province_names, ax=ax)
        ax.set_ylabel('Provincia Real')
        ax.set_xlabel('Provincia Predicha')
        ax.set_title('Matriz de Confusión - Predicción de Provincias')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Reporte de clasificación
        from sklearn.metrics import classification_report
        st.subheader("📊 Reporte Detallado por Provincia")
        report = classification_report(y_test, y_pred, target_names=province_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4))
    
    # Sección de predicción interactiva
    st.subheader("🔮 Predicción Interactiva")
    
    if 'model_supervised' in st.session_state:
        st.markdown("**Ingresa los datos de COVID-19 para predecir la provincia:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_cases = st.number_input("Casos Nuevos", min_value=0, value=100, step=10)
            recoveries = st.number_input("Recuperados", min_value=0, value=50, step=10)
        
        with col2:
            deaths = st.number_input("Fallecidos", min_value=0, value=5, step=1)
            vaccinations = st.number_input("Vacunaciones", min_value=0, value=1000, step=100)
        
        with col3:
            hospitalized = st.number_input("Hospitalizados", min_value=0, value=20, step=5)
            tests_conducted = st.number_input("Tests Realizados", min_value=0, value=500, step=50)
        
        input_data = np.array([[new_cases, recoveries, deaths, vaccinations, hospitalized, tests_conducted]])
        
        # Normalizar si es necesario
        if st.session_state.get('scaler') is not None:
            input_data_scaled = st.session_state['scaler'].transform(input_data)
        else:
            input_data_scaled = input_data
        
        prediction = st.session_state['model_supervised'].predict(input_data_scaled)[0]
        prediction_proba = st.session_state['model_supervised'].predict_proba(input_data_scaled)[0]
        
        st.session_state['current_prediction'] = {
            'input': {
                'New_Cases': int(new_cases),
                'Recoveries': int(recoveries),
                'Deaths': int(deaths),
                'Vaccinations': int(vaccinations),
                'Hospitalized': int(hospitalized),
                'Tests_Conducted': int(tests_conducted)
            },
            'output_class': int(prediction),
            'output_label': province_names[prediction]
        }
        
        st.success(f"🗺️ **Predicción: {province_names[prediction]}**")
        
        # Mostrar probabilidades
        st.subheader("Probabilidades por Provincia")
        prob_df = pd.DataFrame({
            'Provincia': province_names,
            'Probabilidad': prediction_proba
        }).sort_values('Probabilidad', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(prob_df['Provincia'], prob_df['Probabilidad'], color='#45B7D1')
        ax.set_xlabel('Probabilidad')
        ax.set_title('Probabilidad de Pertenencia a cada Provincia')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(prob_df.style.background_gradient(subset=['Probabilidad'], cmap='Blues'))
    else:
        st.info("👆 Entrena el modelo primero para usar esta función")

# =============================================
# MODO 3: MODELO NO SUPERVISADO (K-MEANS)
# =============================================
elif mode == "🔍 Modelo No Supervisado (K-Means)":
    st.header("🔍 Modelo No Supervisado: K-Means Clustering")
    
    st.markdown("""
    **Objetivo:** Agrupar registros similares de COVID-19 en clusters sin usar las etiquetas de provincia.
    
    **K-Means** identifica patrones naturales en los datos agrupando registros con características similares.
    """)
    
    # Parámetros
    col1, col2, col3 = st.columns(3)
    with col1:
        n_clusters = st.slider("Número de Clusters (K)", 2, 10, len(province_names))
    with col2:
        max_iter = st.number_input("Máximo de iteraciones", value=300, min_value=100)
    with col3:
        random_state = st.number_input("Semilla aleatoria", value=42, min_value=0, key='kmeans_seed')
    
    # Normalización (recomendada para K-Means)
    use_scaling = st.checkbox("Normalizar datos", value=True, 
                             help="Muy recomendado para K-Means debido a diferentes escalas")
    
    if st.button("🚀 Entrenar Modelo K-Means", type="primary"):
        # Normalizar datos si es necesario
        if use_scaling:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state['scaler_kmeans'] = scaler
        else:
            X_scaled = X
            st.session_state['scaler_kmeans'] = None
        
        # Entrenar modelo
        model_unsupervised = KMeans(n_clusters=n_clusters, max_iter=max_iter, 
                                   random_state=random_state, n_init=10)
        cluster_labels = model_unsupervised.fit_predict(X_scaled)
        
        # Calcular métricas
        silhouette = silhouette_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
        
        # Guardar en session_state
        st.session_state['model_unsupervised'] = model_unsupervised
        st.session_state['cluster_labels'] = cluster_labels
        st.session_state['X_scaled'] = X_scaled
        st.session_state['metrics_unsupervised'] = {
            'silhouette_score': float(silhouette),
            'davies_bouldin': float(davies_bouldin),
            'n_clusters': int(n_clusters)
        }
        
        # Mostrar métricas
        st.success("✅ Modelo K-Means entrenado exitosamente!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Silhouette Score", f"{silhouette:.4f}", 
                   help="Rango: -1 a 1. Valores cercanos a 1 indican clusters bien definidos")
        col2.metric("📊 Davies-Bouldin Index", f"{davies_bouldin:.4f}",
                   help="Valores más bajos indican mejor separación de clusters")
        col3.metric("🎯 Clusters", n_clusters)
        
        # Interpretación automática
        if silhouette > 0.5:
            st.success("✅ Excelente separación de clusters")
        elif silhouette > 0.3:
            st.info("ℹ️ Separación de clusters aceptable")
        else:
            st.warning("⚠️ Los clusters tienen baja separación")
        
        # Visualización de clusters
        st.subheader("📊 Visualización de Clusters")
        
        # Análisis de componentes principales para visualización 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico 1: Clusters encontrados
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                  cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        axes[0].set_title('Clusters Encontrados por K-Means (PCA)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # Gráfico 2: Provincias reales (para comparación)
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                                  cmap='Set1', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        axes[1].set_title('Provincias Reales (Referencia)')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Provincia')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Distribución de clusters
        st.subheader("📊 Distribución de Registros por Cluster")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            cluster_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Número de Registros')
            ax.set_title('Tamaño de cada Cluster')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Tabla de distribución
            cluster_df = pd.DataFrame({
                'Cluster': range(n_clusters),
                'Cantidad': [np.sum(cluster_labels == i) for i in range(n_clusters)],
                'Porcentaje': [f"{100*np.sum(cluster_labels == i)/len(cluster_labels):.2f}%" 
                             for i in range(n_clusters)]
            })
            st.dataframe(cluster_df, hide_index=True, use_container_width=True)
        
        # Análisis de características por cluster
        st.subheader("📈 Características Promedio por Cluster")
        
        df_with_clusters = df_processed.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        cluster_means = df_with_clusters.groupby('Cluster')[feature_names].mean()
        
        st.dataframe(cluster_means.round(2))
        
        # Heatmap de características por cluster
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(cluster_means.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Características')
        ax.set_title('Perfil de cada Cluster (valores promedio)')
        plt.tight_layout()
        st.pyplot(fig)

# =============================================
# MODO 4: ZONA DE EXPORTACIÓN
# =============================================
elif mode == "💾 Zona de Exportación":
    st.header("💾 Zona de Exportación (Dev Tools)")
    
    st.markdown("""
    Esta sección permite exportar los modelos entrenados y sus resultados en formatos
    que pueden ser consumidos por aplicaciones frontend (React) o reutilizados en Python.
    """)
    
    # Exportar Modelo Supervisado
    st.subheader("📤 Exportar Modelo Supervisado (Gaussian Naive Bayes)")
    
    if 'model_supervised' in st.session_state:
        # Crear JSON
        json_supervised = {
            "model_type": "Supervised",
            "model_name": "Gaussian Naive Bayes",
            "algorithm": "GaussianNB",
            "dataset": "Pakistan COVID-19 Dataset",
            "features": feature_names,
            "target_classes": province_names.tolist(),
            "metrics": st.session_state['metrics_supervised'],
            "training_info": {
                "test_size": "20%",
                "normalization": "StandardScaler" if st.session_state.get('scaler') is not None else "None"
            }
        }
        
        if 'current_prediction' in st.session_state:
            json_supervised['sample_prediction'] = st.session_state['current_prediction']
        
        # Botón de descarga JSON
        json_str = json.dumps(json_supervised, indent=2)
        st.download_button(
            label="📥 Descargar JSON (Supervisado)",
            data=json_str,
            file_name="gaussian_nb_covid_results.json",
            mime="application/json"
        )
        
        with st.expander("👁️ Ver JSON"):
            st.code(json_str, language='json')
        
        # Botón de descarga PKL
        pkl_supervised = pickle.dumps(st.session_state['model_supervised'])
        st.download_button(
            label="📥 Descargar Modelo .pkl (Supervisado)",
            data=pkl_supervised,
            file_name="gaussian_nb_covid_model.pkl",
            mime="application/octet-stream"
        )
        
        # Exportar scaler si existe
        if st.session_state.get('scaler') is not None:
            pkl_scaler = pickle.dumps(st.session_state['scaler'])
            st.download_button(
                label="📥 Descargar Scaler .pkl",
                data=pkl_scaler,
                file_name="scaler.pkl",
                mime="application/octet-stream"
            )
        
        st.success("✅ Modelo supervisado listo para exportar")
    else:
        st.warning("⚠️ Entrena el modelo supervisado primero en la sección correspondiente")
    
    st.divider()
    
    # Exportar Modelo No Supervisado
    st.subheader("📤 Exportar Modelo No Supervisado (K-Means)")
    
    if 'model_unsupervised' in st.session_state:
        # Crear JSON
        json_unsupervised = {
            "model_type": "Unsupervised",
            "algorithm": "K-Means",
            "dataset": "Pakistan COVID-19 Dataset",
            "features": feature_names,
            "parameters": {
                "n_clusters": st.session_state['metrics_unsupervised']['n_clusters'],
                "max_iter": 300,
                "n_init": 10
            },
            "metrics": {
                "silhouette_score": st.session_state['metrics_unsupervised']['silhouette_score'],
                "davies_bouldin": st.session_state['metrics_unsupervised']['davies_bouldin']
            },
            "cluster_labels": st.session_state['cluster_labels'].tolist(),
            "cluster_distribution": {
                f"Cluster_{i}": int(np.sum(st.session_state['cluster_labels'] == i)) 
                for i in range(st.session_state['metrics_unsupervised']['n_clusters'])
            }
        }
        
        # Botón de descarga JSON
        json_str = json.dumps(json_unsupervised, indent=2)
        st.download_button(
            label="📥 Descargar JSON (No Supervisado)",
            data=json_str,
            file_name="kmeans_covid_results.json",
            mime="application/json"
        )
        
        with st.expander("👁️ Ver JSON"):
            st.code(json_str, language='json')
        
        # Botón de descarga PKL
        pkl_unsupervised = pickle.dumps(st.session_state['model_unsupervised'])
        st.download_button(
            label="📥 Descargar Modelo .pkl (No Supervisado)",
            data=pkl_unsupervised,
            file_name="kmeans_covid_model.pkl",
            mime="application/octet-stream"
        )
        
        st.success("✅ Modelo no supervisado listo para exportar")
    else:
        st.warning("⚠️ Entrena el modelo no supervisado primero en la sección correspondiente")
    
    st.divider()
    
    # Instrucciones de uso
    st.subheader("📖 Instrucciones de Uso")
    
    st.markdown("""
    ### Cómo usar los archivos exportados:
    
    **Archivos JSON:**
    - Pueden ser consumidos directamente por aplicaciones React/JavaScript
    - Contienen todas las métricas y resultados del modelo
    - Formato legible y fácil de parsear
    
    **Archivos .pkl (Pickle):**
    - Contienen el modelo entrenado completo
    - Pueden ser cargados en Python para hacer predicciones
    
    ```python
    # Ejemplo de carga del modelo en Python
    import pickle
    
    # Cargar modelo
    with open('gaussian_nb_covid_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Hacer predicción
    nueva_prediccion = model.predict([[100, 50, 5, 1000, 20, 500]])
    ```
    
    **Integración con React:**
    ```javascript
    // Leer el JSON en React
    fetch('gaussian_nb_covid_results.json')
        .then(response => response.json())
        .then(data => {
            console.log('Accuracy:', data.metrics.accuracy);
            console.log('Provincias:', data.target_classes);
        });
    ```
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(f"""
**Proyecto de Machine Learning**  
Dataset: COVID-19 Pakistan  
Modelos: Gaussian NB + K-Means  

**Características:**
- {len(df)} registros
- {len(province_names)} provincias
- {len(feature_names)} variables

Streamlit App v2.0
""")