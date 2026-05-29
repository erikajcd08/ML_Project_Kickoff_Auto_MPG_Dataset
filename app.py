import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Auto MPG - ML Project", page_icon="🚗", layout="wide")

@st.cache_data
def load_and_prepare():
    auto_mpg = fetch_ucirepo(id=9)
    df = pd.concat([auto_mpg.data.features, auto_mpg.data.targets], axis=1)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
    df = pd.get_dummies(df, columns=['origin'], drop_first=False)
    df['power_to_weight'] = df['horsepower'] / df['weight']
    df['displacement_per_cylinder'] = df['displacement'] / df['cylinders']
    return df

@st.cache_data
def train_models():
    df = load_and_prepare()
    X = df.drop(columns=['mpg'])
    y = df['mpg']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    cols_to_scale = ['displacement', 'cylinders', 'horsepower', 'weight',
                     'acceleration', 'model_year', 'power_to_weight',
                     'displacement_per_cylinder']
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    models = {
        'KNN Regressor': KNeighborsRegressor(n_neighbors=5),
        'Regresión Lineal': LinearRegression(),
        'Árbol de Decisión': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Random Forest Optimizado': RandomForestRegressor(
            max_depth=10, min_samples_split=2, n_estimators=200, random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'R²': round(r2_score(y_test, y_pred), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            'model': model
        }
    return results, X_train, X_test, y_train, y_test, scaler

def mpg_to_l100km(mpg):
    return round(235.214 / mpg, 2)

# ── Navegación ───────────────────────────────────────────────
st.sidebar.title("🚗 Auto MPG Project")
st.sidebar.markdown("---")
slide = st.sidebar.radio("Navegación", [
    "1. Título",
    "2. Visión general",
    "3. Datos y preparación",
    "4. Feature Engineering & Selection",
    "5. Model Building & Evaluation",
    "6. Hyperparameter Tuning & Cross Validation",
    "7. Hallazgos & Insights",
    "8. Aplicación real",
    "9. Dificultades y aprendizajes",
    "10. Trabajo futuro",
    "11. Cierre"
])

df = load_and_prepare()

# ════════════════════════════════════════════════════════════
# SLIDE 1 — TÍTULO
# ════════════════════════════════════════════════════════════
if slide == "1. Título":
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center'>
            <h1>🚗 AUTO MPG</h1>
            <h3>Predicción del Consumo de Combustible en Vehículos (1970–1982)</h3>
            <br>
            <p style='color:gray; font-style:italic; font-size:0.9em'>
                MPG = Miles Per Gallon = cuántas millas recorre un coche con un galón de gasolina
            </p>
            <br>
            <h4>Erika Campo Díaz</h4>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SLIDE 2 — VISIÓN GENERAL
# ════════════════════════════════════════════════════════════
elif slide == "2. Visión general":
    st.title("📌 Visión general del proyecto")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pregunta de investigación")
        st.info("❓ ¿Es posible predecir cuánto consume un coche únicamente a partir de sus características técnicas?")
        st.markdown("### Hipótesis")
        st.markdown("""
        - 💡 A mayor peso del coche, mayor consumo
        - 💡 A mayor cilindrada, mayor consumo
        - 💡 Los coches más modernos son más eficientes
        - 💡 Los coches europeos y japoneses son más eficientes que los americanos
        """)
    with col2:
        st.markdown("### Impacto potencial")
        st.markdown("""
        Los fabricantes podrían estimar la eficiencia de un diseño
        antes de construirlo, ahorrando costes de prototipado.
        También útil para regulaciones medioambientales y
        comparativa de vehículos por parte del consumidor.
        """)
        st.markdown("### Tipo de problema")
        with st.container(border=True):
            st.markdown("""
            🎯 **Regresión supervisada**

            Tenemos datos históricos de coches con sus características
            y su consumo real. El modelo aprende de esos datos para
            predecir el consumo de coches nuevos que nunca ha visto.
            """)

# ════════════════════════════════════════════════════════════
# SLIDE 3 — DATOS Y PREPARACIÓN
# ════════════════════════════════════════════════════════════
elif slide == "3. Datos y preparación":
    st.title("📊 Selección y preparación de los datos")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dataset")
        with st.container(border=True):
            st.markdown("""
            **Auto MPG — UCI Machine Learning Repository**
            398 filas · 8 columnas · Coches fabricados entre 1970 y 1982
            """)
        st.markdown("### 💡 ¿Qué es el MPG?")
        st.info("""
        MPG significa *Miles Per Gallon*, es decir, cuántas millas
        recorre un coche con un galón de gasolina.
        **Cuanto más alto el MPG, más eficiente es el coche y menos
        contamina.** En España usamos litros por 100 km, pero el
        concepto es el mismo al revés: más MPG = menos consumo.
        """)
        st.markdown("### Limpieza realizada")
        st.warning("⚠️ **Valores nulos:** 6 registros sin dato de potencia → rellenados con el valor más representativo (mediana)")
        st.success("✅ **Duplicados:** Ninguno encontrado — el dataset estaba bastante limpio")
        st.info("ℹ️ **Origen del coche:** Convertido en 3 columnas independientes (USA, Europa, Japón) con valores 0 o 1")
        st.markdown("### ¿Qué información tenemos de cada coche?")
        traducciones = pd.DataFrame({
            'Variable': ['mpg', 'cylinders', 'displacement', 'horsepower',
                         'weight', 'acceleration', 'model_year', 'origin'],
            'Significado': [
                'Consumo (millas por galón) 🎯',
                'Número de cilindros del motor',
                'Cilindrada (tamaño del motor)',
                'Potencia en caballos',
                'Peso del coche (en libras)',
                'Aceleración (seg. hasta 60 mph)',
                'Año del modelo (70 = 1970)',
                'País de origen del fabricante'
            ]
        })
        st.dataframe(traducciones, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### Vista del dataset (primeras filas)")
        cols_show = ['mpg', 'cylinders', 'horsepower', 'weight', 'model_year']
        st.dataframe(df[cols_show].head(10), use_container_width=True)
        st.markdown("### Estadísticas básicas")
        st.dataframe(
            df[cols_show].describe().round(2),
            use_container_width=True
        )

# ════════════════════════════════════════════════════════════
# SLIDE 4 — FEATURE ENGINEERING & SELECTION
# ════════════════════════════════════════════════════════════
elif slide == "4. Feature Engineering & Selection":
    st.title("⚙️ Feature Engineering & Selection")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Nuevas variables creadas")
        with st.container(border=True):
            st.markdown("**Ratio potencia / peso**")
            st.markdown("""
            Imagina dos coches con los mismos caballos: uno pesa 1.000 kg
            y otro 2.000 kg. El ligero será más eficiente. Esta variable
            captura exactamente eso.
            """)
        with st.container(border=True):
            st.markdown("**Cilindrada por cilindro**")
            st.markdown("""
            Divide el tamaño total del motor entre el número de cilindros.
            Nos dice cuán "grande" es cada cilindro individualmente,
            lo que influye directamente en el consumo.
            """)
        st.markdown("### Escalado de datos")
        st.info("""
        Las variables tienen escalas muy distintas (ej: peso en miles de
        libras vs. año en decenas). Se normalizaron para que el modelo
        las trate en igualdad de condiciones.
        """)
        st.markdown("### División de datos")
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.success("**Entrenamiento 80% — 318 coches**  \nEl modelo aprende aquí")
        with col_b:
            st.error("**Test 20%**  \n80 coches")
    with col2:
        st.markdown("### Mapa de correlaciones")
        numeric_cols = ['mpg', 'displacement', 'cylinders', 'horsepower',
                        'weight', 'acceleration', 'model_year']
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm',
                    fmt='.2f', ax=ax)
        ax.set_title('Correlación entre variables')
        st.pyplot(fig)
        plt.close()
        st.caption("🔴 Rojo = a más valor, más consume · 🟢 Verde = a más valor, menos consume")

# ════════════════════════════════════════════════════════════
# SLIDE 5 — MODEL BUILDING & EVALUATION
# ════════════════════════════════════════════════════════════
elif slide == "5. Model Building & Evaluation":
    st.title("🤖 Model Building & Evaluation")
    st.markdown("---")
    with st.spinner("Entrenando modelos..."):
        results, X_train, X_test, y_train, y_test, scaler = train_models()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Modelos entrenados")
        with st.container(border=True):
            st.markdown("""
            **KNN** — pregunta: "¿a qué coches ya vistos se parece más este?"
            y predice basándose en sus vecinos más cercanos.
            """)
        with st.container(border=True):
            st.markdown("""
            **Regresión Lineal** — busca la fórmula matemática más simple
            que relaciona las características con el consumo.
            """)
        with st.container(border=True):
            st.markdown("""
            **Árbol de Decisión** — toma decisiones tipo "si pesa más de X
            y tiene más de Y cilindros, entonces consume Z".
            """)
        with st.container(border=True):
            st.markdown("""
            **Random Forest** — combina cientos de árboles de decisión
            y promedia sus respuestas para ser más preciso.
            """)
        st.markdown("### Métricas de evaluación")
        with st.container(border=True):
            st.markdown("""
            **R²** — de 0 a 1, cuánto del consumo real explica el modelo.
            Un R² de 0.92 significa que explica el 92% de la variación.

            **RMSE** — error medio en MPG. Un RMSE de 2 significa que
            el modelo se equivoca de media en 2 MPG.
            """)
    with col2:
        st.markdown("### Tabla comparativa")
        results_df = pd.DataFrame([
            {'Modelo': name, 'R²': v['R²'], 'Error medio (MPG)': v['RMSE']}
            for name, v in results.items()
        ]).sort_values('R²', ascending=False)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.markdown("### Comparativa visual R²")
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ['#1D9E75' if r == results_df['R²'].max() else '#378ADD'
                  for r in results_df['R²']]
        ax.barh(results_df['Modelo'], results_df['R²'], color=colors)
        ax.set_xlabel('R²')
        ax.set_title('R² por modelo (más alto = mejor)')
        ax.set_xlim(0.7, 1.0)
        for i, v in enumerate(results_df['R²']):
            ax.text(v + 0.002, i, str(v), va='center', fontsize=10)
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════
# SLIDE 6 — HYPERPARAMETER TUNING & CROSS VALIDATION
# ════════════════════════════════════════════════════════════
elif slide == "6. Hyperparameter Tuning & Cross Validation":
    st.title("🔧 Hyperparameter Tuning & Cross Validation")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ¿Qué es el Hyperparameter Tuning?")
        st.markdown("""
        Cada modelo tiene "botones de configuración" que se pueden ajustar
        para mejorar su rendimiento. El proceso de encontrar la mejor
        combinación de esos botones se llama **Hyperparameter Tuning**.
        Usamos **GridSearchCV**, que prueba automáticamente todas las
        combinaciones posibles.
        """)
        st.markdown("### Configuraciones probadas en Random Forest")
        params_df = pd.DataFrame({
            'Parámetro': ['Número de árboles', 'Profundidad máxima', 'Mín. datos por división'],
            'Valores probados': ['100 ó 200', '5, 10 ó sin límite', '2 ó 5']
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)
        st.success("✅ **Mejor configuración:** 200 árboles, profundidad máxima de 10")
        st.warning("⚠️ La mejora fue mínima — el modelo ya estaba bien ajustado desde el principio")
        st.markdown("### Resultado final")
        final_df = pd.DataFrame({
            'Modelo': ['Random Forest base', 'Random Forest Optimizado'],
            'R²': [0.9223, 0.9222],
            'Error medio (MPG)': [2.0443, 2.0449]
        })
        st.dataframe(final_df, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("### ¿Qué es la Cross Validation?")
        st.info("""
        Es como hacer un examen 5 veces, cada vez con preguntas distintas.
        Así nos aseguramos de que el modelo no ha "memorizado" los datos,
        sino que realmente ha aprendido.

        Dividimos los datos en 5 partes y el modelo se evalúa en cada
        una de ellas. El resultado final es el promedio de las 5 rondas
        — mucho más fiable que una sola evaluación.
        """)
        st.markdown("### Esquema visual")
        for i in range(1, 6):
            cols = st.columns(6)
            cols[0].caption(f"Ronda {i}")
            for j in range(1, 6):
                if j == i:
                    cols[j].error("Test")
                else:
                    cols[j].success("Train")

# ════════════════════════════════════════════════════════════
# SLIDE 7 — HALLAZGOS & INSIGHTS
# ════════════════════════════════════════════════════════════
elif slide == "7. Hallazgos & Insights":
    st.title("🔍 Hallazgos & Insights")
    st.markdown("---")
    with st.spinner("Cargando resultados..."):
        results, X_train, X_test, y_train, y_test, scaler = train_models()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ¿Qué factores determinan más el consumo?")
        rf = results['Random Forest Optimizado']['model']
        importances = pd.Series(
            rf.feature_importances_, index=X_train.columns
        ).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        importances.plot(kind='barh', color='steelblue', ax=ax)
        ax.set_title('Importancia de cada variable (Random Forest)')
        ax.set_xlabel('Importancia relativa')
        st.pyplot(fig)
        plt.close()
        st.markdown("### Conclusiones clave")
        st.markdown("""
        - El **tamaño del motor** (cilindrada) es el factor más determinante: motores más grandes consumen más (≈39%)
        - Un coche más **pesado** siempre consume más, independientemente de su origen o año
        - Curiosamente, el **país de origen** casi no influye cuando ya conocemos las características técnicas
        """)
    with col2:
        st.markdown("### ¿Por qué los coches fueron volviéndose más eficientes?")
        fig, ax = plt.subplots(figsize=(7, 4))
        df_plot = load_and_prepare()
        sns.lineplot(x='model_year', y='mpg', data=df_plot,
                     estimator='mean', errorbar=None, color='green', ax=ax)
        ax.set_title('Consumo medio por año del modelo')
        ax.set_xlabel('Año (70 = 1970)')
        ax.set_ylabel('MPG medio')
        ax.axvline(x=73, color='red', linestyle='--', alpha=0.7, label='Crisis petróleo 1973')
        ax.axvline(x=79, color='orange', linestyle='--', alpha=0.7, label='Crisis petróleo 1979')
        ax.legend(fontsize=9)
        st.pyplot(fig)
        plt.close()
        st.markdown("### 🛢️ La Crisis del Petróleo")
        st.warning("""
        En **1973**, los países productores de petróleo cortaron el
        suministro a Occidente y el precio de la gasolina se disparó.
        Las colas en las gasolineras eran kilométricas. En **1979**,
        una segunda crisis volvió a sacudir al mundo.

        El impacto fue inmediato: los fabricantes **se vieron obligados
        a diseñar vehículos mucho más eficientes**. Esto es exactamente
        lo que vemos en los datos: a partir de 1973, el consumo medio
        mejora año a año de forma notable.
        """)
        st.success("🏆 **Random Forest · R² = 0.92 · Error medio = 2.04 MPG (≈ 0.86 L/100km)**")

# ════════════════════════════════════════════════════════════
# SLIDE 8 — APLICACIÓN REAL
# ════════════════════════════════════════════════════════════
elif slide == "8. Aplicación real":
    st.title("🌍 Aplicación real e impacto")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ¿Dónde se puede aplicar este modelo?")
        with st.container(border=True):
            st.markdown("""
            **Fabricantes de coches:** antes de construir un prototipo,
            podrían introducir las características técnicas del diseño y
            obtener una estimación del consumo — ahorrando millones en
            pruebas físicas.
            """)
        with st.container(border=True):
            st.markdown("""
            **Reguladores y gobiernos:** identificar qué características
            técnicas tienen mayor impacto en las emisiones para diseñar
            normativas más efectivas.
            """)
        with st.container(border=True):
            st.markdown("""
            **Consumidores:** comparar vehículos de forma objetiva y elegir
            el más eficiente según sus necesidades.
            """)
        st.markdown("### Consideraciones éticas")
        st.warning("""
        Este modelo está entrenado con datos de los años 70–80.
        Aplicarlo directamente a vehículos modernos, híbridos o eléctricos
        daría resultados incorrectos. Toda herramienta de predicción debe
        usarse dentro de su contexto.
        """)
    with col2:
        st.markdown("### 🎮 Prueba el modelo tú mismo")
        st.markdown("Mueve los controles y obtén una predicción real del modelo:")
        with st.spinner("Cargando modelo..."):
            results, X_train, X_test, y_train, y_test, scaler = train_models()
        cyl = st.slider("Número de cilindros", 3, 8, 4)
        disp = st.slider("Tamaño del motor (cilindrada)", 68, 455, 150)
        hp = st.slider("Potencia en caballos", 46, 230, 90)
        wt = st.slider("Peso del coche (libras)", 1613, 5140, 2500)
        acc = st.slider("Aceleración (seg. hasta 60 mph)", 8.0, 24.8, 15.0)
        yr = st.slider("Año del modelo (70 = 1970)", 70, 82, 76)
        if st.button("🔮 Predecir consumo", use_container_width=True):
            ptw = hp / wt
            dpc = disp / cyl
            sample = pd.DataFrame([[disp, cyl, hp, wt, acc, yr,
                                     False, False, False, ptw, dpc]],
                                   columns=X_train.columns)
            cols_to_scale = ['displacement', 'cylinders', 'horsepower', 'weight',
                             'acceleration', 'model_year', 'power_to_weight',
                             'displacement_per_cylinder']
            sample[cols_to_scale] = scaler.transform(sample[cols_to_scale])
            pred = results['Random Forest Optimizado']['model'].predict(sample)[0]
            l100 = mpg_to_l100km(pred)
            st.success(f"⛽ Consumo estimado: **{pred:.2f} MPG**")
            st.info(f"🇪🇸 Equivalente a **{l100} litros / 100 km**")

# ════════════════════════════════════════════════════════════
# SLIDE 9 — DIFICULTADES Y APRENDIZAJES
# ════════════════════════════════════════════════════════════
elif slide == "9. Dificultades y aprendizajes":
    st.title("🧗 Dificultades encontradas y aprendizajes")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Principales dificultades")
        with st.container(border=True):
            st.markdown("""
            **Límite de la API:** la fuente de datos online tiene un límite
            diario de peticiones, lo que obligó a pausar el trabajo y
            reorganizar los tiempos.
            """)
        with st.container(border=True):
            st.markdown("""
            **Variables muy relacionadas entre sí:** el peso, la cilindrada,
            los cilindros y la potencia están tan correlacionados que es
            difícil saber cuál influye realmente y cuál solo acompaña.
            """)
        with st.container(border=True):
            st.markdown("""
            **Dataset pequeño:** con solo 398 registros, el modelo tiene
            menos margen para generalizar bien a casos que no ha visto.
            """)
        with st.container(border=True):
            st.markdown("""
            **Compatibilidad entre versiones:** algunas funciones de las
            librerías han cambiado y generaron avisos que había que resolver.
            """)
    with col2:
        st.markdown("### Aprendizajes clave")
        st.success("""
        ✅ Los datos limpios y bien preparados son más importantes que
        el modelo en sí — sin una buena base, ningún algoritmo funciona
        correctamente.
        """)
        st.success("""
        ✅ Escalar los datos no es opcional en ciertos modelos: sin
        normalización, el KNN habría ignorado variables importantes
        simplemente por tener valores más pequeños.
        """)
        st.success("""
        ✅ Comparar varios modelos con métricas objetivas es la única
        forma de saber cuál funciona mejor — la intuición puede engañar.
        """)
        st.success("""
        ✅ La optimización automática de parámetros no siempre mejora
        el resultado: a veces el modelo ya está bien ajustado desde
        el principio.
        """)

# ════════════════════════════════════════════════════════════
# SLIDE 10 — TRABAJO FUTURO
# ════════════════════════════════════════════════════════════
elif slide == "10. Trabajo futuro":
    st.title("🚀 Trabajo futuro y mejoras posibles")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Próximos pasos")
        with st.container(border=True):
            st.markdown("""
            **Modelos más avanzados:** probar Gradient Boosting y XGBoost
            para ver si superan el R² actual de 0.92.
            """)
        with st.container(border=True):
            st.markdown("""
            **Datos más actuales:** ampliar el dataset con vehículos
            modernos, híbridos y eléctricos para construir un modelo
            aplicable hoy.
            """)
        with st.container(border=True):
            st.markdown("""
            **Despliegue real:** convertir el modelo en un servicio web
            que cualquier fabricante pueda consultar en tiempo real.
            """)
        st.markdown("### Mejoras técnicas")
        st.info("Aplicar reducción de dimensionalidad para separar mejor las variables que están muy correlacionadas entre sí.")
        st.info("Añadir herramientas de interpretabilidad que expliquen, predicción a predicción, por qué el modelo ha dado ese resultado concreto.")
    with col2:
        st.markdown("### Conclusión final")
        with st.container(border=True):
            st.markdown("""
            Volviendo a la pregunta con la que empezamos:

            *"¿Es posible predecir cuánto consume un coche únicamente
            a partir de sus características técnicas?"*

            **La respuesta es sí.**

            El modelo Random Forest es capaz de predecir el consumo
            con un margen de error de apenas 2 MPG, explicando el 92%
            de la variación en el consumo real.

            Esto demuestra que las características técnicas de un
            vehículo no son solo números — son la huella digital
            de su eficiencia.
            """)
        st.success("🏆 Random Forest · R² = 0.92 · Error medio = 2.04 MPG")

# ════════════════════════════════════════════════════════════
# SLIDE 11 — CIERRE
# ════════════════════════════════════════════════════════════
elif slide == "11. Cierre":
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center'>
            <h1>🚗</h1>
            <h2>¡Gracias! 🙌</h2>
            <h3>Q&A</h3>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    pass