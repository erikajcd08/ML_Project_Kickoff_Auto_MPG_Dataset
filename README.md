# ML_Project_Kickoff_Auto_MPG_Dataset

### Predicción del Consumo de Combustible en Coches (Auto MPG)
---

## Objetivo del proyecto

Este proyecto tiene como objetivo predecir el consumo de combustible de un coche (medido en millas por galón, MPG) a partir de sus características técnicas, utilizando diferentes modelos de Machine Learning.

---

## Dataset

**Auto MPG Dataset** — UCI Machine Learning Repository  
Fuente: https://archive.ics.uci.edu/dataset/9/auto+mpg  

**Variables analizadas:**
- `mpg` → Variable objetivo: consumo en millas por galón
- `cylinders` → Número de cilindros del motor
- `displacement` → Cilindrada del motor (pulgadas³)
- `horsepower` → Caballos de potencia
- `weight` → Peso del coche (libras)
- `acceleration` → Aceleración (segundos en llegar a 60 mph)
- `model_year` → Año del modelo (70-82, equivale a 1970-1982)
- `origin` → País de origen (1=USA, 2=Europa, 3=Japón)

---

## Pregunta de investigación

¿Es posible predecir el consumo de un coche únicamente a partir de sus características técnicas?

---

## Hipótesis

- A mayor peso del coche, mayor consumo (menor MPG)
- A mayor cilindrada, mayor consumo
- Los coches más modernos son más eficientes
- Los coches europeos y japoneses son más eficientes que los americanos

---

## Proceso de análisis

- **Python:** Limpieza, feature engineering, modelado y visualización
- **Scikit-learn:** Modelos de Machine Learning y evaluación
- **Pandas / NumPy:** Manipulación de datos
- **Matplotlib / Seaborn:** Visualización
- **Jupyter Notebook:** Entorno de desarrollo
- **GitHub:** Control de versiones

---

## Metodología

El proyecto siguió cuatro fases:

**Día 1 — EDA:** 
- Exploración y visualización del dataset. 
- Análisis de distribuciones, correlaciones y tendencias temporales.

**Día 2 — Preparación:**
-Limpieza de nulos, One Hot Encoding de la variable `origin`, feature engineering (creación de `power_to_weight` y `displacement_per_cylinder`), Feature Scaling con StandardScaler y Train/Test Split (80/20).

**Día 3 — Modelos base:** 
- Entrenamiento y evaluación de KNN Regressor, Linear Regression y Decision Tree Regressor. Comparación mediante R² y RMSE.

**Día 4 — Modelos avanzados:** 
- Random Forest con optimización de hiperparámetros mediante GridSearchCV con Cross Validation de 5 folds. 
- Análisis de Feature Importance.

---

## Principales resultados

**Comparativa de modelos:**

| Modelo | R² | RMSE |
|---|---|---|
| KNN Regressor | 0.8907 | 2.4244 |
| Linear Regression | 0.8746 | 2.5963 |
| Decision Tree | 0.8142 | 3.1603 |
| Random Forest | 0.9223 | 2.0443 |
| Random Forest Optimizado | 0.9222 | 2.0449 |

**Feature Importance:**
- `displacement` es la variable más influyente en el consumo (≈39%)
- Le siguen `weight`, `cylinders`, `horsepower` y `model_year`
- El origen del coche apenas influye una vez controladas las características técnicas

---

## Conclusiones

- El modelo **Random Forest** es el mejor con un R² de 0.92, lo que significa que explica el 92% de la variación en el consumo.
- El modelo se equivoca de media **2.04 MPG**, un resultado muy bueno.
- Las hipótesis se confirman: peso, cilindrada y potencia son los principales predictores del consumo.
- Los coches más modernos son significativamente más eficientes, reflejo de la crisis del petróleo de los años 70.
- El GridSearchCV no mejoró significativamente el Random Forest base, lo que indica que el modelo ya estaba bien ajustado por defecto.

---

## Obstáculos encontrados

- La API de UCI tiene límite de peticiones diarias, lo que obligó a esperar para recargar los datos en alguna ocasión.
- El dataset es pequeño (398 instancias), lo que limita la capacidad de generalización de los modelos.
- Alta multicolinealidad entre variables como `displacement`, `cylinders`, `horsepower` y `weight`.

---

## Próximos pasos

- Probar modelos más avanzados como Gradient Boosting o XGBoost.
- Ampliar el dataset con coches más modernos para ver si las tendencias se mantienen.
- Explorar si el modelo funciona bien para coches eléctricos o híbridos con variables adicionales.

---

## Autora

- Erika Campo Díaz

---

## Enlace a la presentación
Streamlit: https://mlprojectkickoffautompgdataset-jkwnszufnnzpp9ayftqnqq.streamlit.app/