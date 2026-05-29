import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error


def train_knn(X_train, y_train, n_neighbors=5):
    """Entrena un modelo KNN Regressor."""
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def train_linear_regression(X_train, y_train):
    """Entrena un modelo de Regresión Lineal."""
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


def train_decision_tree(X_train, y_train, random_state=42):
    """Entrena un modelo Decision Tree Regressor."""
    dt = DecisionTreeRegressor(random_state=random_state)
    dt.fit(X_train, y_train)
    return dt


def train_random_forest(X_train, y_train, random_state=42):
    """Entrena un modelo Random Forest Regressor."""
    rf = RandomForestRegressor(random_state=random_state)
    rf.fit(X_train, y_train)
    return rf


def tune_random_forest(X_train, y_train, random_state=42):
    """
    Optimiza los hiperparámetros del Random Forest
    mediante GridSearchCV con Cross Validation de 5 folds.
    """
    params = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        params, cv=5, scoring='r2'
    )
    grid.fit(X_train, y_train)
    print("Mejores parámetros:", grid.best_params_)
    print("Mejor R² en CV:", round(grid.best_score_, 4))
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test, model_name):
    """Evalúa un modelo y devuelve R² y RMSE."""
    y_pred = model.predict(X_test)
    r2 = round(r2_score(y_test, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
    print(f"=== {model_name} ===")
    print(f"R²: {r2}")
    print(f"RMSE: {rmse}\n")
    return r2, rmse


def plot_feature_importance(model, feature_names):
    """Visualiza la importancia de las variables del Random Forest."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances_sorted = importances.sort_values()

    plt.figure(figsize=(8, 6))
    importances_sorted.plot(kind='barh', color='steelblue')
    plt.title('Importancia de las variables (Random Forest)')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.show()