import pandas as pd
from ucimlrepo import fetch_ucirepo


def load_data():
    """Carga el dataset Auto MPG desde UCI."""
    auto_mpg = fetch_ucirepo(id=9)
    df = pd.concat([auto_mpg.data.features, auto_mpg.data.targets], axis=1)
    return df


def clean_data(df):
    """
    Limpia el dataset:
    - Imputa los 6 valores nulos de horsepower con la mediana.
    - Convierte horsepower a numérico.
    """
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
    return df


def encode_origin(df):
    """Convierte la columna 'origin' en variables dummy (One Hot Encoding)."""
    df = pd.get_dummies(df, columns=['origin'], drop_first=False)
    return df