import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def feature_engineering(df):
    """
    Crea nuevas variables a partir de las existentes:
    - power_to_weight: ratio potencia/peso
    - displacement_per_cylinder: cilindrada media por cilindro
    """
    df['power_to_weight'] = df['horsepower'] / df['weight']
    df['displacement_per_cylinder'] = df['displacement'] / df['cylinders']
    return df


def split_data(df, test_size=0.2, random_state=42):
    """Separa el dataset en features/target y train/test."""
    X = df.drop(columns=['mpg'])
    y = df['mpg']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Aplica StandardScaler a las columnas numéricas continuas.
    Las columnas binarias de origin no se escalan.
    """
    cols_to_scale = ['displacement', 'cylinders', 'horsepower', 'weight',
                     'acceleration', 'model_year', 'power_to_weight',
                     'displacement_per_cylinder']

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    return X_train, X_test, scaler