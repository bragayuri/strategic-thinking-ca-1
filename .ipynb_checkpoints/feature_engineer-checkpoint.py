import pandas as pd
import numpy as np
import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# feature_engineering.py

def add_features(X):
    X = X.copy()
    X['age'] = 2025 - X['year']
    X['mileage_per_year'] = X['Kilometer'] / X['age'].replace(0, 1)
    X['is_luxury_brand'] = X['make'].isin(['BMW', 'Audi', 'Mercedes', 'Lexus', 'Porsche']).astype(int)
    return X.drop(columns=['year', 'Kilometer'])


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, current_year=None):
        self.current_year = current_year or datetime.datetime.now().year

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['year']      = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
        df['Kilometer'] = pd.to_numeric(df['Kilometer'], errors='coerce').fillna(0)
        df['age']              = self.current_year - df['year']
        df['mileage_per_year'] = df['Kilometer'] / df['age'].replace(0, 1)
        df['Kilometer_log']    = np.log1p(df['Kilometer'])
        return df[['make', 'model', 'fuel_type', 'transmission', 'age', 'mileage_per_year', 'Kilometer_log']]
