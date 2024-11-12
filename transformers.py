from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RemoveDuplicatesTransformer:
    def __init__(self):
        pass
    
    def transform(self, df):
        # Eliminamos las filas duplicadas y retornamos el dataframe
        return df.drop_duplicates()

class FilterDataTransformer:
    def __init__(self):
        pass
    
    def transform(self, df):
        # Aplicamos los filtros para 'average_daily_rate' y 'adults'
        df = df[df['average_daily_rate'] >= 0]
        df = df[df['adults'] > 0]
        return df

class BoolDataTransformer:
    def __init__(self):
        pass
    
    def transform(self, df):
        # Convertimos las columnas de 'children' y 'required_car_parking_spaces' en binarias
        df['children'] = np.where(df.children== 'children', 1, 0)
        df['required_car_parking_spaces'] = np.where(df.required_car_parking_spaces== 'parking', 1, 0)
        return df

class DateColsTransformer:
    def __init__(self):
        pass
    
    def transform(self, df):
        # Convertimos la columna de 'arrival_date' en datetime y creamos dos nuevas columnas
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        df['year_month'] = df['arrival_date'].dt.to_period('M').astype(str)
        df['month'] = df['arrival_date'].dt.month
        # Eliminamos las filas con la fecha de arribo porque solo utilizaremos los meses como features
        df = df.drop(columns=['arrival_date', 'year_month'])
        return df

class TargetTransformer:
    def __init__(self):
        pass
    
    def transform(self, df):
        # Create target column
        df = df.rename(columns={"children": "target"})
        
        return df

class TotalNightsTransformer:
    def __init__(self):
        pass
    
    def transform(self, df):
        # Create  column
        df['total_nights'] = df.stays_in_weekend_nights + df.stays_in_week_nights
        
        return df

class CountryTransformer:
    def __init__(self):
        pass
    
    def transform(self, df):
        # Create  column
        country_counts = df['country'].value_counts(normalize=True) * 100
        countries_to_keep = country_counts[country_counts >= 2].index.tolist()

        # Replace countries under 2% by "Others"
        df['country'] = df['country'].apply(lambda x: x if x in countries_to_keep else 'Others')
        
        # Replace null values by "Others"
        df['country'].fillna('Others', inplace=True)
        
        return df

