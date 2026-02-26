# preprocessor.py
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class DemandPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform_demand(self, df: pd.DataFrame):
        features = ['demand_kg', 'supply_kg', 'price_per_kg', 'weather_index']
        scaled = self.scaler.fit_transform(df[features])
        df_scaled = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in features], index=df.index)
        return pd.concat([df, df_scaled], axis=1)

    def create_sequences(self, df_scaled, seq_len=30):
        data = df_scaled[[c for c in df_scaled if '_scaled' in c]].values
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len, 0])   # predict demand_scaled
        return np.array(X), np.array(y
