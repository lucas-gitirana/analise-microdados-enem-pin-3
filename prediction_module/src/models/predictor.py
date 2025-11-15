import joblib
import pandas as pd
import json

from ..data_preprocess.preprocess import preprocess_data

class EnemPredictor:

    def __init__(self, model_path, encoders_path, cols_path):
        self.model = joblib.load(model_path)

        # encoders usados no treinamento
        self.encoders = joblib.load(encoders_path)

        # lista de colunas esperadas
        with open(cols_path, "r") as f:
            self.columns_expected = json.load(f)

    # ----------------------------------------------------
    # 1. Pré-processar exatamente como no treinamento
    # ----------------------------------------------------
    def preprocess_for_prediction(self, input_dict):

        df = pd.DataFrame([input_dict])

        # === 1) APLICAR OS MAPEAMENTOS MANUAIS ===
        from ..data_preprocess.preprocess import MAP_SEXO, MAP_RENDA, MAP_ESCOLARIDADE, MAP_TP_ESCOLA, MAP_BOOL

        mapping_cols = {
            "TP_SEXO": MAP_SEXO,
            "TP_FAIXA_RENDA": MAP_RENDA,
            "TP_ESCOLARIDADE_RESP": MAP_ESCOLARIDADE,
            "TP_ESCOLA": MAP_TP_ESCOLA,
            "IN_COMPUTADOR": MAP_BOOL,
            "IN_INTERNET": MAP_BOOL
        }

        for col, mapping in mapping_cols.items():
            if col in df:
                df[col] = df[col].map(mapping).fillna("ND").astype(str)

        # === 2) PRÉ-PROCESSAMENTO (SEM CATEGORIZAR NOTAS) ===
        df, _ = preprocess_data(df, categorizar_colunas=True)

        # === 3) APLICAR LABELENCODERS DO TREINAMENTO ===
        for col, encoder in self.encoders.items():
            if col in df:
                df[col] = encoder.transform(df[col].astype(str))

        # === 4) GARANTIR TODAS AS COLUNAS ===
        for col in self.columns_expected:
            if col not in df:
                df[col] = -1

        df = df[self.columns_expected]

        return df

    # ----------------------------------------------------
    # 2. Predição
    # ----------------------------------------------------
    def predict(self, inputs_dict):
        df = self.preprocess_for_prediction(inputs_dict)

        print("\n=== DEBUG DF ENVIADO AO MODELO ===")
        print(df)

        pred = self.model.predict(df)[0]
        return int(pred)
