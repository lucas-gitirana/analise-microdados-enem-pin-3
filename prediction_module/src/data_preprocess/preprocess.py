import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

from ..database.operations import load_data, saveData_BD

def preprocess_data(df: pd.DataFrame, categorizar_colunas=True):
    df = removerColunas(df)
    df = tratarDadosFaltantes(df)

    encoders = {}
    if categorizar_colunas:
        df = categorizar_nota(df)
        df, encoders = codificarVariaveisCategoricas(df)

    # Preencher NaNs restantes com -1 (para algoritmos que não aceitam NaN)
    df.fillna(-1, inplace=True)

    # with open("prediction_module/src/saved_model/encoders.pkl", "wb") as f:
    #     joblib.dump(encoders, f)

    return df, encoders


def removerColunas(df):
    colunas_remover = [
        "NU_INSCRICAO",
        "CO_PROVA_CH", "CO_PROVA_CN" , "CO_PROVA_LC", "CO_PROVA_MT",
        "NU_NOTA_COMP1", "NU_NOTA_COMP2", "NU_NOTA_COMP3", "NU_NOTA_COMP4", "NU_NOTA_COMP5",
        "TP_STATUS_REDACAO", "TP_ST_CONCLUSAO",
        "TP_PRESENCA_MT", "TP_PRESENCA_LC", "TP_PRESENCA_CN", "TP_PRESENCA_CH",
        "CO_UF_ESC", "CO_UF_PROVA",
        "CO_MUNICIPIO_PROVA", "CO_MUNICIPIO_ESC",  "TP_NACIONALIDADE",
        "TX_GABARITO_CH", "TX_GABARITO_CN", "TX_GABARITO_LC", "TX_GABARITO_MT",
        "TX_RESPOSTAS_CH", "TX_RESPOSTAS_CN", "TX_RESPOSTAS_LC", "TX_RESPOSTAS_MT"
    ]
    return df.drop(columns=[c for c in colunas_remover if c in df.columns], errors="ignore")

# Manter as colunas NaN e tratar como ND as categóricas --> Object
def tratarDadosFaltantes(df):
    colunas_notas = ["NU_NOTA_CH","NU_NOTA_CN","NU_NOTA_LC","NU_NOTA_MT","NU_NOTA_REDACAO"]
    for col in colunas_notas:
        if col in df.columns:
            df[col] = df[col].replace(-1, np.nan)

    # Colunas q comecam com NO_ e categóricas
    no_cols = [c for c in df.columns if c.startswith("NO_") and df[c].dtype == "object"]
    for col in no_cols:
            df[col] = df[col].fillna("ND").astype(str)

    # Outras categóricas
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if col not in no_cols:
            df[col] = df[col].fillna("ND").astype(str)

    return df

def categorizar_nota(df):
    colunas_notas = ["NU_NOTA_CH", "NU_NOTA_CN", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
    for col in colunas_notas:
        if col in df.columns:
            serie = df[col].dropna()
            if len(serie) > 0:
                # Cálculo dos limites dos percentis
                p15 = np.nanpercentile(serie, 15)
                p85 = np.nanpercentile(serie, 85)

                # Aplicar categorização conforme os percentis
                df[col] = df[col].apply(
                    lambda x: -1 if pd.isna(x)
                    else (0 if x < p15 else (2 if x > p85 else 1))
                )
            else:
                df[col] = -1
    return df


def codificarVariaveisCategoricas(df):
    #Transforma variáveis categóricas em números usando LabelEncoder
    encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


if __name__ == "__main__":
    # Select dados
    df = load_data()

    # Pré-processar
    df_processado, _ = preprocess_data(df)

    # Save DataBase
    saveData_BD(df_processado, 'dados_enem_consolidado')

    print("Dados processados com sucesso!")
    print(df_processado.head(30))
    print(df_processado.describe())
