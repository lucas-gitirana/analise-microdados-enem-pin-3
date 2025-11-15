import pandas as pd
from sqlalchemy import text

from ..database.connection import engine

def load_data():
    tabela = "dados_enem_consolidado"
    query = f"SELECT * FROM {tabela}"
    df = pd.read_sql(query, engine)
    return df

def saveData_BD(df, nomeTabela:str):
    df.to_sql(nomeTabela, engine, if_exists='replace', index=False)
    print(f"{len(df)} registros salvos na tabela '{nomeTabela}'")
