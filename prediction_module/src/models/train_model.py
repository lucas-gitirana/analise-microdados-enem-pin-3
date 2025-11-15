from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from imblearn.over_sampling import SMOTE
from database.operations import load_data

import joblib
import os
import json

#MODELS
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def preprocess_data(df, target_col, nota_cols):
     #Filtragem de dados i.e remoção de dados ausentes
    df_filtered = df[df[target_col] != -1].copy()

    #Ignorar demais notas (modelo não pode treinar com as demais notas) ==> evitar vazamento de informação
    ignore_cols = [col for col in nota_cols if col != target_col]
    print(f"Ignorando colunas: {ignore_cols}")

    #Evitação de vazamento de informação
    X = df_filtered.drop(columns=[target_col] + ignore_cols, errors="ignore")
    y = df_filtered[target_col]

    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    return X_res, y_res


def train_model(X_train, y_train):
    """"
        Treina o modelo RandomForest com RandomizedSearchCV

        @param X_train: Conjunto de treinamento (features)
        @param y_train: Rótulos de treinamento
        @return: modelo treinado
    """

    #OBS: Se overfitting → aumente min_samples_leaf, min_samples_split ou reduza max_depth.
    param_dist = {
        'n_estimators': [200, 400, 600],
        'max_depth': [10, 12, 15, 18],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    model = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1_macro',
        cv=3,
        verbose=2
    )

    #model = DecisionTreeClassifier(max_depth=10, class_weight="balanced",random_state=42)

    #Treinamento
    model.fit(
        X_train, y_train
    )

    return model


def metrics(model, X_test, y_test, X_columns):
    """
        Avaliação das métricas do modelo e apresentação das importância das features

        @param model: modelo treinado
        @param X_test: conjunto de teste
        @param y_test: rótulos reais do conjunto de teste
        @param X_columns: nomes das colunas de entrada
    """
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    #Importancia das variáveis (Mostra as 30 mais importantes)
    importances = pd.Series(model.best_estimator_.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False).head(30))

    #Visualização
    importances.sort_values(ascending=True).tail(15).plot(kind="barh")

def save_model(model, target_col):
    #CAminho
    save_dir = "./saved_model"
    os.makedirs(save_dir, exist_ok=True)

    model_filename = os.path.join(save_dir, f"randomForest_{target_col}.pkl")

    #Salvar
    joblib.dump(model.best_estimator_, model_filename)
    print(f"Modelo salvo com sucesso em: {model_filename}")


if __name__ == "__main__":
    #Loading Dados
    df = load_data()

    #Coluna de Treinamento
    target_col = "NU_NOTA_CH"
    print(df[target_col].value_counts())

    #Listas de colunas (target_col)
    nota_cols = ["NU_NOTA_CH", "NU_NOTA_MT", "NU_NOTA_CN", "NU_NOTA_LC", "NU_NOTA_REDACAO"]

    #Pré-processamento/Manipulação dos dados
    X, y = preprocess_data(df, target_col, nota_cols)

    with open("./saved_model/columns.json", "w") as f:
        json.dump(list(X.columns), f)

    #Divisão de 80% treino e 20% teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Treinamento
    model = train_model(X_train, y_train)

    #Metricas
    metrics(model, X_test, y_test, X.columns)

    #Salvar Modelo
    save_model(model, target_col)