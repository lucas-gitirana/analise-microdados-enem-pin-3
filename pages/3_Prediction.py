import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prediction_module.src.models.predictor import EnemPredictor
import traceback

# --- Estilo customizado ---
st.markdown("""
    <style>
        body { background-color: #111111; color: #fff; }
        .stTabs [role="tablist"] button {
            font-size: 16px;
            font-weight: bold;
        }
        div[data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: bold;
        }
        .big-button button {
            font-size: 18px;
            font-weight: bold;
            padding: 0.6em 2em;
            border-radius: 10px;
        }
        .stRadio > label {
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Predição de Desempenho")

# -------------------------
# Carrega predictors (cache)
# -------------------------
@st.cache_resource
def load_predictors():
    base = "prediction_module/src/saved_model"
    encoders_path = f"{base}/encoders.pkl"
    cols_path = f"{base}/columns.json"

    # Ajuste os caminhos se necessário; atualmente todos apontam para o mesmo model
    # return {
    #     "CH": EnemPredictor(f"{base}/randomForest_NU_NOTA_CH.pkl", encoders_path=encoders_path, cols_path=cols_path),
    #     "CN": EnemPredictor(f"{base}/randomForest_NU_NOTA_CN.pkl", encoders_path=encoders_path, cols_path=cols_path),
    #     "LC": EnemPredictor(f"{base}/randomForest_NU_NOTA_LC.pkl", encoders_path=encoders_path, cols_path=cols_path),
    #     "MT": EnemPredictor(f"{base}/randomForest_NU_NOTA_MT.pkl", encoders_path=encoders_path, cols_path=cols_path),
    #     "RED": EnemPredictor(f"{base}/randomForest_NU_NOTA_REDACAO.pkl", encoders_path=encoders_path, cols_path=cols_path),
    # }

    return {
        "CH": EnemPredictor(f"{base}/randomForest_NU_NOTA_CH.pkl", encoders_path=encoders_path, cols_path=cols_path),
        "CN": EnemPredictor(f"{base}/randomForest_NU_NOTA_CH.pkl", encoders_path=encoders_path, cols_path=cols_path),
        "LC": EnemPredictor(f"{base}/randomForest_NU_NOTA_CH.pkl", encoders_path=encoders_path, cols_path=cols_path),
        "MT": EnemPredictor(f"{base}/randomForest_NU_NOTA_CH.pkl", encoders_path=encoders_path, cols_path=cols_path),
        "RED": EnemPredictor(f"{base}/randomForest_NU_NOTA_CH.pkl", encoders_path=encoders_path, cols_path=cols_path),
    }

models = load_predictors()

MAP_SEXO = {
    "Masculino": "M",
    "Feminino": "F",
    "Prefiro não informar": "ND"
}

MAP_RENDA = {
    "Até 1 SM": "B",
    "1-3 SM": "C",
    "3-5 SM": "D",
    "Mais de 5 SM": "E"
}

MAP_ESCOLARIDADE = {
    "Fundamental": "A",
    "Ensino Médio": "C",
    "Superior": "E",
    "Pós-graduação": "G",
    "Não informado": "ND"
}

MAP_TP_ESCOLA = {
    "Pública": "1",
    "Privada": "2",
    "Federal": "3"
}

MAP_BOOL = {"Sim": "1", "Não": "0"}

# Função wrapper que tenta prever e captura erros com diagnóstico
def predict_notas_real(inputs):
    results = {}
    diagnostics = {}
    for area, model_key in [
        ("Linguagens e Códigos", "LC"),
        ("Ciências Humanas", "CH"),
        ("Ciências da Natureza", "CN"),
        ("Matemática", "MT"),
        ("Redação", "RED")
    ]:
        predictor = models[model_key]
        try:
            # Tenta preparar e prever
            # mostramos o DataFrame preparado para debug se precisar
            df_prepared = predictor.preprocess_for_prediction(inputs)
            # guarda diagnóstico parcial
            diagnostics[area] = {
                "df_head": df_prepared.head(3).to_dict(orient="list"),
                "dtypes": df_prepared.dtypes.apply(lambda x: str(x)).to_dict()
            }
            pred = predictor.model.predict(df_prepared)[0]
            results[area] = int(pred)
        except Exception as e:
            # Guarda erro para exibir na UI
            results[area] = None
            diagnostics[area] = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    return results, diagnostics

# Inicializa session_state
if "notas" not in st.session_state:
    st.session_state.notas = {
        "Linguagens e Códigos": 0,
        "Ciências Humanas": 0,
        "Ciências da Natureza": 0,
        "Matemática": 0,
        "Redação": 0
    }

default_values = {
    "sexo": "Masculino",
    "idade": 20,
    "renda": "1-3 SM",
    "esc_pai": "Ensino Médio",
    "esc_mae": "Superior",
    "escola": "Pública",
    # removi IN_INTERNET e IN_COMPUTADOR porque não existem no modelo treinado
}

for k, v in default_values.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Tabs
tab1, tab2 = st.tabs(["🎯 Simulação de Resultado", "📌 Variáveis Importantes"])

# ================= TAB 1 =================
with tab1:
    st.info("Preencha os campos socioeconômicos e veja a previsão dinâmica de desempenho em cada área do ENEM.")

    # --- Cards com métricas ---
    st.subheader("📊 Resultado da Predição")
    cards_placeholder = st.empty()

    def render_cards():
        with cards_placeholder:
            cols = st.columns(5)
            for (area, nota), col in zip(st.session_state.notas.items(), cols):
                display = nota if (nota is not None) else "—"
                col.metric(area, display)

    # Render inicial (sempre uma vez só)
    render_cards()

    # --- Formulário ---
    st.subheader("🧑‍🎓 Dados do Participante")
    with st.form("prediction_form"):
        sexo = st.radio("Sexo", ["Masculino", "Feminino", "Prefiro não informar"],
                        horizontal=True, index=["Masculino", "Feminino", "Prefiro não informar"].index(st.session_state.sexo))

        idade = st.slider("Idade", 0, 100, st.session_state.idade)

        renda = st.radio("Renda Familiar", ["Até 1 SM", "1-3 SM", "3-5 SM", "Mais de 5 SM"],
                        horizontal=True, index=["Até 1 SM", "1-3 SM", "3-5 SM", "Mais de 5 SM"].index(st.session_state.renda))

        col1, col2 = st.columns(2)
        with col1:
            esc_pai = st.select_slider(
                "Escolaridade do Pai",
                options=["Fundamental", "Ensino Médio", "Superior", "Pós-graduação", "Não informado"],
                value=st.session_state.esc_pai
            )
        with col2:
            esc_mae = st.select_slider(
                "Escolaridade da Mãe",
                options=["Fundamental", "Ensino Médio", "Superior", "Pós-graduação", "Não informado"],
                value=st.session_state.esc_mae
            )

        escola = st.radio("Tipo da Escola", ["Pública", "Privada", "Federal"],
                        horizontal=True, index=["Pública", "Privada", "Federal"].index(st.session_state.escola))

        col1, col2 = st.columns(2)
        with col1:
            limpar = st.form_submit_button("🗑️ Limpar")
        with col2:
            submitted = st.form_submit_button("📊 Gerar Nova Previsão")

    # --- Ações dos botões ---
    if submitted:
        st.session_state.update({
            "sexo": sexo,
            "idade": idade,
            "renda": renda,
            "esc_pai": esc_pai,
            "esc_mae": esc_mae,
            "escola": escola,
        })

        # Monta o dicionário de inputs (removi IN_INTERNET / IN_COMPUTADOR)
        # Atenção: os nomes das chaves devem ser exatamente os nomes das colunas originais antes do LabelEncoder
        inputs_model = {
            "TP_SEXO": MAP_SEXO[sexo],
            "TP_FAIXA_ETARIA": idade,
            "Q006": MAP_RENDA[renda],
            "Q001": MAP_ESCOLARIDADE[esc_pai],
            "Q002": MAP_ESCOLARIDADE[esc_mae],
            "TP_ESCOLA": MAP_TP_ESCOLA[escola],
            # "IN_INTERNET": MAP_BOOL[internet],
            # "IN_COMPUTADOR": MAP_BOOL[computador],
        }

        # Prediz e captura possíveis diagnósticos
        notas_pred, diagnostics = predict_notas_real(inputs_model)

        # Se houve erro em alguma área, mostra diagnóstico
        any_error = any(v is None for v in notas_pred.values())
        if any_error:
            st.error("Ocorreu um erro em uma ou mais predições. Veja o diagnóstico abaixo (útil para depuração).")
            for area, diag in diagnostics.items():
                st.markdown(f"**{area}**")
                if "error" in diag:
                    st.code(diag["error"])
                    st.text_area(f"Traceback - {area}", diag.get("traceback",""), height=200)
                else:
                    st.write("DataFrame preparado (head):")
                    st.json(diag["df_head"])
                    st.write("dtypes:")
                    st.json(diag["dtypes"])
            # atualiza session_state.notas com o que foi calculado (None onde falhou)
            st.session_state.notas = notas_pred
        else:
            # tudo ok -> atualiza notas e renderiza
            st.session_state.notas = notas_pred
            render_cards()

    if limpar:
        for k, v in default_values.items():
            st.session_state[k] = v
        st.session_state.notas = {
            "Linguagens e Códigos": 0,
            "Ciências Humanas": 0,
            "Ciências da Natureza": 0,
            "Matemática": 0,
            "Redação": 0
        }
        st.rerun()

    # --- Gráfico ---
    st.subheader("📈 Visualização Gráfica")
    df_notas = pd.DataFrame({
        "Área": list(st.session_state.notas.keys()),
        "Nota Prevista": [v if v is not None else 0 for v in st.session_state.notas.values()]
    })
    fig = px.bar(df_notas, x="Área", y="Nota Prevista", text="Nota Prevista",
                color="Área", title="Notas Previstas por Área do ENEM")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.info("Veja as variáveis que mais impactam no resultado do modelo.")

    variaveis = [
        {"nome": "Renda Familiar", "descricao": "A renda está fortemente relacionada ao acesso a materiais de estudo e cursos preparatórios."},
        {"nome": "Tipo de Escola", "descricao": "Estudantes de escolas privadas, em média, têm acesso a mais recursos de aprendizagem."},
        {"nome": "Escolaridade da Mãe", "descricao": "Pesquisas indicam que o nível de escolaridade da mãe tem alta correlação com o desempenho escolar."},
        {"nome": "Sexo", "descricao": "Diferenças de desempenho entre homens e mulheres são observadas em algumas áreas do ENEM."},
    ]

    cols = st.columns(2)

    for i, var in enumerate(variaveis):
        # Criar uma função dialog única para cada variável
        @st.dialog(var["nome"])
        def abrir_dialogo(v=var):
            st.markdown(f"### {v['nome']}")
            st.write(v["descricao"])
            st.info("🔎 Aqui futuramente poderemos mostrar gráficos explicativos do impacto desta variável.")
            # Exemplo de gráfico fake
            st.bar_chart({"Impacto": np.random.randint(1, 100, size=5)})

        with cols[i % 2]:
            if st.button(var["nome"], use_container_width=True, key=f"btn_{i}"):
                abrir_dialogo()
