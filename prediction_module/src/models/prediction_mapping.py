# ==========================
# MAPEAMENTOS DE ENTRADA
# ==========================

MAP_SEXO = {
    "Masculino": "M",
    "Feminino": "F",
    "Prefiro não informar": "ND",
}

MAP_RENDA = {
    "Até 1 SM": "A",
    "1-3 SM": "B",
    "3-5 SM": "C",
    "Mais de 5 SM": "D",
}

MAP_ESCOLARIDADE = {
    "Fundamental": "1",
    "Ensino Médio": "2",
    "Superior": "3",
    "Pós-graduação": "4",
    "Não informado": "ND",
}

MAP_TIPO_ESCOLA = {
    "Pública": "1",
    "Privada": "2",
    "Federal": "3",
}

MAP_BINARY = {
    "Sim": "1",
    "Não": "0",
}

# lista final de campos esperados pelo modelo
FEATURES = [
    "TP_SEXO",
    "TP_FAIXA_ETARIA",
    "Q006",
    "Q001",
    "Q002",
    "TP_ESCOLA",
    "IN_INTERNET",
    "IN_COMPUTADOR",
]
