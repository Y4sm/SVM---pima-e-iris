"""Configuração e tratamento dos conjuntos de dados (Pima Indians Diabetes e Iris)."""
from pathlib import Path
import numpy as np
import pandas as pd

PASTA = Path(__file__).parent
DADOS = PASTA / "dados"
SAIDA = PASTA / "modelos"


def zeros_para_nan(X, atributos, zeros):
    """Troca zeros fisiologicamente impossíveis por NaN (depois imputados pela mediana)."""
    X = pd.DataFrame(X, columns=atributos).copy()
    X[zeros] = X[zeros].replace(0, np.nan)
    return X


# ---------------------------------------------------------------- leitura
PIMA_COLUNAS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


def _ler_pima(caminho) -> pd.DataFrame:
    bruto = pd.read_csv(caminho, header=None)
    if pd.to_numeric(bruto.iloc[0], errors="coerce").isna().any():      # 1ª linha é cabeçalho
        df = pd.read_csv(caminho)
        df.columns = [c.strip() for c in df.columns]
        faltam = set(PIMA_COLUNAS) - set(df.columns)
        if faltam:
            raise ValueError(f"Colunas ausentes no CSV: {sorted(faltam)}")
        df = df[PIMA_COLUNAS]
    else:
        df = bruto
        df.columns = PIMA_COLUNAS
    return df.apply(pd.to_numeric, errors="coerce")


def _ler_iris(caminho) -> pd.DataFrame:
    """4 primeiras colunas = medidas, última = espécie; ignora coluna Id."""
    bruto = pd.read_csv(caminho, header=None)
    if pd.to_numeric(bruto.iloc[0, :4], errors="coerce").isna().any():
        bruto = pd.read_csv(caminho)
    bruto = bruto[[c for c in bruto.columns if str(c).strip().lower() != "id"]]
    df = bruto.iloc[:, :4].apply(pd.to_numeric, errors="coerce")
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    def cod(v):
        s = str(v).lower()
        for i, nome in enumerate(("setosa", "versicolor", "virginica")):
            if nome in s:
                return i
        return int(float(s))
    df["species"] = bruto.iloc[:, -1].map(cod)
    return df


# ---------------------------------------------------------------- configuração
DATASETS = {
    "pima": {
        "id": "pima", "nome": "Diabetes (Pima Indians)", "entidade": "paciente",
        "csv": DADOS / "pima-indians-diabetes.csv", "ler": _ler_pima,
        "atributos": PIMA_COLUNAS[:-1], "alvo": "Outcome",
        "classes": {0: "Sem diabetes", 1: "Com diabetes"}, "positiva": 1, "classe_alerta": 1,
        "zeros_invalidos": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "class_weight": "balanced", "scoring": "roc_auc",
        "grade": {"svm__kernel": ["linear", "rbf"], "svm__C": [0.1, 1, 10, 100],
                  "svm__gamma": ["scale", 0.001, 0.01, 0.1]},
        "rotulos": {
            "Pregnancies": "Número de gestações", "Glucose": "Glicose (mg/dL)",
            "BloodPressure": "Pressão diastólica (mmHg)", "SkinThickness": "Espessura da dobra cutânea (mm)",
            "Insulin": "Insulina sérica de 2 h (µU/mL)", "BMI": "IMC (kg/m²)",
            "DiabetesPedigreeFunction": "Histórico familiar de diabetes (função)", "Age": "Idade (anos)",
            "Outcome": "Diagnóstico (0 = não, 1 = sim)"},
        "descricao": {
            "Pregnancies": "Quantas vezes a paciente engravidou.",
            "Glucose": "Glicose no plasma 2 h após teste oral de tolerância à glicose.",
            "BloodPressure": "Pressão arterial diastólica.",
            "SkinThickness": "Espessura da dobra cutânea do tríceps.",
            "Insulin": "Insulina sérica 2 h após o teste de glicose.", "BMI": "Índice de massa corporal.",
            "DiabetesPedigreeFunction": "Pontuação que resume o histórico de diabetes na família.",
            "Age": "Idade em anos.", "Outcome": "Alvo: 1 se teve diagnóstico de diabetes, 0 caso contrário."},
        "limites": {"Pregnancies": (0, 20, 1), "Glucose": (0, 300, 1), "BloodPressure": (0, 200, 1),
                    "SkinThickness": (0, 100, 1), "Insulin": (0, 900, 1), "BMI": (0.0, 70.0, 0.1),
                    "DiabetesPedigreeFunction": (0.0, 2.5, 0.001), "Age": (21, 100, 1)},
        "dica_form": "Glicose, pressão, dobra cutânea, insulina e IMC aceitam 0 para “não informado”, "
                     "do mesmo modo que o treinamento tratou os dados (imputação pela mediana).",
        "scatter": ("Glucose", "BMI"),
        "aviso": lambda n: (
            "**Aviso:** esta previsão é apenas educacional e **não é um diagnóstico**. O modelo foi treinado "
            f"com {n} mulheres de origem Pima, com 21 anos ou mais, e pode não valer para outras pessoas. "
            "Erra em uma parcela dos casos (veja a seção 3) e usa poucas variáveis. Diagnóstico e decisões "
            "clínicas exigem avaliação de um profissional de saúde."),
    },
    "iris": {
        "id": "iris", "nome": "Flores Iris (3 espécies)", "entidade": "flor",
        "csv": DADOS / "iris.csv", "ler": _ler_iris,
        "atributos": ["sepal_length", "sepal_width", "petal_length", "petal_width"], "alvo": "species",
        "classes": {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"},
        "positiva": None, "classe_alerta": None, "zeros_invalidos": [],
        "class_weight": None, "scoring": "f1_macro",
        "grade": {"svm__kernel": ["linear", "rbf"], "svm__C": [0.1, 1, 10, 100],
                  "svm__gamma": ["scale", 0.01, 0.1, 1]},
        "rotulos": {"sepal_length": "Comprimento da sépala (cm)", "sepal_width": "Largura da sépala (cm)",
                    "petal_length": "Comprimento da pétala (cm)", "petal_width": "Largura da pétala (cm)",
                    "species": "Espécie"},
        "descricao": {"sepal_length": "Comprimento da sépala, a folha que protege o botão da flor.",
                      "sepal_width": "Largura da sépala.", "petal_length": "Comprimento da pétala.",
                      "petal_width": "Largura da pétala.", "species": "Alvo: espécie da flor."},
        "limites": {"sepal_length": (3.0, 9.0, 0.1), "sepal_width": (1.0, 5.0, 0.1),
                    "petal_length": (0.5, 8.0, 0.1), "petal_width": (0.0, 3.5, 0.1)},
        "dica_form": "Os valores iniciais são a mediana do conjunto de dados. Medidas em centímetros.",
        "scatter": ("petal_length", "petal_width"),
        "aviso": lambda n: (
            f"**Aviso:** a previsão é educacional e pode errar. O modelo foi treinado com apenas {n} flores e "
            "só conhece 3 espécies: diante de outra flor, ele ainda assim escolherá uma das três. Versicolor e "
            "virginica se confundem entre si, sobretudo com medidas intermediárias (veja a seção 3). Medidas "
            "fora da faixa dos dados de treino geram previsões pouco confiáveis."),
    },
}


def carregar_dados(ds) -> pd.DataFrame:
    """Dados tratados: sem nulos e sem linhas duplicadas (duplicatas poderiam cair em treino e teste)."""
    return ds["ler"](ds["csv"]).dropna().drop_duplicates().reset_index(drop=True)


def relatorio_limpeza(ds) -> dict:
    bruto = ds["ler"](ds["csv"])
    sem_nulos = bruto.dropna()
    return {"n_original": int(len(bruto)), "n_nulos": int(len(bruto) - len(sem_nulos)),
            "n_duplicados": int(sem_nulos.duplicated().sum()),
            "n_final": int(len(sem_nulos.drop_duplicates()))}
