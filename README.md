# SVM · Diabetes (Pima, binário) e Iris (3 classes)

    pip install -r requirements.txt
    python treino.py            # treina os dois (cerca de 2 min); ou: python treino.py pima / python treino.py iris
    streamlit run app.py        # escolha o conjunto de dados na barra lateral

- `dados/`: `pima-indians-diabetes.csv` e `iris.csv` (substitua mantendo o nome para usar os seus).
- `datasets.py`: configuração de cada conjunto (colunas, classes, rótulos, limites do formulário) e limpeza.
- `treino.py`: validação cruzada estratificada de 10 folds, aninhada (kernel, C e gamma); modelo final com todos os dados.
- `app.py`: análise, resultado, desempenho e exploração dos dados, para os dois conjuntos.
- `modelos/`: modelos treinados e métricas.
