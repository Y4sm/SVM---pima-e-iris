"""Tratamento, validação k-fold aninhada e treino da SVM. Uso: python treino.py [pima|iris]  (sem argumento: todos)"""
import json
import os
import sys
import warnings

os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, label_binarize
from sklearn.svm import SVC

from datasets import DATASETS, SAIDA, carregar_dados, relatorio_limpeza, zeros_para_nan

K, K_INTERNO, SEMENTE = 10, 5, 42


def caminhos(ds_id):
    SAIDA.mkdir(exist_ok=True)
    return SAIDA / f"modelo_{ds_id}.joblib", SAIDA / f"metricas_{ds_id}.json"


def montar_pipeline(ds, probabilidade=False):
    # Tudo dentro do pipeline: mediana e padronização usam só os dados de treino de cada rodada.
    passos = []
    if ds["zeros_invalidos"]:
        passos += [("zeros", FunctionTransformer(zeros_para_nan, kw_args={
                        "atributos": ds["atributos"], "zeros": ds["zeros_invalidos"]})),
                   ("imputar", SimpleImputer(strategy="median"))]
    passos += [("escalar", StandardScaler()),
               ("svm", SVC(probability=probabilidade, class_weight=ds["class_weight"], random_state=SEMENTE))]
    return Pipeline(passos)


def buscar(ds, X, y, probabilidade=False):
    cv = StratifiedKFold(K_INTERNO, shuffle=True, random_state=SEMENTE)
    return GridSearchCV(montar_pipeline(ds, probabilidade), ds["grade"], scoring=ds["scoring"],
                        cv=cv, refit=True, n_jobs=-1).fit(X, y)


def limpar(params):
    return {k.replace("svm__", ""): v for k, v in params.items()}


def pontuacoes(modelo, X, binario):
    s = modelo.decision_function(X)                       # binário: 1 coluna; multiclasse: 1 por classe
    return np.column_stack([-s, s]) if binario else s


def treinar(ds_id):
    ds = DATASETS[ds_id]
    df = carregar_dados(ds)
    X, y = df[ds["atributos"]], df[ds["alvo"]]
    ids = list(ds["classes"]); binario = len(ids) == 2
    ref = ids.index(ds["positiva"]) if binario else None     # binário: métricas da classe positiva
    cab = (lambda v: float(v[ref])) if binario else (lambda v: float(np.mean(v)))   # senão: média macro
    gx = np.linspace(0, 1, 101)
    folds, cm = [], np.zeros((len(ids), len(ids)), dtype=int)
    tpr = [[] for _ in ids]; prc = [[] for _ in ids]
    pc = {c: {"precision": [], "recall": [], "f1": [], "auc_roc": []} for c in ids}

    # Validação cruzada ESTRATIFICADA ANINHADA: hiperparâmetros escolhidos só com os K-1 folds de treino.
    for i, (tr, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEMENTE).split(X, y), 1):
        m = buscar(ds, X.iloc[tr], y.iloc[tr])
        y_te = y.iloc[te].to_numpy(); pred = m.predict(X.iloc[te])
        S = pontuacoes(m, X.iloc[te], binario)
        Y = label_binarize(y_te, classes=ids)
        if binario:
            Y = np.column_stack([1 - Y[:, 0], Y[:, 0]])
        p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=ids, zero_division=0)
        aucs = []
        for j, c in enumerate(ids):
            fp, tp, _ = roc_curve(Y[:, j], S[:, j]); pcs, rcs, _ = precision_recall_curve(Y[:, j], S[:, j])
            tpr[j].append(np.interp(gx, fp, tp).tolist()); prc[j].append(np.interp(gx, rcs[::-1], pcs[::-1]).tolist())
            aucs.append(auc(fp, tp))
            for k, v in (("precision", p[j]), ("recall", r[j]), ("f1", f[j]), ("auc_roc", aucs[-1])):
                pc[c][k].append(float(v))
        cm += confusion_matrix(y_te, pred, labels=ids)
        folds.append({"fold": i, "n_teste": int(len(te)), "acuracia": float(accuracy_score(y_te, pred)),
                      "precision": cab(p), "recall": cab(r), "f1": cab(f), "auc_roc": cab(aucs),
                      "parametros": limpar(m.best_params_)})
        print(f"[{ds_id}] fold {i:>2}/{K}: acc={folds[-1]['acuracia']:.3f} f1={folds[-1]['f1']:.3f}")

    resumo = {n: {"media": float(np.mean([f[n] for f in folds])), "desvio": float(np.std([f[n] for f in folds], ddof=1))}
              for n in ("acuracia", "precision", "recall", "f1", "auc_roc")}
    final = buscar(ds, X, y, probabilidade=True)              # modelo do app: todos os dados
    metricas = {
        "k": K, "k_interno": K_INTERNO, "binario": binario, "n_total": int(len(df)), "limpeza": relatorio_limpeza(ds),
        "resumo": resumo,
        "por_classe": {str(c): {k: {"media": float(np.mean(v)), "desvio": float(np.std(v, ddof=1))}
                                for k, v in d.items()} for c, d in pc.items()},
        "folds": folds, "matriz_confusao": cm.tolist(), "grade_x": gx.tolist(), "tpr_folds": tpr, "precision_folds": prc,
        "prevalencia": float(np.mean(y == ds["positiva"])) if binario else None,
        "parametros_finais": limpar(final.best_params_),
    }
    arq_modelo, arq_met = caminhos(ds_id)
    joblib.dump(final.best_estimator_, arq_modelo)
    arq_met.write_text(json.dumps(metricas, ensure_ascii=False))
    return metricas


if __name__ == "__main__":
    for ds_id in (sys.argv[1:] or list(DATASETS)):
        m = treinar(ds_id)
        print(f"\n== {DATASETS[ds_id]['nome']} · {K} folds aninhados · limpeza {m['limpeza']}")
        for n, r in m["resumo"].items():
            print(f"{n:>10}: {r['media']:.3f} ± {r['desvio']:.3f}")
        print("Matriz de confusão:", m["matriz_confusao"], "| parâmetros finais:", m["parametros_finais"], "\n")
