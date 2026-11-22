"""App Streamlit com SVM: Diabetes (Pima, binário) e Iris (3 classes). Uso: streamlit run app.py"""
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from datasets import DATASETS, carregar_dados

st.set_page_config(page_title="SVM · Classificação", layout="wide")

PAGINAS = ["1. Análise", "2. Resultado da classificação", "3. Desempenho do modelo", "4. Exploração dos dados"]
PALETA = ["#1d4ed8", "#d9480f", "#15803d"]

# ---------------------------------------------------------------- seleção do conjunto
st.sidebar.title("SVM")
ds_id = st.sidebar.selectbox("Conjunto de dados", list(DATASETS), format_func=lambda k: DATASETS[k]["nome"], key="ds")
ds = DATASETS[ds_id]
ATR, ALVO, CLASSES = ds["atributos"], ds["alvo"], ds["classes"]
IDS = list(CLASSES); NOMES = [CLASSES[i] for i in IDS]; CORES = dict(zip(IDS, PALETA))
ROT, DESC = ds["rotulos"], ds["descricao"]
ent = ds["entidade"]


@st.cache_resource
def carregar_modelo(k):
    import treino
    arq_modelo, arq_met = treino.caminhos(k)
    if not arq_modelo.exists() or not arq_met.exists():
        treino.treinar(k)
    return joblib.load(arq_modelo), json.loads(arq_met.read_text())


@st.cache_data
def dados(k):
    return carregar_dados(DATASETS[k])


modelo, met = carregar_modelo(ds_id)
df = dados(ds_id)
binario = met["binario"]
curto = [n.replace("Iris ", "") for n in NOMES]
AVISO = ds["aviso"](met["n_total"])

if "pagina" not in st.session_state:
    st.session_state["pagina"] = PAGINAS[0]
st.sidebar.radio("Seção", PAGINAS, key="pagina")
st.sidebar.caption("Projeto educacional de classificação com SVM.")
st.title(f"{ds['nome']}")


def enviar():
    st.session_state[f"amostra_{ds_id}"] = {c: st.session_state[f"in_{ds_id}_{c}"] for c in ATR}
    st.session_state["pagina"] = PAGINAS[1]


def valor_inicial(c):
    mn, mx, passo = ds["limites"][c]
    s = df[c].replace(0, np.nan) if c in ds["zeros_invalidos"] else df[c]
    v = float(s.median())
    return int(round(v)) if isinstance(passo, int) else round(v, len(str(passo).split(".")[1]))


pagina = st.session_state["pagina"]

# ---------------------------------------------------------------- 1. Análise
if pagina == PAGINAS[0]:
    st.header(f"Análise de {ent}")
    st.write(f"Informe os dados e solicite a classificação pela SVM.")
    with st.form(f"form_{ds_id}"):
        cols = st.columns(2)
        for i, c in enumerate(ATR):
            mn, mx, passo = ds["limites"][c]
            ajuda = ("0 = não medido; será tratado como ausente e imputado." if c in ds["zeros_invalidos"] else DESC[c])
            cols[i % 2].number_input(ROT[c], min_value=mn, max_value=mx, step=passo, value=valor_inicial(c),
                                     key=f"in_{ds_id}_{c}", help=ajuda)
        st.form_submit_button("Classificar", type="primary", on_click=enviar)
    st.info(ds["dica_form"])

# ---------------------------------------------------------------- 2. Resultado
elif pagina == PAGINAS[1]:
    st.header("Resultado da classificação")
    am = st.session_state.get(f"amostra_{ds_id}")
    if am is None:
        st.warning("Nenhuma análise ainda. Preencha o formulário na seção 1.")
    else:
        X = pd.DataFrame([am], columns=ATR)
        classe = int(modelo.predict(X)[0])
        proba = modelo.predict_proba(X)[0]
        msg = f"Classe estimada: **{CLASSES[classe]}**"
        (st.error if classe == ds["classe_alerta"] else st.success)(msg)
        st.metric("Probabilidade estimada da classe", f"{proba[IDS.index(classe)]:.0%}")
        st.subheader("Probabilidade por classe")
        st.bar_chart(pd.DataFrame({"Probabilidade": proba}, index=NOMES))
        st.caption("Probabilidades por calibração sigmoide (Platt), aproximadas. A classe vem do lado do "
                   "hiperplano (ou da votação entre pares de classes) e pode diferir da maior probabilidade "
                   "em casos próximos da fronteira.")
        st.subheader("Dados informados")
        st.dataframe(pd.DataFrame({"Variável": [ROT[c] for c in ATR], "Valor": [am[c] for c in ATR]}), hide_index=True)
        ausentes = [ROT[c] for c in ds["zeros_invalidos"] if am[c] == 0]
        if ausentes:
            st.info("Valores 0 tratados como ausentes e imputados pela mediana do treino: " + ", ".join(ausentes) + ".")
        fora = [ROT[c] for c in ATR if c not in ds["zeros_invalidos"] and not df[c].min() <= am[c] <= df[c].max()]
        if fora:
            st.error("Valores fora da faixa vista no treino (extrapolação): " + ", ".join(fora) + ".")
        st.warning(AVISO)
        st.button("Nova análise", on_click=lambda: st.session_state.update(pagina=PAGINAS[0]))

# ---------------------------------------------------------------- 3. Desempenho
elif pagina == PAGINAS[2]:
    st.header("Desempenho do modelo")
    k, p = met["k"], met["parametros_finais"]
    tipo = (f"Precision, recall e F1 são da classe “{CLASSES[ds['positiva']]}”." if binario
            else "Precision, recall, F1 e AUC são **médias entre as classes** (macro).")
    st.write(
        f"Validação cruzada **estratificada com {k} folds**, aninhada: em cada rodada, a SVM é treinada com {k - 1} "
        f"folds (escolhendo kernel, C e gamma por validação interna de {met['k_interno']} folds) e avaliada no fold "
        f"restante. Cada um dos {met['n_total']} registros é testado uma vez. {tipo} Valores: média ± desvio entre "
        f"folds. O modelo do app é treinado com todos os dados (kernel **{p['kernel']}**, C = **{p['C']}**, "
        f"gamma = **{p['gamma']}**).")
    suf = "" if binario else " (macro)"
    nomes = [("Acurácia", "acuracia"), (f"Precision{suf}", "precision"), (f"Recall{suf}", "recall"),
             (f"F1-score{suf}", "f1"), (f"AUC-ROC{suf}", "auc_roc")]
    for col, (nome, key) in zip(st.columns(5), nomes):
        r = met["resumo"][key]
        col.metric(nome, f"{r['media']:.3f}", f"± {r['desvio']:.3f}", delta_color="off")
    if binario:
        st.caption("Precision: dos classificados como positivos, quantos realmente são. "
                   "Recall: dos positivos reais, quantos o modelo encontrou.")
    g1, g2, g3 = st.columns(3)
    cm = np.array(met["matriz_confusao"]); n = len(IDS)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(n), curto, fontsize=8); ax.set_yticks(range(n), curto, fontsize=8)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real"); ax.set_title("Matriz de confusão (soma dos folds)", fontsize=10)
    g1.pyplot(fig)
    x = np.array(met["grade_x"])
    mostrar = [IDS.index(ds["positiva"])] if binario else range(n)     # binário: só a classe positiva
    for g, chave, titulo, xl, yl in [(g2, "tpr_folds", "Curva ROC (média dos folds)", "Taxa de falsos positivos",
                                      "Taxa de verdadeiros positivos"),
                                     (g3, "precision_folds", "Curva Precision × Recall (média dos folds)", "Recall", "Precision")]:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        for j in mostrar:
            m = np.array(met[chave][j]); c = IDS[j]
            if binario:
                for t in m:
                    ax.plot(x, t, color="gray", alpha=0.25, lw=1)
            lab = f"AUC = {met['resumo']['auc_roc']['media']:.3f}" if binario and chave == "tpr_folds" else curto[j]
            ax.plot(x, m.mean(0), lw=2, color=CORES[c] if not binario else "#d9480f", label=lab)
            ax.fill_between(x, np.clip(m.mean(0) - m.std(0), 0, 1), np.clip(m.mean(0) + m.std(0), 0, 1),
                            color=CORES[c] if not binario else "#d9480f", alpha=0.12)
        if chave == "tpr_folds":
            ax.plot([0, 1], [0, 1], "--", color="gray")
        elif binario:
            ax.axhline(met["prevalencia"], ls="--", color="gray", label="Linha de base")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(titulo, fontsize=10)
        ax.legend(loc="lower right" if chave == "tpr_folds" else "upper right", fontsize=8)
        g.pyplot(fig)
    st.subheader("Desempenho por classe")
    st.dataframe(pd.DataFrame([{"Classe": CLASSES[c], **{nm: f"{met['por_classe'][str(c)][k2]['media']:.3f} ± "
                                f"{met['por_classe'][str(c)][k2]['desvio']:.3f}" for nm, k2 in
                                [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("AUC-ROC", "auc_roc")]}}
                               for c in IDS]), hide_index=True)
    st.subheader("Resultado por fold")
    tab = pd.DataFrame(met["folds"])
    tab["parametros"] = tab["parametros"].apply(lambda d: f"{d['kernel']}, C={d['C']}, gamma={d['gamma']}")
    tab = tab.rename(columns={"fold": "Fold", "n_teste": "Testados", "acuracia": "Acurácia", "precision": "Precision",
                              "recall": "Recall", "f1": "F1", "auc_roc": "AUC-ROC", "parametros": "Hiperparâmetros escolhidos"})
    st.dataframe(tab.round(3), hide_index=True)
    st.warning(AVISO)

# ---------------------------------------------------------------- 4. Exploração
else:
    st.header("Exploração dos dados")
    lim = met["limpeza"]
    a, b, c, d = st.columns(4)
    a.metric("Registros originais", lim["n_original"]); b.metric("Duplicados removidos", lim["n_duplicados"])
    c.metric("Registros usados", lim["n_final"]); d.metric("Classes", len(IDS))
    t = df.copy()
    if ds["zeros_invalidos"]:
        t[ds["zeros_invalidos"]] = t[ds["zeros_invalidos"]].replace(0, np.nan)
        st.caption(f"Em {', '.join(ds['zeros_invalidos'])}, o zero é impossível e foi tratado como ausente; no pipeline é "
                   "imputado pela mediana do treino.")
    st.caption(f"Tratamento: {lim['n_nulos']} nulos; {lim['n_duplicados']} duplicada(s) removida(s), pois duplicatas "
               "poderiam cair em treino e teste. Depois, padronização (média 0, desvio 1) dentro do pipeline, "
               "pois a SVM depende de distâncias.")
    st.subheader("Variáveis")
    info = pd.DataFrame({"Variável": [ROT[c] for c in ATR], "Descrição": [DESC[c] for c in ATR]})
    if ds["zeros_invalidos"]:
        info["Zeros (ausentes)"] = [str(int((df[c] == 0).sum())) if c in ds["zeros_invalidos"] else "—" for c in ATR]
    st.dataframe(info, hide_index=True)
    st.subheader("Estatísticas descritivas" + (" (zeros inválidos removidos)" if ds["zeros_invalidos"] else ""))
    st.dataframe(t[ATR].describe().T.round(2))
    st.subheader("Distribuição por classe")
    nl = int(np.ceil(len(ATR) / 4))
    fig, axes = plt.subplots(nl, 4, figsize=(14, 3.2 * nl), squeeze=False)
    for ax, col in zip(axes.ravel(), ATR):
        for cl in IDS:
            ax.hist(t.loc[t[ALVO] == cl, col].dropna(), bins=15, alpha=0.6, color=CORES[cl], label=CLASSES[cl])
        ax.set_title(ROT[col], fontsize=9)
    for ax in axes.ravel()[len(ATR):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7); fig.tight_layout(); st.pyplot(fig)
    g1, g2 = st.columns(2)
    with g1:
        xa, ya = ds["scatter"]
        st.subheader(f"{ROT[xa].split(' (')[0]} × {ROT[ya].split(' (')[0]}")
        fig, ax = plt.subplots(figsize=(5, 4))
        for cl in IDS:
            s = t[t[ALVO] == cl]
            ax.scatter(s[xa], s[ya], color=CORES[cl], label=CLASSES[cl], alpha=0.7, s=18)
        ax.set_xlabel(ROT[xa]); ax.set_ylabel(ROT[ya]); ax.legend(fontsize=8)
        st.pyplot(fig)
    with g2:
        st.subheader("Correlação")
        corr = t.corr(); nc = len(corr)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(nc), corr.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(nc), corr.columns, fontsize=7)
        for i in range(nc):
            for j in range(nc):
                ax.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout(); st.pyplot(fig)
    cont = df[ALVO].value_counts().sort_index()
    st.caption("Balanceamento das classes: " + "; ".join(f"{CLASSES[i]} = {n}" for i, n in cont.items()) +
               (". As classes são desbalanceadas; por isso o treino usa class_weight='balanced'." if ds["class_weight"] else "."))
