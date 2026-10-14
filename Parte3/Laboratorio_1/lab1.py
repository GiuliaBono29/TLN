from pathlib import Path

import pandas as pd
import numpy as np
import nltk
import itertools
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy import stats

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
SBERT_MODEL = "all-MiniLM-L6-v2"

#INIZIO funzioni confronto_lessicale_semantico
#caricare csv
#usato in confronto_lessicale_semantico per caricare il csv con le definizioni
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.lower().str.strip()

    required = {"word", "definition", "concreteness", "specificity"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Il CSV deve avere le colonne: {required}. "
            f"Trovate: {set(df.columns)}"
        )

    df = df[["word", "definition", "concreteness", "specificity"]].dropna().reset_index(drop=True)
    df["def_id"] = [f"{row.word}_{i}" for i, row in df.iterrows()]

    print(f"Caricamento definizioni")
    return df

#preparazione dati e tokenizzazione
#usato in build_pairwise_df per calcolare sim_lex tra coppie di definizioni
def preprocess(text: str, remove_stopwords: bool = True) -> set:
    text = text.lower() #conversione minuscolo
    text = re.sub(r"[^a-z\s]", "", text)    #rimozione punteggiatura e numeri
    tokens = nltk.word_tokenize(text)   #tokenizzazione

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    lemmas = {lemmatizer.lemmatize(t) for t in tokens if len(t) > 1}
    return lemmas

#calcolo similarità lessicale -> quanto due set di parole si assomigliano? SIMLEX
#usato in build_pairwise_df per calcolare sim_lex tra coppie di definizioni
def jaccard(set_a: set, set_b: set) -> float:
    intersection = set_a & set_b    #intersezione {radici, tronco, fogliame, carta}
                                    #             {radici, carta, frutta, fiori}
                                    #           = {radici, carta}
    union = set_a | set_b           #unione       {radici, tronco, fogliame, carta}
                                    #             {radici, carta, frutta, fiori}
                                    #           = {radici, tronco, fogliame, carta, frutta, fiori}
    if not union:                   #caso insiemi vuoti, evita divisione per zero
        return 0.0
    return len(intersection) / len(union)   #divisione per ottenere un avlore tra 0 e 1 (0 diversi, 1 uguale)

#creazione di vettori di embedding con il modello SBERT -> ogni frase rappresenta un punto nello spazio semantico, 
#più sono vicini i punti più sono simili semanticamente SIMSEM
#usato in confronto_lessicale_semantico per calcolare sim_sem tra coppie di definizioni
def compute_embeddings(texts: list, model_name: str = SBERT_MODEL) -> np.ndarray:
    print(f"Caricamento modello SBERT")
    model = SentenceTransformer(model_name)
    print(f"Calcolo embeddings")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embeddings calcolati")
    return embeddings

#crea tutte le coppie possibili di definizioni e calcola sim_lex e sim_sem per ciascuna coppia
#usato in confronto_lessicale_semantico per costruire un dataframe con tutte le coppie di definizioni e i valori di similarità
def build_pairwise_df(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    n = len(df)
    pairs = list(itertools.combinations(range(n), 2))   #crea tutte le combinazioni di coppie senza ripetizioni

    print(f"Calcolo coppie")
    lemma_sets = [preprocess(text) for text in df["definition"]] #pre-processa tutte le definizioni
    cos_matrix = cosine_similarity(embeddings)  #matrice di cosine similarity per tutte le coppie di definizioni

    rows = []
    for i, j in tqdm(pairs, desc="Coppie"):
        conc = sorted([df.loc[i, "concreteness"], df.loc[j, "concreteness"]])
        spec = sorted([df.loc[i, "specificity"],  df.loc[j, "specificity"]])
        row = {
            #Info parola a (parola, id e definizione)
            "word_A":    df.loc[i, "word"],
            "def_id_A":  df.loc[i, "def_id"],
            "def_A":     df.loc[i, "definition"],
            #Info parola b (parola, id e definizione)
            "word_B":    df.loc[j, "word"],
            "def_id_B":  df.loc[j, "def_id"],
            "def_B":     df.loc[j, "definition"],
            # simsem e simlex
            "sim_lex":   round(jaccard(lemma_sets[i], lemma_sets[j]), 4),
            "sim_sem":   round(float(cos_matrix[i, j]), 4),
            # tipo di coppia (intra o inter) e gruppo di concretezza e specificità
            "pair_type":             "intra" if df.loc[i, "word"] == df.loc[j, "word"] else "inter",    #mette intra se stessa parola, inter se parole diverse
            "gruppo_concreteness":   "-".join(conc),    #mette concrete-concrete, abstract-concrete, abstract-abstract per concreteness
            "gruppo_specificity":    "-".join(spec),    #e generic-generic, generic-specific, specific-specific per specificity
        }
        rows.append(row)

    result_df = pd.DataFrame(rows)
    print(f"DataFrame coppie costruito")
    return result_df
#FINE

#chiamato nel main
def confronto_lessicale_semantico(csv_path: str, output_path: str = "cls_risultati.csv"):
    df = load_csv(csv_path) # Carica dati
    embeddings = compute_embeddings(df["definition"].tolist())  # Calcola embeddings
    pairs_df = build_pairwise_df(df, embeddings)    # Costruisce tutte le coppie con i due valori di similarità
    pairs_df.to_csv(output_path, index=False)   #salva su csv
    print(f"Risultati salvati")
    print("\n── Medie per coppia di parole ──")
    summary = (
        pairs_df.groupby(["word_A", "word_B"])[["sim_lex", "sim_sem"]]
        .mean()
        .round(4)
    )
    print(summary.to_string() + "\n")

    return pairs_df

#Stampa le statistiche descrittive per sim_lex e sim_sem
#chiamato nel main
def statistiche_descrittive(df: pd.DataFrame):
    #BLOCCO 1: statistiche globali con count, mean, std, min, 25%, 50%, 75%, max
    print("STATISTICHE DESCRITTIVE GLOBALI")
    print(df[["sim_lex", "sim_sem"]].describe().round(4).to_string() + "\n")

    #BLOCCO 2: statistiche intra-word per parola (mean, median, std, min, max)
    print("STATISTICHE INTRA-WORD (per parola)")
    intra = df[df["pair_type"] == "intra"]
    for metric in ["sim_lex", "sim_sem"]:
        print(f"\n  {metric.upper()}")
        summary = intra.groupby("word_A")[metric].agg(
            ["mean", "median", "std", "min", "max"]
        ).round(4)
        print(summary.to_string())

    #BLOCCO 3: confronto tra intra e inter (mean, median, std) -> serve per capire se le
    #definizioni di una stessa parola sono più simili tra loro rispetto a definizioni di parole diverse
    print("\nCONFRONTO INTRA vs INTER")
    comparison = df.groupby("pair_type")[["sim_lex", "sim_sem"]].agg(
        ["mean", "median", "std"]
    ).round(4)
    print(comparison.to_string() + "\n")

#Produzione istogramma simlex/simsem e boxplot simlex/simsem
def crea_grafici(df: pd.DataFrame, output_path: str):
    intra = df[df["pair_type"] == "intra"]
    inter = df[df["pair_type"] == "inter"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Fase 2 — Analisi Quantitativa", fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    #GRAFICO 1: Istogramma simlex per intra vs inter (inta= stessa definizione, inter=definizioni diverse)
    #vorrei ottenere basso per inter e alto per intra -> definizioni di parole diverse usano parole generalmente diverse
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(intra["sim_lex"], bins=20, alpha=0.7, color="#4C72B0", label="intra-word")
    ax1.hist(inter["sim_lex"], bins=20, alpha=0.7, color="#DD8452", label="inter-word")
    ax1.set_title("Distribuzione sim_lex")
    ax1.set_xlabel("Jaccard (similarità lessicale)")
    ax1.set_ylabel("Frequenza")
    ax1.legend()
    ax1.axvline(intra["sim_lex"].mean(), color="#4C72B0", linestyle="--", linewidth=1.2,
                label=f"media intra={intra['sim_lex'].mean():.3f}")
    ax1.axvline(inter["sim_lex"].mean(), color="#DD8452", linestyle="--", linewidth=1.2,
                label=f"media inter={inter['sim_lex'].mean():.3f}")

    #GRAFICO 2: Istogramma sim_sem per intra vs inter (inta= stessa definizione, inter=definizioni diverse)
    #vorrei ottenere basso per inter e alto per intra -> definizioni di parole diverse sono generalmente distanti
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(intra["sim_sem"], bins=20, alpha=0.7, color="#4C72B0", label="intra-word")
    ax2.hist(inter["sim_sem"], bins=20, alpha=0.7, color="#DD8452", label="inter-word")
    ax2.set_title("Distribuzione sim_sem")
    ax2.set_xlabel("Cosine similarity (similarità semantica)")
    ax2.set_ylabel("Frequenza")
    ax2.legend()
    ax2.axvline(intra["sim_sem"].mean(), color="#4C72B0", linestyle="--", linewidth=1.2)
    ax2.axvline(inter["sim_sem"].mean(), color="#DD8452", linestyle="--", linewidth=1.2)

    #GRAFICO 3: Boxplot sim_lex per parola -> faccio vedere quante parole comuni ci sono tra le varie definizioni di una stessa parola
    #parole concrete hanno più parole in comune tra le definizioni, parole astratte meno
    ax3 = fig.add_subplot(gs[1, 0])
    words = intra["word_A"].unique()
    data_lex = [intra[intra["word_A"] == w]["sim_lex"].values for w in words]
    bp1 = ax3.boxplot(data_lex, labels=words, patch_artist=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title("sim_lex per parola (intra-word)")
    ax3.set_xlabel("Parola")
    ax3.set_ylabel("Jaccard")

    #GRAFICO 4: Boxplot sim_sem per parola —> faccio vedere quanto sono vicine semanticamente le definizioni di una stessa parola
    ax4 = fig.add_subplot(gs[1, 1])
    data_sem = [intra[intra["word_A"] == w]["sim_sem"].values for w in words]
    bp2 = ax4.boxplot(data_sem, labels=words, patch_artist=True)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title("sim_sem per parola (intra-word)")
    ax4.set_xlabel("Parola")
    ax4.set_ylabel("Cosine similarity")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Grafici salvati")
    plt.show()

def aggregazione_gruppi(df: pd.DataFrame):
    intra = df[df["pair_type"] == "intra"]

    # Media sim_lex e sim_sem per ogni singola parola (intra)
    per_parola = (intra.groupby("word_A")[["sim_lex", "sim_sem"]].mean().round(4))

    meta = intra[["word_A", "gruppo_concreteness", "gruppo_specificity"]].copy()
    meta["concreteness"] = meta["gruppo_concreteness"].apply(lambda x: x.split("-")[0])
    meta["specificity"]  = meta["gruppo_specificity"].apply(lambda x: x.split("-")[0])
    meta = meta.drop_duplicates(subset="word_A").set_index("word_A")
    per_parola = per_parola.join(meta[["concreteness", "specificity"]])

    agg_conc = per_parola.groupby("concreteness")[["sim_lex", "sim_sem"]].mean().round(4)
    agg_spec = per_parola.groupby("specificity")[["sim_lex", "sim_sem"]].mean().round(4)

    return per_parola, agg_conc, agg_spec


def crea_grafici_aggregati(df: pd.DataFrame, output_path: str):

    intra = df[df["pair_type"] == "intra"]

    # Calcola valori per parola
    per_parola = intra.groupby("word_A")[["sim_lex", "sim_sem"]].mean().round(4)
    meta = intra[["word_A", "gruppo_concreteness", "gruppo_specificity"]].copy()
    meta["concreteness"] = meta["gruppo_concreteness"].apply(lambda x: x.split("-")[0])
    meta["specificity"]  = meta["gruppo_specificity"].apply(lambda x: x.split("-")[0])
    meta = meta.drop_duplicates(subset="word_A").set_index("word_A")
    per_parola = per_parola.join(meta[["concreteness", "specificity"]])

    agg_conc = per_parola.groupby("concreteness")[["sim_lex", "sim_sem"]].mean().round(4)
    agg_spec = per_parola.groupby("specificity")[["sim_lex", "sim_sem"]].mean().round(4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Fase 3 — Aggregazione per Concretezza e Specificità",
                 fontsize=15, fontweight="bold")

    colors_conc = {"abstract": "#C44E52", "concrete": "#4C72B0"}
    colors_spec = {"generic": "#55A868",  "specific": "#DD8452"}

    # GRAFICO 1 — sim_lex per concretezza
    ax = axes[0, 0]
    bars = ax.bar(agg_conc.index, agg_conc["sim_lex"],
                  color=[colors_conc[g] for g in agg_conc.index], alpha=0.8)
    ax.set_title("sim_lex per concretezza")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0, agg_conc["sim_lex"].max() * 1.3)
    for bar, val in zip(bars, agg_conc["sim_lex"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    # GRAFICO 2 — sim_sem per concretezza
    ax = axes[0, 1]
    bars = ax.bar(agg_conc.index, agg_conc["sim_sem"],
                  color=[colors_conc[g] for g in agg_conc.index], alpha=0.8)
    ax.set_title("sim_sem per concretezza")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0, agg_conc["sim_sem"].max() * 1.3)
    for bar, val in zip(bars, agg_conc["sim_sem"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    # GRAFICO 3 — sim_lex per specificità
    ax = axes[1, 0]
    bars = ax.bar(agg_spec.index, agg_spec["sim_lex"],
                  color=[colors_spec[g] for g in agg_spec.index], alpha=0.8)
    ax.set_title("sim_lex per specificità")
    ax.set_ylabel("Jaccard")
    ax.set_ylim(0, agg_spec["sim_lex"].max() * 1.3)
    for bar, val in zip(bars, agg_spec["sim_lex"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    # GRAFICO 4 — sim_sem per specificità
    ax = axes[1, 1]
    bars = ax.bar(agg_spec.index, agg_spec["sim_sem"],
                  color=[colors_spec[g] for g in agg_spec.index], alpha=0.8)
    ax.set_title("sim_sem per specificità")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0, agg_spec["sim_sem"].max() * 1.3)
    for bar, val in zip(bars, agg_spec["sim_sem"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[✓] Grafici Fase 3 salvati in: {output_path}")
    plt.show()

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    CSV_DIR     = BASE_DIR / "csv"
    GRAFICI_DIR = BASE_DIR / "grafici"

    CSV_PATH       = CSV_DIR / "definitions.csv"
    OUTPUT_CSV     = CSV_DIR / "cls_risultati.csv"
    OUTPUT_PLOT    = GRAFICI_DIR / "grafici.png"
    OUTPUT_PLOT_F3 = GRAFICI_DIR / "grafici_aggregati.png"

    # Fase 1
    pairs_df = confronto_lessicale_semantico(CSV_PATH, OUTPUT_CSV)
    
    # Fase 2
    statistiche_descrittive(pairs_df)
    crea_grafici(pairs_df, OUTPUT_PLOT)
    
    # Fase 3
    aggregazione_gruppi(pairs_df)
    crea_grafici_aggregati(pairs_df, OUTPUT_PLOT_F3)
