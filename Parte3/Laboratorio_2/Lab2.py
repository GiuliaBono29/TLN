import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import numpy as np

SBERT_MODEL_INSTANCE = None

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Synset di riferimento per ogni parola target
# Sono i synset "corretti" con cui confrontare quelli trovati
TARGET_SYNSETS = {
    "tree":   wn.synset("tree.n.01"),
    "teapot": wn.synset("teapot.n.01"),
    "music":  wn.synset("music.n.04"),
    "ethic":  wn.synset("ethics.n.02"),
}

def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return lemmas

#la prima parola della definizione (con sinset) è considerata il "genere" della definizione, e viene usata per recuperare i synset del genere
def get_genus(lemmas: list):
    for lemma in lemmas:
        synsets = wn.synsets(lemma, pos=wn.NOUN)
        if synsets:
            return lemma, synsets
    return None, []

#per ogni significato del genus, recupero i synset candidati da WordNet
def get_candidate_synsets(genus_synsets: list) -> list:
    candidates = []   #lista sinset dei figli del genus (per tutti i sensi del genus)
    depth = 3   #distanza genitore figlio es pianta - albero
    for genus in genus_synsets:
        hypo = list(genus.closure(lambda s: s.hyponyms(), depth=depth))
        candidates.extend(hypo)  #recupero dei synset candidati da WordNet per ogni lemma
    return list(set(candidates))

#APPROCCIO 1 - Overlap
#metodo di ricerca del synset migliore per una definizione data, confrontando i candidati con il synset target
def find_best_synset_overlap(definition: str, target_word: str) -> dict:
    target_synset = TARGET_SYNSETS.get(target_word) #recupero il sinset corretto per la parola target
    #trovo le parole chiave della definizione
    lemmas = preprocess(definition) #preprocessing della definizione
    
    #la prima parola della definizione (con sinset) è considerata il "genere" della definizione, e viene usata per recuperare i synset del genere
    genus_word, genus_synsets = get_genus(lemmas)
    if genus_word is None:
        return {"error": "Nessun genere trovato nella definizione"}
    candidates = get_candidate_synsets(genus_synsets)  #recupero dei synset candidati da WordNet per ogni lemma
    if not candidates:
        return {"error": "Nessun synset candidato trovato"}
    
    best_synset  = None
    best_score   = -1
    best_details = {}

    # per ogni candidato, calcola lo score rispetto al synset target e aggiorna il migliore trovato
    for candidate in candidates:
        def_candidate = set(preprocess(candidate.definition()))  #solo definizioni, no esempi
        def_original = set(lemmas)  #definizione degli studenti
        overlap = len(def_original & def_candidate) / len(def_original | def_candidate) if (def_original | def_candidate) else 0.0
        if overlap > best_score:
            best_score   = overlap
            best_synset  = candidate


    return {
        "best_synset":    best_synset.name(),
        "gloss":          best_synset.definition(),
        "genus_word":     genus_word,
        "score":          best_score,
        "correct":        best_synset.name() == target_synset.name(),
    }

#APPROCCIO 2 - SBERT
#carico modello sbert    
def get_sbert_model():
    global SBERT_MODEL_INSTANCE
    if SBERT_MODEL_INSTANCE is None:
        print("Caricamento modello SBERT...")
        SBERT_MODEL_INSTANCE = SentenceTransformer("all-MiniLM-L6-v2")
    return SBERT_MODEL_INSTANCE

#metodo di ricerca del synset migliore per una definizione, confrontando l'embedding della definizione
#con l'embedding della gloss di ogni synset candidato
def find_best_synset_sbert(definition: str, target_word: str) -> dict:
    target_synset = TARGET_SYNSETS.get(target_word) #recupero il sinset corretto per la parola target
    #trovo le parole chiave della definizione
    lemmas = preprocess(definition) #preprocessing della definizione
    
    #la prima parola della definizione (con sinset) è considerata il "genere" della definizione, e viene usata per recuperare i synset del genere
    genus_word, genus_synsets = get_genus(lemmas)
    if genus_word is None:
        return {"error": "Nessun genere trovato nella definizione"}
    
    candidates = get_candidate_synsets(genus_synsets)  #recupero dei synset candidati da WordNet per ogni lemma
    if not candidates:
        return {"error": "Nessun synset candidato trovato"}
    
    model = get_sbert_model()  #carico modello sbert
    
    def_embedding = model.encode([definition], convert_to_numpy=True)   #calcolo embedding della definizione originale
    
    def_candidate   = [s.definition() for s in candidates]    #calcolo embedding di ogni definizone cadnidata
    def_candidate_emb = model.encode(def_candidate, convert_to_numpy=True)  #embedding delle definizioni candidate
    
    #calcolo cosine similarity tra definizione e ogni candidato
    #restituisce un array di valori di similarità e ogni valore corrisponde alla similiratità del candidato
    similarities = cos_sim(def_embedding, def_candidate_emb)[0] 

    #Synset con similarity più alta
    best_idx     = int(np.argmax(similarities))
    best_synset  = candidates[best_idx]
    best_score   = float(similarities[best_idx])

    return {
        "best_synset_sbert": best_synset.name(),
        "gloss_sbert":       best_synset.definition(),
        "genus_word":        genus_word,
        "score_sbert":       round(best_score, 4),
        "correct_sbert":     best_synset.name() == target_synset.name(),
    }

# APPROCCIO 3 — IBRIDO: WordNet + SBERT combinati
def normalizza(valori: list) -> list:
    min_v = min(valori)
    max_v = max(valori)
    if max_v == min_v:
        return [0.5] * len(valori)
    return [(v - min_v) / (max_v - min_v) for v in valori]

#metodo di ricerca del synset migliore per una definizione, combinando i punteggi di WordNet e SBERT con una media pesata
#Valore test1: peso_wn=0.5, peso_sbert=0.5 test2: peso_wn=0.6, peso_sbert=0.4 test3: peso_wn=0.4, peso_sbert=0.6
def find_best_synset_ibrido(definition: str, target_word: str, peso_wn: float = 0.4, peso_sbert: float = 0.6) -> dict:
    target_synset = TARGET_SYNSETS.get(target_word) #recupero il sinset corretto per la parola target
    lemmas = preprocess(definition)
    
    #la prima parola della definizione (con sinset) è considerata il "genere" della definizione, e viene usata per recuperare i synset del genere
    genus_word, genus_synsets = get_genus(lemmas)
    if genus_word is None:
        return {"error": "Nessun genere trovato nella definizione"}

    candidates = get_candidate_synsets(genus_synsets)  #synset della parola target
    if not candidates:
        return {"error": "Nessun synset trovato"}

    model = get_sbert_model()

    #score WordNet per ogni candidato
    scores_wn = []
    for candidate in candidates:
        candidate_lemmas = set(preprocess(candidate.definition()))
        def_set      = set(lemmas)
        overlap = len(def_set & candidate_lemmas) / len(def_set | candidate_lemmas) if (def_set | candidate_lemmas) else 0.0
        scores_wn.append(overlap)

    #score SBERT per ogni candidato
    def_emb   = model.encode([definition], convert_to_numpy=True)   #definizione originale
    
    def_candidate   = [s.definition() for s in candidates]
    def_candidate_emb = model.encode(def_candidate, convert_to_numpy=True)
    scores_sbert = cos_sim(def_emb, def_candidate_emb)[0].tolist()

    #normalizza entrambi gli score
    scores_wn_norm    = normalizza(scores_wn)
    scores_sbert_norm = normalizza(scores_sbert)

    #combina con media pesata
    scores_ibridi = [
        (scores_wn_norm[i] * peso_wn) + (scores_sbert_norm[i] * peso_sbert)
        for i in range(len(candidates))
    ]

    #synset con score ibrido più alto
    best_idx    = int(np.argmax(scores_ibridi))
    best_synset = candidates[best_idx]
    best_score  = scores_ibridi[best_idx]

    return {
        "best_synset_ibrido": best_synset.name(),
        "gloss_ibrido":       best_synset.definition(),
        "genus":              genus_word,
        "score_ibrido":       round(best_score, 4),
        "correct_ibrido":     best_synset.name() == target_synset.name(),
    }

#metodo di stampa dei risultati del confronto tra i tre approcci
def risultati_finali(df: pd.DataFrame, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for word in df["word"].unique():
            f.write(f"\n{'='*60}\n")
            f.write(f"* {word.upper()}\n")
            f.write(f"{'='*60}\n")

            sub = df[df["word"] == word].reset_index(drop=True)

            for i, row in sub.iterrows():
                lemmas     = preprocess(row["definition"])
                genus_word = next((l for l in lemmas if wn.synsets(l, pos=wn.NOUN)), "—")

                f.write(f"\n  - def {i+1}: {row['definition']}\n")
                f.write(f"    lemmi : {lemmas}\n")
                f.write(f"    genus : {genus_word}\n")
                f.write(f"    predizione_wn: {str(row.get('best_synset', '—')):<30} "
                        f"predizione_sbert: {str(row.get('best_synset_sbert', '—')):<30} "
                        f"predizione_ibrido: {str(row.get('best_synset_ibrido', '—'))}\n")
                f.write(f"    score_wn: {str(row.get('score', '—')):<10} "
                        f"score_sbert: {str(row.get('score_sbert', '—')):<10} "
                        f"score_ibrido: {str(row.get('score_ibrido', '—'))}\n")

    print(f"Risultati dettagliati salvati in: {output_path}")
    
    print(f"\n{'='*60}")
    print("RIEPILOGO FINALE")
    print(f"{'='*60}")
    print(f"\n{'':10} {'WordNet':>12} {'SBERT':>12} {'Ibrido':>12}")
    print("-" * 50)

    for word in df["word"].unique():
        sub     = df[df["word"] == word]
        n_total = len(sub)
        n_wn    = sub["correct"].sum()
        n_sbert = sub["correct_sbert"].sum()
        n_ibr   = sub["correct_ibrido"].sum()

        print(f"{word:<10} {f'{n_wn}/{n_total}':>12} {f'{n_sbert}/{n_total}':>12} {f'{n_ibr}/{n_total}':>12}")
                


def lab2(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.lower().str.strip()
    df = df[["word", "definition", "concreteness", "specificity"]].dropna()
    print(f"Caricate {len(df)} definizioni.\n")

    results = []
    for _, row in df.iterrows():
        res1 = find_best_synset_overlap(row["definition"], row["word"])
        res2 = find_best_synset_sbert(row["definition"], row["word"])
        res3 = find_best_synset_ibrido(row["definition"], row["word"])

        result = {
            "word":         row["word"],
            "concreteness": row["concreteness"],
            "specificity":  row["specificity"],
            "definition":   row["definition"],
        }
        result.update(res1)
        result.update(res2)
        result.update(res3)
        results.append(result)

    # converti in DataFrame prima di passarlo alle funzioni
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    risultati_finali(results_df, output_path.with_name("lab2_dettaglio.txt"))
    

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    
    CSV_PATH = BASE_DIR / "definitions.csv"
    OUTPUT_PATH = BASE_DIR / "lab2_risultati.csv"

    lab2(CSV_PATH, OUTPUT_PATH)