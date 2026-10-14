import random

#  Domande e relazioni ConceptNet


QUESTIONS = [
    # ── UsedFor ──────────────────────────────────────────────
    {"id": 1, "text": "What is it used for?  UsedFor",                                    "relation": "UsedFor"},
    {"id": 2, "text": "What action does it help you perform?  UsedFor",                   "relation": "UsedFor"},
    {"id": 3, "text": "What problem does it solve?  UsedFor",                             "relation": "UsedFor"},
    {"id": 32, "text": "In what kind of activity would you use it?  UsedFor",             "relation": "UsedFor"},
    {"id": 33, "text": "What goal do people achieve by using it?  UsedFor",               "relation": "UsedFor"},

    # ── AtLocation ───────────────────────────────────────────
    {"id": 4, "text": "Where is it typically found?  AtLocation",                                       "relation": "AtLocation"},
    {"id": 34, "text": "In what kind of building or place would you expect to find it?  AtLocation",    "relation": "AtLocation"},
    {"id": 35, "text": "What environment is it naturally associated with?  AtLocation",                 "relation": "AtLocation"},

    # ── MadeOf ───────────────────────────────────────────────
    {"id": 7, "text": "What material is it made of?  MadeOf",                           "relation": "MadeOf"},
    {"id": 36, "text": "What resource was used to produce it?  MadeOf",                 "relation": "MadeOf"},

    # ── IsA, MemberOf InstanceOf ──────────────────────────────────────────────────
    {"id": 10, "text": "What category does it belong to?  IsA",                      "relation": "MemberOf"},
    {"id": 12, "text": "Is it a tool, a food, an animal, or something else?  IsA",   "relation": "IsA"},
    {"id": 38, "text": "How would you classify it in a encyclopedia?  IsA",          "relation": "IsA"},
    {"id": 39, "text": "What broader concept does it fall under?  IsA",              "relation": "InstanceOf"},

    # ── HasA ─────────────────────────────────────────────────
    {"id": 15, "text": "What are its main components?  HasA",                     "relation": "HasA"},
    {"id": 40, "text": "What physical parts can you identify on it?  HasA",       "relation": "HasA"},
    {"id": 41, "text": "What feature or element does it always include?  HasA",   "relation": "HasA"},

    # ── PartOf ───────────────────────────────────────────────
    {"id": 16, "text": "Is it part of something larger?  PartOf",                       "relation": "PartOf"},
    {"id": 42, "text": "What bigger object or system does it belong to?  PartOf",       "relation": "PartOf"},
    {"id": 43, "text": "Without what larger thing would it lose its purpose?  PartOf",  "relation": "PartOf"},

    # ── CapableOf ────────────────────────────────────────────
    {"id": 17, "text": "What can it do on its own?  CapableOf",                                    "relation": "CapableOf"},
    {"id": 44, "text": "What function can it perform without human intervention?  CapableOf",      "relation": "CapableOf"},
    {"id": 45, "text": "What physical or mechanical action is it capable of?  CapableOf",          "relation": "CapableOf"},

    # ── HasProperty ──────────────────────────────────────────
    {"id": 9, "text": "Is it hard or soft?  HasProperty",                                                "relation": "HasProperty"},
    {"id": 19, "text": "What color is it typically?  HasProperty",                                       "relation": "HasProperty"},
    {"id": 20, "text": "What size is it roughly? (small/medium/large)  HasProperty",                     "relation": "HasProperty"},
    {"id": 21, "text": "Is it heavy or light?  HasProperty",                                             "relation": "HasProperty"},
    {"id": 46, "text": "What texture does it have? (smooth, rough, sticky...)  HasProperty",             "relation": "HasProperty"},
    {"id": 47, "text": "What shape does it typically have?  HasProperty",                                "relation": "HasProperty"},

    # ── ReceivesAction ───────────────────────────────────────
    {"id": 24, "text": "What do you do with it? (hold, wear, eat, drive...)  ReceivesAction",       "relation": "ReceivesAction"},
    {"id": 48, "text": "What verb best describes how you interact with it?  ReceivesAction",        "relation": "ReceivesAction"},
    {"id": 49, "text": "What action do people perform on it to make it work?  ReceivesAction",      "relation": "ReceivesAction"},

    # ── CreatedBy ────────────────────────────────────────────
    {"id": 25, "text": "Who or what conceived it?  CreatedBy",                         "relation": "CreatedBy"},
    {"id": 50, "text": "What kind of worker or machine produces it?  CreatedBy",       "relation": "CreatedBy"},
    {"id": 51, "text": "Where is it typically manufactured or crafted?  CreatedBy",    "relation": "CreatedBy"},

    # ── SimilarTo ────────────────────────────────────────────
    {"id": 26, "text": "What is something similar to it?  SimilarTo",                          "relation": "SimilarTo"},
    {"id": 52, "text": "What other object shares its purpose or shape?  SimilarTo",            "relation": "SimilarTo"},
    {"id": 53, "text": "What would you use as a replacement if it were missing?  SimilarTo",   "relation": "SimilarTo"},

    # ── Antonym ──────────────────────────────────────────────
    {"id": 27, "text": "What is the opposite of it?  Antonym",                           "relation": "Antonym"},
    {"id": 54, "text": "What concept contrasts most strongly with it?  Antonym",         "relation": "Antonym"},
    {"id": 55, "text": "What word means the reverse of what it represents?  Antonym",    "relation": "Antonym"},

    # ── Causes ───────────────────────────────────────────────
    {"id": 28, "text": "What happens when you use it?  Causes",                                 "relation": "Causes"},
    {"id": 22, "text": "What effect does it produce?  Causes",                                  "relation": "Causes"},
    {"id": 56, "text": "What change does it bring about in the world around it?  Causes",       "relation": "Causes"},
    {"id": 57, "text": "What consequence follows from using it?  Causes",                       "relation": "Causes"},

    # ── HasPrerequisite ──────────────────────────────────────
    {"id": 29, "text": "Does using it require a specific skill?  HasPrerequisite",                   "relation": "HasPrerequisite"},
    {"id": 58, "text": "What must happen before you can use it?  HasPrerequisite",                   "relation": "HasPrerequisite"},
    {"id": 59, "text": "What knowledge or tool do you need before using it?  HasPrerequisite",       "relation": "HasPrerequisite"},

    # ── HasContext ───────────────────────────────────────────
    {"id": 30, "text": "In what specific ambience is it typically used? (lab, sports...)",      "relation": "HasContext"},
    {"id": 60, "text": "In what professional field would you most likely encounter it?  HasContext",        "relation": "HasContext"},
    {"id": 61, "text": "What hobby or discipline is it most associated with?  HasContext",                  "relation": "HasContext"},

]

# implementazione

CSV_PATH = "conceptnet-en.csv"

#in conceptnet gli spazi sono _
def normalize(text: str) -> str:
    return text.lower().strip().replace(" ", "_")


def load_index(csv_path: str) -> dict:
    
    """
    Dato che otteniamo le risposte, usiamo gli end delle relazioni, ovvero le parole in relazione,  in combinazione con le relazioni stesse come indice
      index[(relation, end_std)] = {start_label, ...}

      [Kinfe] [IsA] [Tool]
      [Hammer] [IsA] [Tool]
      [Saw] [IsA] [Tool]

      indice [IsA] [Tool] = knife, hammer, saw
    """
    def add(rel, end, start):
        index.setdefault((rel, end), set()).add(start)

    print("  Loading ConceptNet index...", flush=True)
    index = {}

    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            rel_std   = parts[1]    # /r/Relation
            start_std = parts[2]    # /c/en/knife
            end_std   = parts[3]    # /c/en/cutting

            # Normalizza: spazi -> underscore, tutto minuscolo
            rel = rel_std.split("/")[-1]
            #cutting food -> cutting_food
            end = end_std.split("/c/en/")[-1].split("/")[0].replace(" ", "_").replace("-", "_").lower()
            start = start_std.split("/c/en/")[-1].split("/")[0].replace(" ", "_").replace("-", "_").lower()


            # gestione casi salvataggio
            match rel:

                #cut_food -> cut, cut_food
                case "UsedFor":
                    words = end.split("_")
                    for i in range(1, len(words) + 1):
                        prefix = "_".join(words[:i])
                        add(rel, prefix, start)

                case "AtLocation":
                    add("AtLocation", end, start)

                # aumento il numero di candidati per concetti simili
                case "IsA" | "MemberOf" | "InstanceOf":
                    add("IsA", end, start)
                    add("MemberOf", end, start)
                    add("InstanceOf", end, start)

                # relazione simmetrica
                case "HasA" | "PartOf":
                    add("HasA", end, start)
                    add("PartOf", start, end)

                case "HasProperty" | "CapableOf":
                    add("HasProperty", end, start)
                    add("CapableOf", end, start)

                # relazione simmetrica
                case "Causes":
                    add("Causes", end, start)
                    add("CapableOf", end, start)

                # relazione simmetrica
                case "SimilarTo":
                    add("SimilarTo", end, start)
                    add("SimilarTo", start, end)

                # relazione simmetrica
                case "Antonym":
                    add("Antonym", end, start)
                    add("Antonym", start, end)

                case "HasPrerequisite" | "HasContext":
                    add("HasPrerequisite", end, start)
                    add("HasContext", end, start)

                case _:
                    add(rel, end, start)

    print(f"  Index ready. Relations loaded: {len(index)}")
    return index

# Istanzia l'indice globale (caricato una volta sola)
_INDEX: dict | None = None
def get_index() -> dict:
    global _INDEX
    if _INDEX is None:
        _INDEX = load_index(CSV_PATH)
    return _INDEX

# sessione domande
def run_session(n_questions): # list[dict]

    selected = random.sample(QUESTIONS, min(n_questions, len(QUESTIONS)))
    answers  = []

    print("\n" + "="*50)
    print(" think of an object!")
    print("="*50)
    print("Answer the questions. I'll try to guess your word.\n")

    #salva le domande con i relativi campi
    for q in selected:
        print(f" {q['text']}")
        user_input = input("     Your answer: ").strip().replace(" ", "_")
        answers.append({
            "question": q["text"],
            "relation": q["relation"],
            "answer"  : user_input,
        })
        print()

    return answers


def get_candidates(relation: str, answer: str): # set[str]

    #potrebbero esserci relazioni multiple
    rel    = relation.split("+")[0]
    answer = normalize(answer)
    index  = get_index()
 
    # Exact match: prendo una copia del set che risponde alla query
    results = set(index.get((rel, answer), set()))
 
    # 2. Partial match
    for (r, end), candidates in index.items():
        #Se la risposta è contenuta parzialmente. Es: Cutting contiene Cut
        if r == rel and answer in end and end != answer:
            #unione
            results |= candidates
 
    return results

    '''
    answers.append
            "question": q["text"],
            "relation": q["relation"],
            "answer"  : user_input,   
    '''
def intersect_candidates(answers: list[dict]):
    """
    Per ogni risposta recupera i candidati da ConceptNet.
    Assegna a ogni candidato una probabilità:
      P(word) = numero di query in cui appare / numero totale di query con risultati
    """
    scores  = {}   # word -> conteggio apparizioni
    n_valid = 0    # query che hanno restituito almeno un risultato

    for row in answers:


        answer = row["answer"]

        #se l'utente non ha risposto
        if not answer:
            continue

        n_valid += 1

        print(f"  Querying: [{row['relation']}] -> '{answer}' ...", end=" ", flush=True)
        candidates = get_candidates(row["relation"], answer)
        print(f"{len(candidates)} candidates found")

        #se non ci sono candidati alla risposta
        if not candidates:
            print("    -> Skipped (no results)")
            continue

        # Aggiorna scores e traccia le parole nuove di questa query
        new_words = []
        for word in candidates:
            if word not in scores:
                new_words.append(word)
            scores[word] = scores.get(word, 0) + 1
 
        # Stampa: top 10 per probabilità, a parità ordine per count DESC poi alfabetico
        #-x[1] ordine decrescente, x[0] ordine alfabetico
        top = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:10]
        print(f"    -> Top 10 (valid queries so far: {n_valid}, new words this round: {len(new_words)}):")
        for word, count in top:
            prob  = count / n_valid
            badge = " $" if word in new_words else ""
            print(f"       - {word:<25} P={prob:.2f}  ({count}/{n_valid}){badge}")

    #gioco fallito, nessuna query di successo
    if n_valid == 0:
        return []

    result = []
    for word, count in scores.items():
        freq = count / n_valid
        result.append((word, freq))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def print_summary(answers: list[dict]):
    # matrice relazione | domanda | risposta
    print("\n" + "="*50)
    print("   SESSION SUMMARY")
    print("="*50)
    print(f"{'Relation':<18} {'Question':<47} {'Answer'}")
    print("-"*50)
    for row in answers:
        print(f"{row['relation']:<18} {row['question']:<47} {row['answer']}")
    print("="*100)


def print_result(results: list[tuple[str, float]]):

    #risultato finale con probabilità
    print("\n" + "="*50)
    print("   CONCEPTNET RESULT")
    print("="*50)
 
    if not results:
        print("  No matching concept found.")
    else:
        print(f"  Top candidates ({len(results)} total):")
        for i, (word, prob) in enumerate(results[:10], 1):
            bar = "█" * int(prob * 20)
            print(f"    {i:2}. {word:<25} P={prob:.2f}  {bar}")
 
    print("="*55)



if __name__ == "__main__":
    N = 20  # numero di domande per sessione

    # 1) Q&A con l'utente
    session_answers = run_session(n_questions=N)

    # 2) Riepilogo risposte
    print_summary(session_answers)

    # 3) Lookup su ConceptNet
    print("\n" + "="*50)
    print("   SEARCHING CONCEPTNET...")
    print("="*50)
    candidates = intersect_candidates(session_answers)

    # 4) Risultato finale
    print_result(candidates)