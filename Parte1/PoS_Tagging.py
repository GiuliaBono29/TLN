import math
import re
from collections import defaultdict

from debugpy.common.timestamp import current
from overrides.typing_utils import unknown


# LEARNING
# Count PAROLA-TAG C(t,w)
def build_tagging(file_path):
    corpus = defaultdict(lambda: defaultdict(int))

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")

            if "-" in cols[0] or "." in cols[0]:    #togliamo punteggiatura - e .
                continue

            word = cols[1]  #parola nella colonna 1

            tag = cols[3]   #tag nella colonna 3

            if "+" in word: #togliamo punteggiatura +
                continue

            corpus[word][tag] += 1  #nel corpus dico che quella parola ha quel tag +1 volta 
            
    return corpus

# Count OCCURRENCE-TAG C(t)
def build_occurrence_tag(corpus):
    tag_counts = defaultdict(int)

    for word, tags in corpus.items():
        for tag, count in tags.items():
            tag_counts[tag] += count        #incremento ogni volta che trovo un dato tag

    return tag_counts

# Count BIGRAM C(t_{i-1}, t_i)
def build_tag_bigrams(file_path):
    bigram_counts = defaultdict(lambda: defaultdict(int))
    START = "<s>"
    END = "</s>"

    with open(file_path, encoding="utf-8") as f:
        sentence_tags = []

        for line in f:
            line = line.strip()
            if not line:    #caso base -> riga VUOTA
                if sentence_tags:
                    prev_tag = START    #per l'inizio della frase
                    for tag in sentence_tags:
                        bigram_counts[prev_tag][tag] += 1   #conto i bigrammi
                        prev_tag = tag  #sposto il tag
                    bigram_counts[prev_tag][END] += 1   #conto il bigramma finale (tag, END)
                    sentence_tags = []  #cancello per ricominciare
                continue

            if line.startswith("#"):
                continue

            cols = line.split("\t") #divisione colonne

            if "-" in cols[0] or "." in cols[0]:    #togliamo punteggiatura - e .
                continue

            if "+" in cols[1]: #togliamo punteggiatura +
                continue

            tag = cols[3]

            sentence_tags.append(tag)   #aggiungo il tag alla lista dei tag della frase corrente
            
    # Se il file finisce senza una riga vuota, processa l'ultima frase        
    if sentence_tags:
        prev_tag = START
        for tag in sentence_tags:
            bigram_counts[prev_tag][tag] += 1
            prev_tag = tag
        bigram_counts[prev_tag][END] += 1

    return bigram_counts

#NOTA PROBABILITÀ EMISSIONE E TRANSIZIONE:
#si potrebbero calcolare o prima in base alle parole del corpus o direttamente durante l'esecuzione di Viterbi
#volendo fare il test con il test set, conviene calcolare tutto prima, in quanto le probabilità si potranno
#ripetere più volte

#Probabilità emissione P(w|t) = C(t,w) / C(t)
#corpus risultato di build_tagging              es. corpus["in"]["ADP"] = 532
#tag_counts risultato di build_occurrence_tag   es. tag_counts["ADP"]  = 2194
def compute_emission_probs(corpus, tag_counts):
    emission = defaultdict(lambda: defaultdict(float))
    
    for word, tags in corpus.items():
        for tag, count in tags.items():
            emission[tag][word] = count / tag_counts[tag]
    
    return emission

#Probabilità transizione P(t_i | t_{i-1}) = C(t_{i-1}, t_i) / C(t_{i-1})
#bigram_counts risultato di build_tag_bigrams   es. bigram_counts["ADP"]["NOUN"] = 924
def compute_transition_probs(bigram_counts):
    transition = defaultdict(lambda: defaultdict(float))
    
    for prev_tag, next_tags in bigram_counts.items():
        total = sum(next_tags.values())             #faccio la somma di tutti i bigrammi che partono da prev_tag
                                                    #non uso tag_counts perché questo non continene <s>
        for next_tag, count in next_tags.items():
            transition[prev_tag][next_tag] = count / total
    
    return transition

#FINE LEARING

#HAPAX
#calcolo nel devset le distribuzioni di tag per parole viste 1 sola volta (hapax)
# e uso questa distribuzione per assegnare probabilità di emissione alle parole sconosciute nel test set
def build_hapax(dev_path, training_corpus):
    # contiamo le parole nel dev set
    word_counts = defaultdict(int)
    word_tags   = defaultdict(lambda: defaultdict(int))

    sentences = extract_sentences(dev_path)
    for sentence in sentences:
        for word, tag in sentence:
            word_counts[word] += 1
            word_tags[word][tag] += 1

    # teniamo solo le hapax
    hapax_tag_counts = defaultdict(int)
    total = 0   #totale hapax

    for word, count in word_counts.items(): # per le parole trovate prima
        if count == 1:                      # se è un hapax
            tag = list(word_tags[word].keys())[0]  # prendo il tag associato a quell'hapax
            hapax_tag_counts[tag] += 1
            total += 1

    # calcoliamo la distribuzione
    hapax_probs = {}
    for tag, count in hapax_tag_counts.items():
        hapax_probs[tag] = count / total

    return hapax_probs

#FINE HAPAX

#VITERBI
#Input:
#sentence: lista di parole da taggare
#emission: emission[tag][word] = P(word | tag)
#transition: transition[prev_tag][next_tag] = P(next_tag | prev_tag)
#tag_counts: tag_counts[tag] = numero totale di occorrenze del tag
#known_words: insieme di parole conosciute
#hapax_probs: hapax_probs[tag] = P(hapax | tag) distribuzione di tag per parole hapax
def viterbi(sentence, emission, transition, tag_counts, known_words, hapax_probs):
    tags      = list(tag_counts.keys()) #lista dei tag
    n_tags    = len(tags)               #numero di tag
    n_words   = len(sentence)           #numero di parole nella frase

    # Mappa tag -> indice per accedere alle colonne
    # es. {"NOUN": 0, "VERB": 1, "ADP": 2, ...} per muoversi tra i tag con indici
    #tag_to_idx = {tag: i for i, tag in enumerate(tags)}

    # Backpointer: matrice intera n_words × n_tags
    # backpointer[t][j] = indice del tag migliore al passo t-1
    # Salvo gli indici numerici es parola 4 ha il backpointer a 2, che corrisponde a ADP
    backpointer = [[0] * n_tags for _ in range(n_words)]

    # PASSO 1: Inizializzazione
    #prima colonna tutta a 0, poi aggiorno in base alla prima parola
    prev_col = [0.0] * n_tags
    word = sentence[0]

    for i, tag in enumerate(tags):
        trans_prob = transition["<s>"].get(tag, 0)  # P(tag | START)
        emit_prob = emission[tag].get(word, 0)  # P(word | tag)
        
        smooth = 8
        if word not in known_words:
            if smooth == 0:
                emith_prob = 0
            elif smooth == 1:
                emit_prob = 1e-6
            elif smooth == 2:
                emit_prob = 1.0 if tag == "NOUN" else 0.0
            elif smooth == 3:
                emit_prob = 1.0 if tag == "VERB" else 0.0
            elif smooth == 4:
                emit_prob = 1.0 if tag == "PROPN" else 0.0
            elif smooth == 5:
                emit_prob = 0.5 if tag in ("NOUN", "VERB") else 0.0
            elif smooth == 6:
                emit_prob = 0.3 if tag in ("NOUN", "VERB", "PROPN") else 0.0
            elif smooth == 7:
                emit_prob = 1 / n_tags
            elif smooth == 8:
                emit_prob = hapax_probs.get(tag, 0)


        # Per evitare il problema di underflow usiamo i logaritmi
        if trans_prob > 0 and emit_prob > 0:
            prev_col[i] = math.log(trans_prob) + math.log(emit_prob)
        else:
            prev_col[i] = float("-inf")
        
        backpointer[0][i] = -1              # nessun predecessore

    # PASSO 2: Ricorrenza
    for t in range(1, n_words):
        curr_col = [0.0] * n_tags
        word = sentence[t]

        for j, curr_tag in enumerate(tags):

            emit_prob = emission[curr_tag].get(word, 0)  # P(word | curr_tag)

            smooth = 8
            if word not in known_words:
                if smooth == 0:
                    emith_prob = 0
                elif smooth == 1:
                    emit_prob = 1e-6
                elif smooth == 2:
                    emit_prob = 1.0 if curr_tag == "NOUN" else 0.0
                elif smooth == 3:
                    emit_prob = 1.0 if curr_tag == "VERB" else 0.0
                elif smooth == 4:
                    emit_prob = 1.0 if curr_tag == "PROPN" else 0.0
                elif smooth == 5:
                    emit_prob = 0.5 if curr_tag in ("NOUN", "VERB") else 0.0
                elif smooth == 6:
                    emit_prob = 0.3 if curr_tag in ("NOUN", "VERB", "PROPN") else 0.0
                elif smooth == 7:
                    emit_prob = 1 / n_tags
                elif smooth == 8:
                    emit_prob = hapax_probs.get(curr_tag, 0)



            if emit_prob > 0:
                log_emit = math.log(emit_prob)
            else:
                log_emit = float("-inf")

            best_score = float("-inf")
            best_prev  = 0

            for i, prev_tag in enumerate(tags):
                trans_prob = transition[prev_tag].get(curr_tag, 0)

                if trans_prob > 0 and prev_col[i] > float("-inf"):
                    score = prev_col[i] + math.log(trans_prob)
                else:
                    score = float("-inf")

                if score > best_score:
                    best_score = score
                    best_prev  = i

            curr_col[j] = best_score + log_emit if log_emit > float("-inf") else float("-inf")  #aggiorno la colonna corrente con il miglior punteggio trovato
                                                                                                #prodotto tra prob emissione e transizione
                                                                                                #-> somma dei logaritmi
            backpointer[t][j] = best_prev   #salvo il backpointer

        prev_col = curr_col  #scorri avanti, prev_col vecchia viene scartata

    # PASSO 3: Terminazione
    best_last_score = float("-inf")
    best_last_idx   = 0

    for i, tag in enumerate(tags):
        end_prob = transition[tag].get("</s>", 0)
        if end_prob > 0 and prev_col[i] > float("-inf"):
            score = prev_col[i] + math.log(end_prob)
        else:
            score = float("-inf")

        if score > best_last_score:
            best_last_score = score
            best_last_idx   = i

    # PASSO 4: Backtracking
    result = [None] * n_words
    result[n_words - 1] = tags[best_last_idx]

    prev_best = best_last_idx
    for t in range(n_words - 2, -1, -1):    #parti dal penultimo passo e vai indietro fino al primo
        prev_best = backpointer[t + 1][prev_best]
        result[t] = tags[prev_best]

    return result

#FINE VITERBI

#TESTING
#estrazione frasi e tag dal file CoNLL-U per testare Viterbi su frasi del dev set
def extract_sentences(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Riga vuota = fine frase
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            # Salta commenti
            if line.startswith("#"):
                continue

            cols = line.split("\t")

            # Salta multi-word token (es. "1-2") e empty nodes (es. "1.1")
            if "-" in cols[0] or "." in cols[0]:
                continue

            # Salta parole con "+"
            if "+" in cols[1]:
                continue

            word = cols[1]
            tag  = cols[3]
            current_sentence.append((word, tag))
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

#Data una lista di frasi (word, tag), estrae solo le parole testare Viterbi
def extract_words(sentences):
    return [[word for word, tag in sentence] for sentence in sentences]

#Data una lista di frasi (word, tag), estrae solo i tag gold per testare Viterbi
def extract_tags(sentences):
    return [[tag for word, tag in sentence] for sentence in sentences]
#FINE TESTING

# BASELINE
# Per ogni parola assegna il tag più frequente nel training, se non è conosciuta nome
def baseline_most_frequent(sentence, tagging):
    result = []
    for word in sentence:
        if word in tagging:
            best_tag = max(tagging[word], key=tagging[word].get)
        else:
            best_tag = "NOUN"
        result.append(best_tag)
    return result
# FINE BASELINE

#VALUTAZIONE
#Per ogni frase del test set, esegue Viterbi e confronta con i tag gold, calcolando accuracy totale, errori su parole conosciute vs sconosciute e stampa risultati
def evaluate(test_path, emission, transition, occurrences, known_words, hapax_probs, language=""):
    test_sentences = extract_sentences(test_path)
    test_words     = extract_words(test_sentences)
    test_tags      = extract_tags(test_sentences)

    # Parole sconosciute
    unknown_words = set()
    for sentence in test_words:
        for word in sentence:
            if word not in known_words:
                unknown_words.add(word)

    total_words   = 0
    total_correct = 0
    total_errors  = 0

    for sentence, gold in zip(test_words, test_tags):                #per ogni frase calcolo viterbi
        predicted = viterbi(sentence, emission, transition, occurrences, known_words, hapax_probs)
        for pred, true in zip(predicted, gold):         #confronto predetti con gold
            total_words += 1
            if pred == true:
                total_correct += 1
            else:
                total_errors += 1

    accuracy = total_correct / total_words if total_words > 0 else 0

    print(f"\n=== RISULTATI {language.upper()} ===")
    print(f"  Frasi nel test set:    {len(test_sentences)}")
    print(f"  Parole totali:         {total_words}")
    print(f"  Parole sconosciute:    {len(unknown_words)}")
    print(f"  Corrette:              {total_correct}")
    print(f"  Errate:                {total_errors}")
    print(f"  Accuracy:              {accuracy:.2%}")

    return accuracy
#FINE VALUTAZIONE

if __name__ == "__main__":
    #LATINO
    latin_train_path = r"C:\Users\GiuliaBono\Desktop\Università\Magistrale\Secondo Anno\TLN\Consegna1\UD_Latin-LLCT-master\la_llct-ud-train.conllu"
    latin_dev_path = r"C:\Users\GiuliaBono\Desktop\Università\Magistrale\Secondo Anno\TLN\Consegna1\UD_Latin-LLCT-master\la_llct-ud-dev.conllu"
    latin_test_path = r"C:\Users\GiuliaBono\Desktop\Università\Magistrale\Secondo Anno\TLN\Consegna1\UD_Latin-LLCT-master\la_llct-ud-test.conllu"

    #LEARNING su training
    print("=== LEARNING LATINO ===")
    tagging_l     = build_tagging(latin_train_path) #quante volte una parola ha un dato tag
    occurrences_l = build_occurrence_tag(tagging_l) #quante volte un dato tag compare nel corpus
    known_words_l = set(tagging_l.keys())   #insieme di parole conosciute (dal corpus)
    tag_bigrams_l = build_tag_bigrams(latin_train_path) #quante volte un dato tag segue un altro dato tag (bigrammi)
    emission_l    = compute_emission_probs(tagging_l, occurrences_l)    #calcolo prob di emissione
    transition_l  = compute_transition_probs(tag_bigrams_l)     #calcolo prob di transizione
    #SMOOTHING PER HAPAX su dev
    hapax_probs_l = build_hapax(latin_dev_path, tagging_l)      #calcolo hapax per parole sconosciute

    #VALUTAZIONE su test
    acc_latin = evaluate(latin_test_path, emission_l, transition_l, occurrences_l, known_words_l, hapax_probs_l, language="Latino")
    
    
    test_sentences_l = extract_sentences(latin_test_path)
    test_words_l     = extract_words(test_sentences_l)
    test_tags_l      = extract_tags(test_sentences_l)

    total_base_l   = 0
    correct_base_l = 0
    #Costruzione baseline: tag più frequent sennò nome
    for words, gold in zip(test_words_l, test_tags_l):
        predicted = baseline_most_frequent(words, tagging_l)
        for pred, true in zip(predicted, gold): #calcolo valori corretti per baseline
            total_base_l += 1
            if pred == true:
                correct_base_l += 1
                
    acc_latin_base = correct_base_l / total_base_l
    print(f"\n=== RISULTATI LATINO BASELINE ===")
    print(f"  Accuracy baseline Latino: {acc_latin_base:.2%}")

    #GRECO
    greek_train_path = r"C:\Users\GiuliaBono\Desktop\Università\Magistrale\Secondo Anno\TLN\Consegna1\UD_Ancient_Greek-Perseus-master\grc_perseus-ud-train.conllu"
    greek_dev_path   = r"C:\Users\GiuliaBono\Desktop\Università\Magistrale\Secondo Anno\TLN\Consegna1\UD_Ancient_Greek-Perseus-master\grc_perseus-ud-dev.conllu"
    greek_test_path  = r"C:\Users\GiuliaBono\Desktop\Università\Magistrale\Secondo Anno\TLN\Consegna1\UD_Ancient_Greek-Perseus-master\grc_perseus-ud-test.conllu"

    #LEARNING
    print("\n=== LEARNING GRECO ===")
    tagging_g     = build_tagging(greek_train_path)
    occurrences_g = build_occurrence_tag(tagging_g)
    known_words_g = set(tagging_g.keys())
    tag_bigrams_g = build_tag_bigrams(greek_train_path)
    emission_g    = compute_emission_probs(tagging_g, occurrences_g)
    transition_g  = compute_transition_probs(tag_bigrams_g)
    hapax_probs_g = build_hapax(greek_dev_path, tagging_g)

    #VALUTAZIONE
    acc_greek = evaluate(greek_test_path, emission_g, transition_g, occurrences_g, known_words_g, hapax_probs_g, language="Greco")
    
    test_sentences_g = extract_sentences(greek_test_path)
    test_words_g     = extract_words(test_sentences_g)
    test_tags_g      = extract_tags(test_sentences_g)

    total_base_g   = 0
    correct_base_g = 0
    for words, gold in zip(test_words_g, test_tags_g):
        predicted = baseline_most_frequent(words, tagging_g)
        for pred, true in zip(predicted, gold):
            total_base_g += 1
            if pred == true:
                correct_base_g += 1
    acc_greek_base = correct_base_g / total_base_g
    print(f"\n=== RISULTATI GRECO BASELINE ===")
    print(f"  Accuracy baseline Greco: {acc_greek_base:.2%}")

    #RISULTATI FINALI
    print(f"\n{'='*50}")
    print(f"  CONFRONTO FINALE")
    print(f"{'='*50}")
    print(f"  {'':30} {'Latino':>10} {'Greco':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline':30} {acc_latin_base:>10.2%} {acc_greek_base:>10.2%}")
    print(f"  {'HMM + Viterbi (hapax)':30} {acc_latin:>10.2%} {acc_greek:>10.2%}")