import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
from collections import Counter
import math
import random as rnd

def extract_words(N_words=1000):        #crea un array tuple (parola, sysnet, contesto di quella parola)
    all_words = []       #lista delle parole
    word_synsets = []       #lista dei synset
    word_contexts = []      #lista dei contesti per ogni parola

    for sentence in semcor.tagged_sents(tag="sem"):

        context = []
        annotated_words = []      #le parole annotate

        for word_sentence in sentence:  #costruisci il contesto e salva le parole annotate

            if isinstance(word_sentence, nltk.Tree) and isinstance(word_sentence.label(), nltk.corpus.reader.wordnet.Lemma):

                for leaf in word_sentence.leaves():     #gestisce casi come Tree(Lemma('united_states.n.01.United_States'), ['United', 'States'])

                    context.append(leaf)

                annotated_words.append(word_sentence)

        for entry in annotated_words: #ora che contesto è completo, salva tutto

            word = entry.label().name()
            synset = entry.label().synset()
            all_words.append(word)
            word_synsets.append(synset)
            word_contexts.append(context)

    #rnd.seed(3)
    indices = rnd.sample(range(len(all_words)), N_words)       #Campiona le parole
    sampled_word_senses = {}

    for i in indices:

        word = all_words[i]
        sampled_word_senses[word] = []

    for word, synset, context in zip(all_words, word_synsets, word_contexts):

        if word in sampled_word_senses:

            sampled_word_senses[word].append((synset, context))

    for i in range(5):
        print(f"word: {all_words[i]}")
        print(f"  synset : {word_synsets[i]}")
        print(f"  context: {word_contexts[i]}")
        print()

    return sampled_word_senses


def compute_entropy(word_senses):       #Calocla l'entropia standard e normalizzata per ogni parola
    word_entropy = {}
    for word, occurrences in word_senses.items():
        synsets = []
        for synset, context in occurrences:
            synsets.append(synset)


        N = len(synsets)            #numero di volte che la parola appare con un significato
        counts = Counter(synsets)   #Elenco significati con numero ripetizioni
        n_sens = len(counts)        #numero significati differenti

        H = 0.0
        for count in counts.values():
            p = count / N
            H += -(p * math.log2(p))

        # entropia massima: se tutti i sensi di una parola sono usati con la stessa frequenza
        H_max = math.log2(n_sens) if n_sens > 1 else 0
        H_norm = H / H_max if H_max > 0 else 0      # quanto uniformemente i sensi vengono usati nel corpus.
        #H_norm alto → parola ambigua

        word_entropy[word] = (H, H_norm, n_sens)
    return word_entropy

def compute_top1_share(word_senses):
    top1_shares = {}

    for word, occurrences in word_senses.items():

        N = len(occurrences)

        # conta le occorrenze di ogni senso
        counts = Counter(synset for synset, context in occurrences)

        # prende il conteggio del senso più frequente
        top1_synset, top1_count = counts.most_common(1)[0]

        top1_shares[word] = {
            "top1_share": top1_count / N,
            "top1_synset": top1_synset,
        }

    return top1_shares

def analyze_complexity(word_entropy, word_senses, top1_shares):  #divide le parola sulla base dell'entropia e prepara i risultati
    results = {}

    for word, (H, H_norm, n_sens) in word_entropy.items():

        frequency = len(word_senses[word])  # quante volte compare nel corpus

        if H_norm < 0.33:
            complexity = "low"

        elif H_norm < 0.66:
            complexity = "medium"

        else:
            complexity = "high"

        results[word] = {
            "H_norm" : H_norm,
            "n_sens" : n_sens,
            "frequency" : frequency,
            "complexity" : complexity,
            "top1_share" : top1_shares[word]["top1_share"],
            "top1_synset" : top1_shares[word]["top1_synset"]
        }

    return results

#utilizzata ai generativa per produrre delle stampe comprensibili dei risultati
def print_complexity_analysis(results):

    groups = {"low": [], "medium": [], "high": []}
    for word, stats in results.items():     # raggruppa le parole nei 3 livelli di complessità
        groups[stats["complexity"]].append((word, stats))       #a seconda del tag complexity aggiunge la tupla

    for level in ["low", "medium", "high"]:
        words_in_group = groups[level]

        # calcola de statistiche generali della categoria
        avg_freq = sum(s["frequency"] for _, s in words_in_group) / len(words_in_group)
        avg_h_norm = sum(s["H_norm"] for _, s in words_in_group) / len(words_in_group)
        avg_n_sens = sum(s["n_sens"] for _, s in words_in_group) / len(words_in_group)
        avg_top_1 = sum(s["top1_share"] for _, s in words_in_group) / len(words_in_group)

        print(f"\n{'=' * 60}")
        print(f"  Complexity: {level.upper()} ({len(words_in_group)} parole)")
        print(f"  H_norm medio  : {avg_h_norm:.3f}")
        print(f"  n_sens medio  : {avg_n_sens:.1f}")
        print(f"  frequenza media: {avg_freq:.1f}")
        print(f"  Top-1 share medio: {avg_top_1:.1f}")
        print(f"{'=' * 60}")

        # top 5 per H_norm
        print(f"\n  Top 5 per H_norm:")
        print(f"  {'Parola':<20} {'H_norm':>8} {'n_sens':>8} {'Freq':>8} {'top1_share':>12} {'top1_synset'}")
        print("  " + "-" * 80)
        for word, stats in sorted(words_in_group, key=lambda x: x[1]["H_norm"], reverse=True)[:5]:
            print(f"  {word:<20} {stats['H_norm']:>8.3f} {stats['n_sens']:>8} {stats['frequency']:>8} {stats['top1_share']:>12.3f} {stats['top1_synset']}")

        # top 5 per n_sens
        print(f"\n  Top 5 per n_sens:")
        print(f"  {'Parola':<20} {'H_norm':>8} {'n_sens':>8} {'Freq':>8} {'top1_share':>12} {'top1_synset'}")
        print("  " + "-" * 48)
        for word, stats in sorted(words_in_group, key=lambda x: x[1]["n_sens"], reverse=True)[:5]:
            print(f"  {word:<20} {stats['H_norm']:>8.3f} {stats['n_sens']:>8} {stats['frequency']:>8} {stats['top1_share']:>12.3f} {stats['top1_synset']}")

        # top 5 per frequenza
        print(f"\n  Top 5 per frequenza:")
        print(f"  {'Parola':<20} {'H_norm':>8} {'n_sens':>8} {'Freq':>8} {'top1_share':>12} {'top1_synset'}")
        print("  " + "-" * 48)
        for word, stats in sorted(words_in_group, key=lambda x: x[1]["frequency"], reverse=True)[:5]:
            print(f"  {word:<20} {stats['H_norm']:>8.3f} {stats['n_sens']:>8} {stats['frequency']:>8} {stats['top1_share']:>12.3f} {stats['top1_synset']}")


if __name__ == "__main__":

    sampled_words = extract_words()
    word_entropy = compute_entropy(sampled_words)
    top1_shares = compute_top1_share(sampled_words)
    results = analyze_complexity(word_entropy, sampled_words, top1_shares)
    print_complexity_analysis(results)