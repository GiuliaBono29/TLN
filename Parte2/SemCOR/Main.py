import math
from collections import defaultdict
import nltk
from nltk.corpus import semcor
from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english'))
from nltk.corpus import wordnet as wn


def get_sentences_from_semcor(): #estrae le frasi dal semcor e i lemmi con rispettivo gold
    sentences = []

    for sentence in semcor.tagged_sents(tag='sem'):

        sentences_words = []
        annotated_words = []

        for entry in sentence:

            #inizio utilizzo di Ia generativa per l'estrazione corretta dei dati
            if isinstance(entry, nltk.Tree):
                sentences_words.extend(entry.leaves())

                # le parole "valide" sono quelle il cui albero ha come etichetta un oggetto Lemma,
                if isinstance(entry.label(), nltk.corpus.reader.wordnet.Lemma):
                    word = entry.label().name()
                    gold = entry.label().synset()
            #fine utilizzo Ai generativa


                    annotated_words.append((word, gold))
            else:
                sentences_words.append(entry)

        sentences_words_clean = clean_sentences_words(sentences_words)
        sentences.append((sentences_words_clean, annotated_words))

    return sentences    #sentences: list[tuple[list[str], list[tuple[Any, Any]]]] = []



inverted_document_frequency = {}

def build_idf(sentences): #contiamo la frequenza nelle frasi del semcor (documenti)

    document_frequency = defaultdict(int)
    N_docs = len(sentences)

    for sentence in sentences:
        sentences_words_clean = sentence[0]  # la lista di parole pulite

        for word in set(sentences_words_clean):
            document_frequency[word] += 1   #conta il numero di apparizioni della parola

    for word, freq in document_frequency.items():
        inverted_document_frequency[word] = math.log(N_docs / freq)     #calcolo idf per ogni parola

def clean_sentences_words(sentences_words):
    cleaned = []
    for t in sentences_words:   #check perchè la signature viene creata concatenando stringhe
        if isinstance(t, list):
            tokens = t
        else:
            tokens = [t]

        for token in tokens:
            token = str(token)
            token_lower = token.lower()
            if (token not in string.punctuation
                    and token_lower not in stop_words
                    and any(c.isalpha() for c in token)):  # almeno una lettera

                cleaned.append(token_lower)
    return cleaned

'''
#versiosne con stopwords

def clean_sentences_words(sentences_words):
    cleaned = []
    for t in sentences_words:
        # se t è una lista, itera sui suoi elementi
        if isinstance(t, list):
            tokens = t
        else:
            tokens = [t]

        for token in tokens:
            token = str(token)
            token_lower = token.lower()
            if (token not in string.punctuation
                    and any(c.isalpha() for c in token)):  # almeno una lettera

                cleaned.append(token_lower)
    return cleaned
'''

def lesk(word, sentence):

    context = set(sentence)

    best_sense = None
    max_overlap = 0.0

    senses = wn.synsets(word) #Recupera tutti i sensi da  wordnet

    if not senses:
        return None

    best_sense = senses[0]

    for sense in senses:
        #contruisce la signature, ovvero la glossa + gli esempi
        gloss = sense.definition()
        examples = " ".join(sense.examples())
        full_synset = gloss + " " + examples
        signature = set(clean_sentences_words(full_synset.split()))
        overlap = compute_overlap(signature, context)

        if overlap > max_overlap:
            max_overlap = overlap
            best_sense  = sense

    return best_sense


#Utilizzo della funzione instersection() suggerito dall'ia
def compute_overlap(signature, context):

    overlap = 0.0

    for word in signature.intersection(context):
        overlap += inverted_document_frequency.get(word,0.0)

    return overlap  #somma degli idf di ogni parola dell'intersezione
    '''
    #versione senza idf
    return len(signature.intersection(context))
    '''



def evaluate_lesk(sentences):
    correct = 0
    total = 0

    for i, (sentences_words_clean, annotated_words) in enumerate(sentences):
        for word, gold in annotated_words:
            predicted = lesk(word, sentences_words_clean)

            total += 1
            if predicted == gold:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Totale istanze : {total}")
    print(f"Corrette       : {correct}")
    print(f"Accuracy       : {accuracy:.2%}")


if __name__ == "__main__":

    sentences = get_sentences_from_semcor()
    for i in range(5):
        print(sentences[i])

    build_idf(sentences)
    evaluate_lesk(sentences)

