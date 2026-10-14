import os
import pandas as pd
import re
import random
from collections import defaultdict, Counter

# CARICAMENTO E PULIZIA DEI DATI
def load_tweets(filepath, text_column):
    df = pd.read_csv(filepath, low_memory=False)            #carico csv
    tweets = df[text_column].dropna().tolist()              #estraggo colonna testo, rimuovo NaN e converto in lista
    return tweets

def clean_tweet(tweet):
    tweet = tweet.strip('"')                        # rimuovo virgolette iniziali/finali
    tweet = re.sub(r'http\S+', '', tweet)           # rimuovo URL
    tweet = re.sub(r'www\.\S+', '', tweet)          # rimuovo URL senza http
    tweet = re.sub(r'pic\.twitter\S+', '', tweet)   # rimuovo immagini
    return tweet

def tokenize(tweet):
    tokens = ['<s>'] + tweet.split() + ['</s>']     # transforma in token e aggiunge start/end
    return tokens

# Preprocessa tutti i tweet e ottieni una lista di token (fa clean e tokenize)
def preprocess(tweets):
    all_tokens = []
    for tweet in tweets:
        cleaned = clean_tweet(tweet)
        if cleaned:                    #ignoro tweet vuoti dopo la pulizia
            tokens = tokenize(cleaned)
            all_tokens.extend(tokens)
    return all_tokens

# COSTRUZIONE DEI MODELLI N-GRAM
def build_bigram_model(tokens):
    model = defaultdict(Counter)        #creo dizionario con chiave=parola (wi-1) e valore=parola (wi) + quante volte contato
                                        #es il gatto e il cane -> model["il"] = {"gatto": 1, "cane": 1}
    for i in range(len(tokens) - 1):    #per ogni token
        w1, w2 = tokens[i], tokens[i+1] #conta tutte le volte che una parola w2 segue una parola w1
        model[w1][w2] += 1
    
    for w1 in model:                    #Calcolo delle probabilità
        total = sum(model[w1].values()) #somma totale di tutte le parole che seguono w1
        for w2 in model[w1]:            #tutte le volte che w2 segue w1 diviso il totale di tutte le parole che seguono w1
            model[w1][w2] /= total      #es model["il"] = {"gatto": 3, "cane": 2, "cavallo": 1} total=6
                                        #gatto: 3/6=0.5, cane: 2/6=0.33, cavallo: 1/6=0.17
    return model

def build_trigram_model(tokens):
    model = defaultdict(Counter)        #creo dizionario con chiave=tupla (wi-2, wi-1) e valore=parola (wi) + quante volte contato
                                        #es il gatto dorme -> model[("il", "gatto")] = {"dorme": 1} 
    for i in range(len(tokens) - 2):                        #per ogni token
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]    #conta tutte le volte che una parola w3 segue la coppia di parole (w1, w2)
        model[(w1, w2)][w3] += 1
    
    for context in model:                       #Calcolo delle probabilità. context = tupla (wi-2, wi-1)
        total = sum(model[context].values())    #somma totale di tutte le parole che seguono w1, w2
        for w3 in model[context]:               
            model[context][w3] /= total         #tutte le volte che w3 segue (w1, w2) diviso il totale di tutte le parole che seguono (w1, w2)

    return model

# GENERAZIONE DEI TWEET

def generate_bigram_tweet(model, max_len=30):
    current_word = '<s>'                #partendo dalla stringa iniziale start
    tweet = []
    
    for _ in range(max_len):            #fino alla lunghezza massima si trova la parola seguente più probabile
        next_words = model[current_word]
        if not next_words:
            break
        
        #Campiono la prossima parola in base alle probabilità
        word = random.choices(
            list(next_words.keys()),
            weights=list(next_words.values())
        )[0]
        #es model["il"] = {"gatto": 0.5, "cane": 0.33, "cavallo": 0.17} -> 50% gatto, 33% cane, 17% cavallo
        
        if word == '</s>':
            break
            
        tweet.append(word)
        current_word = word
    
    return ' '.join(tweet)


def generate_trigram_tweet(model, bigram_model, max_len=30):
    # Per il trigramma partiamo con <s> + prima parola campionata dal bigramma
    first_word = random.choices(
        list(bigram_model['<s>'].keys()),
        weights=list(bigram_model['<s>'].values())
    )[0]
    
    tweet = [first_word]
    prev_prev = '<s>'
    prev = first_word
    
    for _ in range(max_len):
        context = (prev_prev, prev)
        next_words = model[context]
        
        if not next_words:                  #BACKOFF: se non ci sono parole che seguono la coppia (prev_prev, prev),
                                            #         torno al modello bigramma usando solo prev
            next_words = bigram_model[prev]
            if not next_words:
                break
        
        word = random.choices(
            list(next_words.keys()),
            weights=list(next_words.values())
        )[0]
        
        if word == '</s>':
            break
            
        tweet.append(word)
        prev_prev = prev
        prev = word
    
    return ' '.join(tweet)


#salvataggio su file di output
def save_tweets_to_file(filename, trump_bigram, trump_trigram, musk_bigram, musk_trigram, n=50):
    with open(filename, 'w', encoding='utf-8') as f: 
        
        f.write("=" * 60 + "\n")
        f.write("TRUMP - BIGRAMMI\n")
        f.write("=" * 60 + "\n")
        for i in range(n):
            f.write(f"{i+1}. {generate_bigram_tweet(trump_bigram)}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("TRUMP - TRIGRAMMI\n")
        f.write("=" * 60 + "\n")
        for i in range(n):
            f.write(f"{i+1}. {generate_trigram_tweet(trump_trigram, trump_bigram)}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("MUSK - BIGRAMMI\n")
        f.write("=" * 60 + "\n")
        for i in range(n):
            f.write(f"{i+1}. {generate_bigram_tweet(musk_bigram)}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("MUSK - TRIGRAMMI\n")
        f.write("=" * 60 + "\n")
        for i in range(n):
            f.write(f"{i+1}. {generate_trigram_tweet(musk_trigram, musk_bigram)}\n")
    
    print(f"File salvato: {filename}")


# ESECUZIONE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Carica i tweet dai csv
trump_tweets  = load_tweets(os.path.join(BASE_DIR, 'realdonaldtrump.csv'), 'content')
musk_tweets   = load_tweets(os.path.join(BASE_DIR, 'all_musk_posts.csv'), 'fullText')

# Preprocessa facendo clean e tokenize
trump_tokens = preprocess(trump_tweets)
musk_tokens  = preprocess(musk_tweets)

# Costruisci i modelli
trump_bigram  = build_bigram_model(trump_tokens)
trump_trigram = build_trigram_model(trump_tokens)

musk_bigram   = build_bigram_model(musk_tokens)
musk_trigram  = build_trigram_model(musk_tokens)

print("\nModelli costruiti con successo!")

output_path = os.path.join(BASE_DIR, 'generated_tweets.txt')
save_tweets_to_file(output_path, trump_bigram, trump_trigram, musk_bigram, musk_trigram, n=50)