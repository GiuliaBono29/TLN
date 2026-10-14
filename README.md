# Tecnologie del Linguaggio Naturale (TLN) – Progetti del corso

Progetti sviluppati per il corso di **Tecnologie del Linguaggio Naturale** della Laurea Magistrale in Informatica, Università di Torino. Coprono l'NLP statistico, dall'etichettatura di sequenze ai modelli di linguaggio fino alla semantica lessicale. Il progetto è stato sviluppato in gruppo per il corso di TLN, insieme a un collaboratore.

## Tecnologie usate

* Python
* NLTK (WordNet, SemCor)
* sentence-transformers (SBERT, `all-MiniLM-L6-v2`)
* BERTopic, UMAP, HDBSCAN
* pandas, NumPy, SciPy, Matplotlib

## Progetti

### Parte 1 – POS tagger HMM per lingue antiche

POS tagger basato su Hidden Markov Model a bigrammi con algoritmo di Viterbi (in spazio logaritmico), addestrato sui treebank Universal Dependencies di latino medievale (LLCT) e greco antico (Perseus). Sono state confrontate sei strategie di smoothing per le parole sconosciute, portando l'accuratezza da una baseline del 62,08% (latino) e 28,18% (greco) a circa il 98% e il 76%.

### Parte 2 – Generazione di tweet con N-grammi e Word Sense Disambiguation

* **Generazione di tweet:** modelli di linguaggio a bigrammi e trigrammi (con backoff) addestrati sui tweet di Donald Trump ed Elon Musk, usati per generare testo e analizzare quanto bene vengano catturati temi e stile di ciascun autore.
* **Word Sense Disambiguation:** algoritmo di Lesk semplificato, con e senza pesatura IDF, valutato su SemCor con le glosse di WordNet (accuratezza di circa il 45%).

### Parte 3 – Laboratori di semantica lessicale

* **Lab 1:** similarità lessicale (Jaccard) e semantica (coseno su SBERT) tra le definizioni date da persone diverse per uno stesso concetto, raggruppate per concretezza e specificità.
* **Lab 2:** ricerca onomasiologica del synset di WordNet a partire da una definizione, confrontando approccio per overlap, SBERT e ibrido.
* **Lab 3:** ambiguità lessicale delle parole in SemCor misurata con l'entropia di Shannon.
* **Lab 4:** topic modeling con BERTopic (embedding SBERT, UMAP, HDBSCAN) su circa 45.000 abstract di arXiv NLP, con confronto tra configurazioni per numero di topic, outlier e tempi di esecuzione.
* **Lab 5:** gioco "20 domande" basato su ConceptNet: il sistema pone domande legate alle relazioni del grafo (UsedFor, AtLocation, IsA, ...) e ordina i concetti candidati in base a quante risposte soddisfano.

