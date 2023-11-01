# Smart Contract: topic modeling 

Questo repository contiene l'analisi di 47,398 smart contracts che sono stati clonati dal seguente repository: [SmartBugs](https://github.com/smartbugs/smartbugs-wild).
Si è proceduto a una clusterizzazione dei contratti sulla base del linguaggio naturale contenuto in ciascun contratto, seguita da una topic modeling.

![Creative and Minimal Portfolio Presentation-2](https://github.com/CariaStefano/Smart-contracts-topic-modeling/assets/79990918/323b14bc-7522-4067-9364-ec0944449530)


## Clusterization
**cluster analysis with tsne and bokeh.**
La clusterizzazione dei contratti si basa in particolare su 2 librerie: tsne e Bokeh.




## Topic modeling 

**Topic modeling with Gensim.**
Per l'analisi dei topic è stata utilizzata la libreria Gensim, la quale implementa l'algoritmo LDA (Latent Dirichlet Allocation).

## Spacy
Per l'eseguzione del codice è necessario, oltre all'installazione delle 
dipendenze eseguire il seguente comando dal terminale:

python3 -m spacy download en_core_web_sm

