import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from nltk.corpus import CategorizedPlaintextCorpusReader, stopwords
import matplotlib.pyplot as plt
import seaborn as sns



root_dir = 'contracts'

contract_corpus = CategorizedPlaintextCorpusReader(
    './%s/' % root_dir,
    r'.*\.sol',
    cat_pattern=r'(\w+)/*',
    encoding='latin-1'
)
contracts = [(contract_corpus.raw(fileid), category)
              for category in contract_corpus.categories()
              for fileid in contract_corpus.fileids(category)]


def preprocess(doc, deacc=False, min_len=3, max_len=15):
    tokens = [
        token for token in gensim.utils.tokenize(doc, lower=False, deacc=deacc, errors='ignore')
        if min_len <= len(token) <= max_len and not token.startswith('_')
    ]
    return tokens



def sent_to_words(sentences):
    for sent in sentences:
        # Remove id contracts
        sent = re.sub('\S*0x\S*\s?', '', sent)

        # Remove new line characters
        sent = re.sub('\s+', ' ', sent)

        # Remove distracting single quotes
        sent = re.sub("\'", "", sent)

        # Remove charaters with exponent
        sent = re.sub("\S*½|¼|¾|¹|²|³|º\S", '', sent)

        # Remove alphanumerical words with lenght 64 charaters (adress)
        sent = re.sub("\w{64}", "", sent)

        yield(preprocess(str(sent), deacc=True))  # deacc=True removes punctuations



def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    parole=[m.group(0) for m in matches]
    return parole

english_stopwords = stopwords.words('english')

english_stopwords.extend(
        ['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get',
         'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot',
         'lack',
         'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

otherstopwords = ['http', 'https', 'file', 'token', 'owner', 'b', 'c', 'text', 'index', 'amount', 'github', 'com',
                  'referal', 'percent', 'percentage', 'length', 'total', 'account', 'sub', 'add', 'success',
                  'successful', 'name', 'sender', 'erc', 'standard', 'basic', 'approval', 'safe', 'allow', 'allowance',
                  'supply', 'balance', 'eth', 'ethereum', 'source', 'blockchain', 'sol', 'gas', 'decimal', 'unit',
                  'initial', 'enough', 'check', 'overflow', 'recipient', 'set', 'give', 'display', 'creator', 'div',
                  'rev', 'min', 'max', 'mul', 'callable', 'update', 'last', 'current', 'array', 'already','approve', 'code',
                  'subtract', 'addition', 'receive', 'addition', 'implement', 'user']
english_stopwords.extend(otherstopwords)

solidity_stopwords = ['pragma', 'bytes', 'bytes32', 'type', 'int', 'string', 'fallback', 'bool', 'var', 'msgvalue',
                      'balance', 'plus', 'minus', 'symbol', 'geq', 'gt', 'eq', 'hex', '&&', 'and', 'or', 'not',
                      'condexpr', 'call', 'delegatecall','send', 'transfer', 'exprnil', 'pnil', 'pcons', 'view',
                      'visibility', 'payable', 'title', 'this', 'solidity', 'uint', 'function', 'meth', 'value', 'return',
                      'returns', 'castinterf', 'interface', 'declaration', 'if', 'else', 'ifelse', 'assigmenet',
                      'sequence', 'any_funct', 'contract_ast', 'cast', 'addr', 'address', 'mapping', 'dev', 'event',
                      'require', 'struct', 'constant', 'constructor', 'param', 'public', 'private', 'contract',
                      'modifier', 'msg', 'true', 'false', 'pure', 'delete', 'payable', 'throw',
                      'memory', 'new', 'internal', 'ether', 'arg', 'args', 'dynargs', 'timestamp', 'datasource',
                      'gaslimit', 'library', 'id', 'using', 'encode', 'external', 'buffer', 'safemath', 'revert', 'emit',
                      'assert', 'callcode', 'error']

english_stopwords.extend(solidity_stopwords)



def list_of_contracts(documents):
    nli = []
    result = []
    for doc in documents:
        nli = []
        for w in doc:
            if len(w) > 1:
                for wo in w:
                    nli.append(wo)
            if len(w)==1:
                nli.append(w[0])

        result.append(nli)

    return result





def lemmatization(texts,
                  stop_words=english_stopwords,
                  allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'],
                  bigram_mod=None):

    texts = [[word for word in preprocess(str(doc)) if word.lower() not in english_stopwords] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner', 'textcat']) #  keeping only tagger component (for efficiency). spacy.load('en', disable=['parser', 'ner', 'textcat'])
    nlp.max_length = 10000000

    newdocs=[]
    for doc in texts:
        #print("sent: " ,sent)
        newsent=[]
        for token in doc:
            newsent.append([word.lower() for word in camel_case_split(token) if word.lower() not in english_stopwords ])
        newdocs.append(newsent)

    result= list_of_contracts(newdocs)
    #print("result: " ,result)
    texts_out = []
    for docum in result:
        document = nlp("".join(str(docum)))
        #print("doc: ", document)
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in document if token.pos_ in allowed_postags and token.lemma_ not in english_stopwords]))
    return texts_out



# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# Run in terminal: python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def getBestKForKMeans(data_vectorized, maxClusters=20):
    sumOfSquareDistance=[]
    K=range(2, maxClusters)
    for k in K:
        km = KMeans(n_clusters=k)
        km= km.fit(data_vectorized)
        sumOfSquareDistance.append(km.inertia_)
    print(sumOfSquareDistance)
    plt.plot(K, sumOfSquareDistance, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum Of Squared Distances')
    plt.title('Elbow Method for Optimal K')
    plt.show()

if __name__ == '__main__':
    n_docs= 30
    print("Preprocessing...")
    data= [str(nltk.Text(contract)) for contract in contracts]

    data_words = list(sent_to_words(data))

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    data_ready = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'], bigram_mod= bigram_mod)
    print(data_ready[:3])


    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=10,  # minimum reqd occurences of a word
                                 stop_words= 'english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=50000,             # max number of uniq words
                                 )

    data_vectorized = vectorizer.fit_transform(data_ready)


    # Materialize the sparse data
    data_dense = data_vectorized.todense()

    # Compute Sparsicity = Percentage of Non-Zero cells
    print("Sparsicity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")



    print("Building LDA model...")
    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=15,  # Number of topics
                                          max_iter=10,  # Max learning iterations
                                          learning_method='online',
                                          random_state=100,  # Random state
                                          batch_size=128,  # n docs in each learning iter
                                          evaluate_every=-1,  # compute perplexity every n iters, default: Don't
                                          n_jobs=-1,  # Use all available CPUs
                                          ).fit(data_vectorized)


    lda_output = lda_model.fit_transform(data_vectorized)
    print(lda_model)  # Model attributes

    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda_model.score(data_vectorized))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda_model.perplexity(data_vectorized))

    # See model parameters
    pprint(lda_model.get_params())

    # Define Search Param
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

    # Init the Model
    lda = LatentDirichletAllocation(learning_method='online')

    # Init Grid Search Class
    print("GridSearchCV...")
    model = GridSearchCV(lda, param_grid=search_params)

    # Do the Grid Search
    model.fit(data_vectorized)

    # Best Model
    best_lda_model = model.best_estimator_

    # Model Parameters
    print("Best Model's Params: ", model.best_params_)

    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)

    # Perplexity
    print("Best Model Perplexity: ", best_lda_model.perplexity(data_vectorized))


    print("Compare LDA Model Performance Scores")


    results = pd.DataFrame(model.cv_results_)

    current_palette = sns.color_palette("Set2", 3)

    plt.figure(figsize=(12, 8))

    sns.lineplot(data=results,
                 x='param_n_components',
                 y='mean_test_score',
                 hue='param_learning_decay',
                 palette=current_palette,
                 marker='o'
                 )

    plt.show()

    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

    #index names
    docnames = ["Doc" + str(i) for i in range(len(contracts))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # Styling
    def color_green(val):
        color = 'green' if val > .1 else 'black'
        return 'color: {col}'.format(col=color)


    def make_bold(val):
        weight = 700 if val > .1 else 400
        return 'font-weight: {weight}'.format(weight=weight)

    # Apply Style
    df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    with open('./report/dominant_topic_for_each_document.html', 'w') as fileWriter:
        fileWriter.write(df_document_topics.render())


    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    pd.DataFrame(df_topic_distribution).to_csv('report/topic_distribution.csv')


    getBestKForKMeans(data_vectorized, maxClusters=20)

