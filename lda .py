# Gensim
import warnings
import nltk
warnings.filterwarnings("ignore")
import gensim
import gensim.corpora as corpora
from pprint import pprint
import numpy as np
import pandas as pd
import pyLDAvis
from pyLDAvis.gensim_models import prepare
import seaborn as sns
import spacy
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_file, save
from nltk.corpus import CategorizedPlaintextCorpusReader
import re
from gensim.models import CoherenceModel
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from collections import Counter
# NLTK Stop words
from nltk.corpus import stopwords
from sklearn.manifold import TSNE

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

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    words=[m.group(0).replace('_', '') for m in matches]
    return words


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
                  'rev', 'min', 'max', 'mul', 'callable', 'update', 'last', 'current', 'array', 'already', 'approve',
                  'code',
                  'subtract', 'addition', 'receive', 'addition', 'implement', 'user']
english_stopwords.extend(otherstopwords)

solidity_stopwords = ['pragma', 'bytes', 'bytes32', 'type', 'int', 'string', 'fallback', 'bool', 'var', 'msgvalue',
                      'balance', 'plus', 'minus', 'symbol', 'geq', 'gt', 'eq', 'hex', '&&', 'and', 'or', 'not',
                      'condexpr', 'call', 'delegatecall', 'send', 'transfer', 'exprnil', 'pnil', 'pcons', 'view',
                      'visibility', 'payable', 'title', 'this', 'solidity', 'uint', 'function', 'meth', 'value',
                      'return',
                      'returns', 'castinterf', 'interface', 'declaration', 'if', 'else', 'ifelse', 'assigmenet',
                      'sequence', 'any_funct', 'contract_ast', 'cast', 'addr', 'address', 'mapping', 'dev', 'event',
                      'require', 'struct', 'constant', 'constructor', 'param', 'public', 'private', 'contract',
                      'modifier', 'msg', 'true', 'false', 'pure', 'delete', 'payable', 'throw',
                      'memory', 'new', 'internal', 'ether', 'arg', 'args', 'dynargs', 'timestamp', 'datasource',
                      'gaslimit', 'library', 'id', 'using', 'encode', 'external', 'buffer', 'safemath', 'revert',
                      'emit',
                      'assert', 'callcode', 'error']

english_stopwords.extend(solidity_stopwords)


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

        yield (preprocess(str(sent), deacc=True))  # deacc=True removes punctuations

    # python3 -m spacy download en_core_web_sm


def process_words(texts,
                  stop_words=english_stopwords,
                  allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'],
                  bigram_mod=None,
                  trigram_mod=None):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in preprocess(str(doc)) if word.lower() not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    # texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load("en_core_web_sm", disable=['parser',
                                                'ner'])  # keeping only tagger component (for efficiency). spacy.load('en', disable=['parser', 'ner', 'textcat'])
    nlp.max_length = 10000000
    contracts = []
    for doc in texts:
        # print("sent: " ,sent)
        sent = []
        for token in doc:
            sent.append([word.lower() for word in camel_case_split(token) if word.lower() not in english_stopwords])
        contracts.append(sent)

    result = list_of_contracts(contracts)
    texts_out = []
    for docum in result:
        document = nlp("".join(str(docum)))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in document if
                                   token.pos_ in allowed_postags]))
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in preprocess(str(doc)) if word.lower() not in english_stopwords] for doc in texts_out]
    return texts_out


def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        if len(row_list) == 0:
            continue
        row = row_list[0] if ldamodel.per_word_topics else row_list
        if isinstance(row, tuple):
            row = [row]
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


def topics_document_words_freq_plot(df_dominant_topic):
    cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    fig, axes = plt.subplots(3, 5, figsize=(16, 14), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins=1000, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 1000), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i], fontdict=dict(size=5))
        ax.set_title('Topic: ' + str(i), fontdict=dict(size=7, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0, 1000, 9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=18)
    plt.show()
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0, 1000, 9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=18)
    plt.show()


def visualize_topics(lda_model, corpus):
    vis = prepare(lda_model, corpus, dictionary=lda_model.id2word, mds='mmds')
    pyLDAvis.save_html(vis, './report/topic_modeling_visualization.html')


def show_topic_clusters(lda_model, corpus, n_topics=15):
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list[0]])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values
    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    # t-distributed Stochastic Neighbor Embedding
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    output_notebook()
    file_name = 'report/topic_modeling_clusters.html'
    output_file(file_name)

    mycolors = np.array([color for name, color in mcolors.XKCD_COLORS.items()])
    plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    save(plot)


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word,
                                                random_state=100, chunksize=1000, update_every=500, iterations=400, passes=20,
                                                alpha='auto', per_word_topics=True, eval_every=None)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


if __name__ == '__main__':
    n_topics = 14
    # n_docs = 1000
    # no_features = 1000
    print("Preprocessing...")
    # data_words = smart_contract_w() #list(sent_to_words(documents))[:n_docs]
    data = [str(nltk.Text(contract)) for contract in contracts]

    data_words = list(sent_to_words(data))

    # # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=20, threshold=100)  # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_ready = process_words(
        data_words,
        bigram_mod=bigram_mod,
        # trigram_mod=trigram_mod
    )

    # print(data_ready)
    # processed Text Data!
    # Create Dictionary
    id2word = corpora.Dictionary(data_ready)
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]
    # Build LDA model
    print("building LDA model...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=n_topics,
                                                random_state=100,
                                                chunksize=1000,
                                                update_every= 500,
                                                passes=20,
                                                alpha='auto',
                                                iterations=400,
                                                per_word_topics=True)

    pprint(lda_model.print_topics())
    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

    ## Model performance
    print("\nModel performance\n")

    ## Perplexity
    print(f"""Perplexity: {lda_model.log_perplexity(corpus)}. Lower is better.
        See https://en.wikipedia.org/wiki/Perplexity.
        The best number of topics minimize perplexity.
        """)

    # Coherence
    coherence = CoherenceModel(
        model=lda_model,
        texts=data_ready,
        dictionary=id2word,
        coherence='c_v'
    )
    # # Corpus coherence
    print(f'Whole model coherence: {coherence.get_coherence()}.')

    # # By topic coherence
    # topic_coherences = coherence.get_coherence_per_topic()
    # print(f"""
    # By topic coherence. Higher is better.
    #     Measure how "well related" are the top words within the same topic.
    #     """)
    #
    # print(f'topic_id | {"top 3 keywords".rjust(45)} | topic coherence')
    # for topic_id in range(n_topics):
    #     words_proba = lda_model.show_topic(topic_id, topn=3)
    #     words = [words for words, proba in words_proba]
    #     print(f'{topic_id:>8} | {str(words).rjust(45)} | {topic_coherences[topic_id]:>8.4f}')

    # Can take a long time to run.
    print("computing coherence values for different number of topics...")
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_ready,
                                                            start=3, limit=40, step=6)

    # Show graph
    limit = 40;
    start = 3;
    step = 6;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    #Print the coherence scores
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    # Select the model and print the topics
    optimal_model = model_list[3]
    model_topics = optimal_model.show_topics(formatted=False)
    pprint(optimal_model.print_topics(num_words=10))

    lda_model= optimal_model




    # Format
    print("dominant topic for each document: ")
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    pd.DataFrame(df_dominant_topic.head(20)).to_csv('report/dominant_topic_for_each_document.csv')

    doc_lens = [len(d) for d in df_dominant_topic.Text]

    # Plot
    plt.figure(figsize=(16, 7), dpi=160)
    plt.hist(doc_lens, bins=1000, color='navy')
    plt.text(750, 7000, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(750, 6000, "Median : " + str(round(np.median(doc_lens))))
    plt.text(750, 4000, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(750, 3000, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(750, 2000, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 1000, 9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    plt.show()

    # Wordcloud of Top N words in each topic

    cols = [color for name, color in mcolors.XKCD_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    cloud = WordCloud(stopwords=english_stopwords,
                      background_color='black',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False, num_topics=15)

    fig, axes = plt.subplots(3, 5, figsize=(16, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    # Topic frequency plot
    topics_document_words_freq_plot(df_dominant_topic)

    # Visualize HTML reports of topics and topic clusters
    show_topic_clusters(lda_model, corpus, n_topics=n_topics)

    visualize_topics(lda_model, corpus)

    topics = lda_model.show_topics(formatted=False, num_topics=15)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(5, 3, figsize=(16, 10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.XKCD_COLORS.items()]

    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i], fontdict=dict(size=5))
        ax_twin.set_ylim(0, 0.030);
        ax.set_ylim(0, 500000)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=11)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right',
                           fontdict=dict(size=5))
        ax.legend(loc='upper left');
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=0.02)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=18, y=1.05)
    plt.show()


    # Group top sentences under each topic
    sent_topics_sorteddf = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf,
                                          grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                         axis=0)

    # Reset Index
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    # Show
    sent_topics_sorteddf.head(30).to_csv('report/most_rappresentative_document_per_topic.csv')
