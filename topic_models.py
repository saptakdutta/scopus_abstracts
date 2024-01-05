# %%Libs
import pandas as pd

pd.options.mode.chained_assignment = None
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    TfidfTransformer,
)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import AffinityPropagation
import gensim
import gensim.corpora as corpora
import re
import numpy as np
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# download the punkt tokenizer model from nltk for word tokenizing, and wordnet
nltk.download("punkt")
nltk.download("wordnet")

# %%import dataset
papers = pd.read_csv("relevant_publications.csv")
print(papers.shape)
papers.head()

# %%Data preprocessing
# remove symbols
papers["Abstract_Cleaned"] = papers.apply(
    lambda row: (re.sub("[^A-Za-z0-9' ]+", " ", row["description"])), axis=1
)
# lowercase
papers["Abstract_Cleaned"] = papers["Abstract_Cleaned"].map(lambda x: x.lower())
# tokenization
papers["Abstract_Cleaned"] = papers.apply(
    lambda row: (word_tokenize(row["Abstract_Cleaned"])), axis=1
)
# stop word removal
extra_stopwords = ["due"]
stop_words = set(stopwords.words("english") + extra_stopwords)
papers["Abstract_Cleaned"] = papers.apply(
    lambda row: ([w for w in row["Abstract_Cleaned"] if w not in stop_words]), axis=1
)
# Lemmatization
lmtzr = WordNetLemmatizer()
papers["Abstract_Cleaned"] = papers.apply(
    lambda row: ([lmtzr.lemmatize(w) for w in row["Abstract_Cleaned"]]), axis=1
)
# view sample
papers["Abstract_Cleaned"][1][:20]

# %% Use affinity propogation to find the likely number of topics
abstracts = papers["description"].values
count_vectorizer = CountVectorizer()
counts = count_vectorizer.fit_transform(abstracts)
tfidf_vectorizer = TfidfTransformer().fit(counts)
tfidf_abstracts = tfidf_vectorizer.transform(counts)
print(tfidf_abstracts.shape)
X = tfidf_abstracts
clustering = AffinityPropagation().fit(X)
clustering
abstract_affinity_clusters = list(clustering.labels_)
print("likely number of topics is: {}".format(len(set(abstract_affinity_clusters))))

# %% Building an LDA topic model
dictionary = corpora.Dictionary(papers["Abstract_Cleaned"])
texts = papers["Abstract_Cleaned"]
corpus = [dictionary.doc2bow(text) for text in papers["Abstract_Cleaned"]]

# %% Building LDA Model
lda_model_gensim = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=17,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    per_word_topics=True,
)

pprint(lda_model_gensim.print_topics())
doc_lda = lda_model_gensim[corpus]

# %% Visualize
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(
    lda_model_gensim, corpus, dictionary, R=10, sort_topics=False
)
pyLDAvis.save_html(vis, "lda_result.html")

# %% you can show the topics like so: the graph will always show topic+1
lda_model_gensim.show_topic(16)

# %% Do the LDA now with sklearn
lda_model_sklearn = LatentDirichletAllocation(
    n_components=17,
    max_iter=10,
    learning_method="online",
    learning_offset=50.0,
    random_state=0,
).fit(tfidf_abstracts)
lda_W = lda_model_sklearn.transform(tfidf_abstracts)
lda_H = lda_model_sklearn.components_


def display_topics(H, W, feature_names, title_list, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("\n", "Topic %d:" % (topic_idx))
        print(
            "Top Words: ",
            " ".join(
                [feature_names[i] for i in topic.argsort()[: -no_top_words - 1 : -1]]
            ),
        )
        top_doc_indices = np.argsort(W[:, topic_idx])[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(title_list[doc_index])


no_top_words = 15
no_top_documents = 4
title_list = papers["title"].tolist()
tf_feature_names = count_vectorizer.get_feature_names_out()
display_topics(
    lda_H, lda_W, tf_feature_names, title_list, no_top_words, no_top_documents
)

# %% Now classify the documents according to their topics and weed out the useless ones
# example for one abstract
lda_model_gensim.get_document_topics(dictionary.doc2bow(papers["Abstract_Cleaned"][0]))
# now do all of them
# create a column called ldatopics
papers["ldatopic"] = 0
for j in np.arange(0, len(papers)):
    topic = lda_model_gensim.get_document_topics(
        dictionary.doc2bow(papers["Abstract_Cleaned"][j])
    )[0][0]
    papers["ldatopic"][j] = topic

# %% Now look at all papers under a certain topic... e.g,
papers["title"][papers["ldatopic"] == 2]

# %% check how the universitites fare under the topics
# Technical University Munich
papers_TUM = papers[papers["affilname"] == "Technical University Munich"]
fig = papers_TUM["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Technical University Munich")
fig.figure.savefig("Figs/TUM.jpg")

# Karlsruhe Institute of Technology (KIT)
papers_KIT = papers[papers["affilname"] == "Karlsruhe Institute of Technology (KIT)"]
fig = papers_KIT["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Karlsruhe Institute of Technology")
fig.figure.savefig("Figs/KIT.jpg")

# University of Stuttgard
papers_UoS = papers[papers["affilname"] == "University of Stuttgard"]
fig = papers_UoS["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("University of Stuttgard")
fig.figure.savefig("Figs/UoS.jpg")

# RWTH Aachen University
papers_AU = papers[papers["affilname"] == "RWTH Aachen University"]
fig = papers_AU["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("RWTH Aachen University")
fig.figure.savefig("Figs/AU.jpg")

# Technical University Berlin
papers_TUB = papers[papers["affilname"] == "Technical University Berlin"]
fig = papers_TUB["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Technical University Berlin")
fig.figure.savefig("Figs/TUB.jpg")

# Technical University Dresden
papers_TUDR = papers[papers["affilname"] == "Technical University Dresden"]
fig = papers_TUDR["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Technical University Dresden")
fig.figure.savefig("Figs/TUDR.jpg")

# Technical University Darmstadt
papers_TUDA = papers[papers["affilname"] == "Technical University Darmstadt"]
fig = papers_TUDA["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Technical University Darmstadt")
fig.figure.savefig("Figs/TUDA.jpg")

# Technical University Braunschweig
papers_TUBR = papers[papers["affilname"] == "Technical University Braunschweig"]
fig = papers_TUBR["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Technical University Braunschweig")
fig.figure.savefig("Figs/TUBR.jpg")

# University of Kassel
papers_UoK = papers[papers["affilname"] == "University of Kassel"]
fig = papers_UoK["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("University of Kassel")
fig.figure.savefig("Figs/UoK.jpg")

# Ruhr University Bochum
papers_RUB = papers[papers["affilname"] == "Ruhr University Bochum"]
fig = papers_RUB["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Ruhr University Bochum")
fig.figure.savefig("Figs/RUB.jpg")

# Technical University Kaiserslautern
papers_TUK = papers[papers["affilname"] == "Technical University Kaiserslautern"]
fig = papers_TUK["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Technical University Kaiserslautern")
fig.figure.savefig("Figs/TUK.jpg")

# Technical University Dortmund
papers_TUD = papers[papers["affilname"] == "Technical University Dortmund"]
fig = papers_TUD["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Technical University Dortmund")
fig.figure.savefig("Figs/TUD.jpg")

# Bauhaus University Weimar
papers_BUW = papers[papers["affilname"] == "Bauhaus University Weimar"]
fig = papers_BUW["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Bauhaus University Weimar")
fig.figure.savefig("Figs/BUW.jpg")

# Humboldt University
papers_HU = papers[papers["affilname"] == "Humboldt University"]
fig = papers_HU["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Humboldt University")
fig.figure.savefig("Figs/HU.jpg")

# Federal Institute for Materials Research and Testing (BAM)
papers_BAM = papers[
    papers["affilname"] == "Federal Institute for Materials Research and Testing (BAM)"
]
fig = papers_BAM["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("Federal Institute for Materials Research and Testing")
fig.figure.savefig("Figs/BAM.jpg")

# University of Hannover
papers_HAN = papers[papers["affilname"] == "University of Hannover"]
fig = papers_HAN["ldatopic"].value_counts().plot(kind="bar")
plt.ylabel("Number of occurences")
plt.title("University of Hannover")
fig.figure.savefig("Figs/HAN.jpg")


# %%Check the authors
