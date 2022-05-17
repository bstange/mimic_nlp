import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
import umap
import hdbscan

import plotly.express as px

import dataframe_image as dfi


def tokenize_documents(doc_list):
    tokens = []
    for doc in doc_list:
        try:
            if doc == np.nan:
                tokens.append([])
            proc_doc = nlp(doc)
            tokens.append([token.lemma_.lower() for token in proc_doc if
                           not token.is_stop
                           and not token.is_punct
                           and not token.is_digit
                           and not token.is_space
                           and not token.like_num])
        except:
            tokens.append([])

    return [' '.join(i) for i in tokens]


nlp = spacy.load('en_core_web_lg')
split_docs = pd.read_csv('./data/section_embeddings.csv')
# embeddings = np.load('./data/section_embeddings.npy', allow_pickle=True)
mask = ~np.isnan(embeddings).any(axis=1)

split_docs['embeddings'] = embeddings
split_docs = split_docs[mask]


selected_docs = split_docs[split_docs['section'].isin(['procedure', 'hpi', 'course', 'discharge_diag'])].copy()
selected_docs['text'].fillna('', inplace=True)

pat_selected_docs = pd.DataFrame(selected_docs.groupby('id')['text'].apply(' '.join))
pat_selected_embeddings = selected_docs.groupby('id')['embeddings'].apply(np.mean).to_list()

pat_selected_embeddings = np.stack(pat_selected_embeddings, axis=1)

pat_selected_docs['id'] = pat_selected_docs.index
selected_doc_text = pat_selected_docs['text'].to_list()

tokens = tokenize_documents(selected_doc_text)

tfidf = TfidfVectorizer(max_df=.2, min_df=1)
vec_docs = tfidf.fit_transform(tokens)
vocab = tfidf.get_feature_names_out()

lda = LatentDirichletAllocation(n_components=30, max_iter=30)
doc_topics = lda.fit_transform(vec_docs)

components = lda.components_
doc_label_index = np.argmax(doc_topics, axis=1)
label_counts = pd.DataFrame(np.unique(doc_label_index, return_counts=True)).T
label_counts.columns = ['index', 'counts']


def generate_labels(vocab, components):
    labels = {}
    for index, component in enumerate(components):
        z = zip(vocab, component)
        top_terms_key = sorted(z, key=lambda t: t[1], reverse=True)[:4]
        top_terms_list = list(dict(top_terms_key).keys())
        print("Topic " + str(index) + ": ", top_terms_list)
        labels[index] = ', '.join(top_terms_list)

    return labels, [labels[i] for i in doc_label_index]


labels, doc_labels = generate_labels(vocab, components)

label_counts.columns = ['index', 'counts']
label_counts['index'] = label_counts['index'].apply(lambda x: labels[x])
label_counts.sort_values(by='counts', ascending=False, inplace=True)
dfi.export(label_counts, '.\images\lda_label_counts.png')


reducer = umap.UMAP(n_neighbors=20, min_dist=.6, n_components=2)
embed_reducer = umap.UMAP(n_neighbors=20, min_dist=.6, n_components=2)

compressed_embeddings = embed_reducer.fit_transform(pat_selected_embeddings)
compressed_embeddings = normalize(compressed_embeddings, axis=0)

embed_cluster = SpectralClustering(n_clusters=10).fit(pat_selected_embeddings)

embed_label_counts = pd.DataFrame(np.unique(embed_cluster.labels_, return_counts=True)).T
embed_label_counts.columns = ['index', 'counts']
embed_label_counts.sort_values(by='counts', ascending=False, inplace=True)
dfi.export(embed_label_counts, '.\images\embedding_label_counts.png')

compressed_vec_docs = reducer.fit_transform(doc_topics)
compressed_vec_docs = normalize(compressed_vec_docs, axis=0)

fig = px.scatter(x=compressed_vec_docs[:, 0], y=compressed_vec_docs[:, 1], color=doc_labels)
fig.write_image('./images/lda_cluster_scatter.png')

fig = px.scatter(x=compressed_embeddings[:, 0], y=compressed_embeddings[:, 1], color=embed_cluster.labels_.astype(str))
fig.write_image('./images/embedding_cluster_scatter.png')


if __name__ == '__main__':
    pass
