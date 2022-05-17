import pandas as pd
import numpy as np
import spacy
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer


def embed_text(spacy_model, embedding_model, text):
    embeddings = []
    nan_array = np.empty(768,)
    nan_array[:] = np.NaN
    for _, i in enumerate(text):
        print(i)
        try:
            if i == np.nan:
                embeddings.append(nan_array)
            else:
                # split sentences
                proc_text = spacy_model(i)
                sents = [sent for sent in proc_text.sents]

                # get embeddings
                embedding = np.array([embedding_model.encode(i.text) for i in sents])

                # combine embeddings
                embeddings.append(
                    np.mean(normalize(embedding, axis=1), axis=0)
                )
        except:
            embeddings.append(nan_array)
    return embeddings


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')
    embed_model = SentenceTransformer('bvanaken/CORe-clinical-diagnosis-prediction')

    docs = pd.read_csv('./data/procnotes.csv')
    docs.index = docs['id']
    split_docs = docs.melt(id_vars=['id'], var_name='section', value_name='text')
    text = split_docs['text'].to_list()
    embeddings = embed_text(nlp, embed_model, text)
    np.save('./data/section_embeddings.npy', embeddings, allow_pickle=True)
    split_docs.to_csv('./data/section_df.csv', index=False)
