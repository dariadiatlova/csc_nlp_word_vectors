import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from operator import itemgetter
from nltk.corpus import stopwords


class Solution:
    def __init__(self):
        self.pairs_df = pd.read_csv('pairs.csv')
        self.texts_df = pd.read_csv('texts.csv')
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=stopwords.words("russian"),
                                          min_df=0.01, max_df=0.7)
        self.svd = TruncatedSVD(n_components=64, n_iter=8, random_state=9)

    def _fit_tfidf(self):
        tfidf = self.vectorizer.fit_transform(np.array(self.texts_df.paragraph))
        return tfidf

    def _svd_truncation(self):
        tfidf = self._fit_tfidf()
        truncated_tfidf = self.svd.fit_transform(tfidf.toarray())
        return truncated_tfidf

    def _agglomerative_clustering(self) -> dict:
        truncated_tfidf = self._svd_truncation()
        cluster_algorithm = AgglomerativeClustering(
            n_clusters=5, affinity='cosine', linkage='average').fit(truncated_tfidf)
        predictions = cluster_algorithm.labels_
        prediction_dictionary = dict(zip(range(1, len(predictions) + 1), predictions))
        return prediction_dictionary

    def _label_pairs(self):
        predictions = self._agglomerative_clustering()
        column_one_idx = itemgetter(*list(self.pairs_df.one))(predictions)
        column_two_idx = itemgetter(*list(self.pairs_df.two))(predictions)
        comparison = np.array(column_one_idx) == np.array(column_two_idx)
        result_df = pd.DataFrame({"id": np.arange(len(comparison)), "gold": comparison.astype(int)})
        result_df.to_csv(
            "/Users/diat.lov/GitHub/text_clustering/submissions/tfidf_svd_agglo_5_cos_maxdf07_mindf001.csv", index=False
        )

    def main(self) -> None:
        self._label_pairs()


if __name__ == "__main__":
    solution = Solution()
    solution.main()
