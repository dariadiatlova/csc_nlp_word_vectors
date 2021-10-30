# coding: utf-8
"""
    Using nothing but
        - Python built-in libraries,
        - sklearn.feature_extraction.text,
        - sklearn.model_selection,
        - sklearn.decomposition,
        - numpy or
        - scipy,

    prepare a matrix of word vectors, that would allow to achieve
    certain results in the task of word similarity evaluation.

    Libs such as sklearn (except for those modules that are listed above),
    pytorch, etc. or any tools automating standard training routines are NOT allowed.

    Again: SVD and NMF from numpy/scipy/sklearn are very much welcome.
    ML models and tools for training word representations (such as gensim, glove, word2vec,
    etc.) are NOT ALLOWED.

    Data for preparing the model: a part of Araneum Russicum provided in the task.
    Tuning your model using the ORIGINAL evaluation dataset used in the task is prohibited.

    CSC-NLP-2021
"""
from collections import defaultdict, Counter
from functools import lru_cache
from typing import List, Iterable, Dict, Callable

import logging
import numpy as np
from scipy import sparse as sp
from tqdm import tqdm

from sklearn.decomposition import TruncatedSVD

log = logging.getLogger()


class Sentences(object):
    """ Итератор по строкам заданных файлов, не загружающий их полностью в RAM """

    def __init__(self, filenames: Iterable[str]):
        self.filenames: Iterable = filenames

    def __iter__(self):
        for filename in self.filenames:
            for line in open(filename, "r+", encoding="utf-8"):
                # возвращает лист из строк, где каждая строка – слово из строки файла
                yield line.strip().replace("ё", "е").split(" ")


class Embeddings(object):
    """
        Класс, строящий векторные представления для каждого из интересующих нас термов
        по набору предобработанных текстов, а также вычисляющий меру их сходства
        на основе косинусного расстояния, и задающий порог для него, говорящий, похожи два данных
        терма или нет (`are_similar`).
    """

    def __init__(self,
                 threshold: float = 0.2,
                 window_half_size: int = 2,
                 min_df: int = 5,
                 max_df: int = 1000000):
        """
            @param: threshold Порог для отсечения по косинусному расстоянию
            @param: window_half_size Половинка "окна", то есть макс. расстояние от центра окна
            @param: min_df Если частота терма ниже этого порога -- выбрасываем
            @param: max_df Если частота терма выше этого порога -- выбрасываем
        """
        self.th, self.whs = threshold, window_half_size
        self.M, self.counts = None, None
        self.min_df, self.max_df = min_df, max_df

        # список термов
        self.id2token: List[str] = []
        # словарь индексов термов
        self.token2id: Dict[str, int] = {}

    @lru_cache(maxsize=100000000)
    def _idx(self, token: str):
        """ Кэширующий метод, получающий индекс терма """

        if token not in self.token2id:
            curr_idx = len(self.id2token)
            self.token2id[token] = curr_idx
            self.id2token.append(token)
        else:
            curr_idx = self.token2id[token]
        return curr_idx

    def _reset(self):
        """ Сброс всего 'выученного' (но не гиперпараметров) """
        self.M, self.counts, self.token2id, self.id2token = None, None, {}, []

    def fit(self, text_reader: Callable):
        """
            Построение матрицы с векторными представлениями
            на основе поданных предобработанных предложений.
            YOUR CODE HERE, так сказать.
        """
        self._reset()
        word_counter = Counter()
        corpus_sent_size = 0

        for sentence in tqdm(text_reader(), "sentences, pre-fit"):
            for c in range(len(sentence)):
                word_counter[sentence[c]] += 1
            corpus_sent_size += 1

        for_exclusion = []

        for key in tqdm(word_counter.keys(), "bad guys"):
            if word_counter[key] < self.min_df or word_counter[key] > self.max_df:
                for_exclusion.append(key)

        for_exclusion = set(for_exclusion)
        log.debug(f"For exclusion, terms: {len(for_exclusion)}.")

        # счётчик, номер строки, номер столбца
        values, fromm, too = [], [], []

        # общее число -- может пригодиться для нормализации и др.
        total_counts = defaultdict(lambda: 0)

        for sentence in tqdm(text_reader(), "sentences", corpus_sent_size):

            # проходим центром окна по предложению
            for c in range(len(sentence)):

                if sentence[c] in for_exclusion:
                    continue

                current_idx = self._idx(sentence[c])
                total_counts[current_idx] += 1

                # TODO: проходим for вдоль окна индексом idx и делаем, что должно
                # TODO: ...не забывая о for_exclusion, если будем использовать
                window = np.arange(c - self.whs, c + self.whs + 1)
                for idx in window:
                    if idx < 0 or idx >= len(sentence) or idx == c or sentence[idx] in for_exclusion:
                        continue

                    values.append(1)
                    fromm.append(current_idx)
                    too.append(self._idx(sentence[idx]))

        # матрица встречаемости слов в контексте друг с другом
        self.M = sp.coo_matrix((values, (fromm, too)))
        self.M.sum_duplicates()
        self.M = self.M.tocsr()

        # укладываем общую встречаемость термов в массив
        # TODO: как бы нам их использовать?
        self.counts = sp.csr_matrix(np.array([total_counts[i] for i in range(len(self.id2token))]))

        all_word_counts = np.sum(self.counts.toarray()[0])
        probabilities = self.counts.toarray()[0] / all_word_counts
        count_matrix = self.M.toarray()
        self.pmi_matrix = np.zeros((len(probabilities), len(probabilities)), dtype=np.float32)

        probability_matrix = probabilities @ probabilities.T
        probability_matrix = np.where(probability_matrix <= 0, 1, probability_matrix)
        pmi_matrix = count_matrix / probability_matrix
        pmi_matrix = np.where(pmi_matrix <= 0, 0, np.log2(pmi_matrix))

        svd = TruncatedSVD(n_components=64, n_iter=8, random_state=8)
        self.M = svd.fit_transform(pmi_matrix)
        log.debug(f"A total of {len(self.id2token)} terms.")
        log.debug(f"Matrix of size {self.M.shape} was built.")

        # nonzero_size = len(self.M.nonzero()[0])
        # log.debug(f"Sparsity: %.2f nonzero values: a total of %d." % \
        #           (100 * nonzero_size / self.M.shape[0] / self.M.shape[1], nonzero_size))
        return None

    def score(self, word1: str, word2: str):
        """ Косинусное расстояние между векторами заданных термов """

        if self.M is None:
            raise Exception("The model has not been trained yet.")

        word1, word2 = word1.replace("ё", "е"), word2.replace("ё", "е")
        id1, id2 = self.token2id[word1], self.token2id[word2]
        v1, v2 = self.M[id1, :], self.M[id2, :]
        return 1 - v1.dot(v2.T) / np.sqrt(np.sum(v1 * v1)) / np.sqrt(np.sum(v2 * v2))

    def are_similar(self, word1: str, word2: str):
        """ Если расстояние ниже порога, то считаем близкими """
        if self.M is None:
            raise Exception("The model has not been trained yet.")

        return self.score(word1, word2) < self.th

    def get(self, word: str):

        word = word.replace("ё", "е")

        if self.M is None:
            raise Exception("The model has not been trained yet.")

        if word in self.token2id:
            return self.M[self.token2id[word]]
        else:
            return None


if __name__ == "__main__":
    import pandas as pd

    logging.basicConfig(level=logging.DEBUG)
    root = logging.getLogger()

    sentences = lambda: Sentences(filenames=["data/rus_araneum_maxicum-1M_lines.txt"])

    whs, th, mindf = 3, 0.35, 10
    embs = Embeddings(window_half_size=whs, threshold=th, min_df=mindf)
    embs.fit(sentences)

    df = pd.read_csv("data/test.csv")
    df["word1"] = df["word1"].map(lambda x: x + "_NOUN")
    df["word2"] = df["word2"].map(lambda x: x + "_NOUN")

    results = {"id": list(df["id"]), "sim": []}
    skipped = 0

    for _, row in tqdm(df.iterrows(), "predictions", df.shape[0]):
        try:
            if embs.are_similar(row["word1"], row["word2"]):
                results["sim"].append(1)
            else:
                results["sim"].append(0)
        except KeyError as ke:
            skipped += 1
            results["sim"].append(0)

    log.debug(f"{skipped} pairs not found")
    pd.DataFrame(results).to_csv(f"submissions/{whs}_th{th}_mindf{mindf}_csc{10}.csv", index=None)
