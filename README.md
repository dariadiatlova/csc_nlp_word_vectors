### Third hometask for CSC NLP fall 2021 course.

[Kaggle competition link](https://www.kaggle.com/c/a-pack-of-chips-and-the-company-it-keeps/overview)


Task overview: predict a binary value for a pair of Russian words, where one means words similarity. 


Train data: 1M sentences.


Approach: index all words in a dictionary, filter rare and popular words from the corpus. Compute PMI matrix for the corpus with the winow size 3. Reduce matrix dimension via SVD truncation. Compute cosine distance for the pair of word embeddings from test dataset. Filter the pairs by similarity score: consider all pairs with cosine distance < 0.35 similar. 

[Hard-coded solution with rus comments](csc_iinlp_hw04.py) 


### Fourth hometask for CSC NLP fall 2021 course.

[Kaggle competition link](https://www.kaggle.com/c/text-clustering)


Task overview: predict a binary value for a pair of texts in russian, where one means words similarity. 


Data: unlabled ~1.3К pairs.


Approach: use LDA for topic modeling and finding out the number of clusters. Vectorize texts using Tf-Idf feutures, drop stopwords and terms that meet less than in 1% of all texts or more than in 75% of texts. Use agglomerative clustering with cosine distance and label pairs according to the cluster ids.  
