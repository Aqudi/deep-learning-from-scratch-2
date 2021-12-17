# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)
print("동시발생 수 계산 ...")
C = create_co_matrix(corpus, vocab_size, window_size)
W = ppmi(C, verbose=True)

print("SVD 계산 ...")
try:
    # truncated SVD (더 빠름)
    from sklearn.utils.extmath import randomized_svd

    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=1)
except ImportError:
    # Full SVD (느림)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
