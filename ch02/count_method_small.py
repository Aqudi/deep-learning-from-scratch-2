# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from common.util import ppmi, preprocess, create_co_matrix

text = "You say goodby and I say hello."
# 정수 인코딩
corpus, word_to_id, id_to_word = preprocess(text)
# 동시발생행렬로 만들기
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD를 이용한 차원 감소
U, S, V = np.linalg.svd(W)

print(C[0])
# 희소벡터인 W
print(W[0])
# SVD를 통해서 밀집벡터인 U로 변환됨
print(U[0])
# 2차원으로 차원 감소
print(U[0, :2])

# 단어 벡터를 그래프로 표현
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
