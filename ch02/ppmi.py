# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import numpy as np
from common.util import ppmi, preprocess, create_co_matrix

text = "You say goodby and I say hello."
# 정수 인코딩
corpus, word_to_id, id_to_word = preprocess(text)
# 동시발생행렬로 만들기
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# 유효 자릿수를 세 자리까지 표시
np.set_printoptions(precision=3)
print("동시발생행렬")
print(C)
print("-" * 50)
print("PPMI")
print(W)
