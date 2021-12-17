# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
from common.util import preprocess, create_co_matrix, cos_similarity

text = "You say goodby and I say hello."
# 정수 인코딩
corpus, word_to_id, id_to_word = preprocess(text)
# 동시발생행렬로 만들기
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(C)
# 단어 "you"와 "i"의 코사인 유사도 계산
c0 = C[word_to_id["you"]]
print(c0)
c1 = C[word_to_id["i"]]
print(c1)
print(cos_similarity(c0, c1))
