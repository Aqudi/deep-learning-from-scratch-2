# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
from common.util import most_similar, preprocess, create_co_matrix, cos_similarity

text = "You say goodby and I say hello."
# 정수 인코딩
corpus, word_to_id, id_to_word = preprocess(text)
# 동시발생행렬로 만들기
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(C)

# you와 가장 비슷한 단어 상위 5개 출력
most_similar("you", word_to_id, id_to_word, C, top=5)
