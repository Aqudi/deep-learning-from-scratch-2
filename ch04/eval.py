# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
from common.util import analogy, most_similar
import pickle

pkl_file = "cbow_params.pkl"

with open(pkl_file, "rb") as f:
    params = pickle.load(f)

    word_vecs = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]

querys = ["you", "year", "car", "toyota"]

# 가장 유사한 단어 상위 N개를 뽑아서 출력
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

analogy("king", "man", "queen", word_to_id, id_to_word, word_vecs)
# 출력: woman (king과 queen이 남성과 여성에 의해 결정)

analogy("take", "took", "go", word_to_id, id_to_word, word_vecs)
# 출력: went (현재형과 과거형 관계)

analogy("car", "cars", "child", word_to_id, id_to_word, word_vecs)
# 출력: children (단수형과 복수형 관계)

analogy("good", "better", "bad", word_to_id, id_to_word, word_vecs)
# 출력: more, less (예상한 worse를 출력하지는 않았지만 비교급이라는 문법적인 의미도 인코딩 되있음을 확인 가능)
 