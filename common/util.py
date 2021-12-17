import numpy as np


def clip_grads():
    pass


def preprocess(text: str):
    """text를 정수 인코딩

    Args:
        text (str): text

    Returns:
        corpus (np.ndarray): 정수 인코딩한 text
        word_to_id (dict): 단어를 정수로 변환
        id_to_word (dict): 정수를 단어로 변환
    """
    text = text.lower().replace(".", " .")
    words = text.split(" ")
    # word-id dictionary
    word_to_id = {}
    # id-word dictionary
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus: np.ndarray, vocab_size: int, window_size: int = 1):
    """입력된 말뭉치로부터 동시발생행렬 생성

    Args:
        corpus (np.ndarray): 정수 인코딩된 말뭉치
        vocab_size (int): 단어 사전 사이즈
        window_size (int, optional): 좌우 몇 개의 단어를 문맥으로 볼 것인지. Defaults to 1.

    Returns:
        np.ndarray: 생성된 동시발생행렬
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    # i번째 단어 주변에 등장한 단어 빈도수 저장
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def cos_similarity(x: np.ndarray, y: np.ndarray, eps=1e-8):
    """두 벡터 사이의 유사도 측정

    eps를 분모에 더해줌으로써 0으로 나누기 오류를 방지한다.
    eps는 매우 작은 값이기 때문에 부동소수점 계산 시에 반올림되어 다른 값에 흡수된다.

    Args:
        x (np.ndarray): 벡터1
        y (np.ndarray): 벡터2
        eps (float, optional): 0으로 나누기 에러 방지. Defaults to 1e-8.

    Returns:
        float: 두 벡터의 유사도
    """
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    return np.dot(nx, ny)
