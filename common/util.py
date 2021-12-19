# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

from common.np import *
from tqdm import tqdm


def to_cpu(x):
    import numpy

    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy

    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


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
    for idx, word_id in tqdm(enumerate(corpus)):
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


def most_similar(
    query: str,
    word_to_id: dict,
    id_to_word: dict,
    word_matrix: np.ndarray,
    top: int = 5,
):
    """[query]와 가장 유사한 상위 top개의 단어 출력

    Args:
        query (str): 질의할 단어
        word_to_id (dict): 단어-정수 사전
        id_to_word (dict): 정수-단어 사전
        word_matrix (np.ndarray): 단어 동시발생행렬
        top (int, optional): 상위 몇 개의 단어를 출력할지 설정. Defaults to 5.
    """
    # 검색어 추출
    if query not in word_to_id:
        print(f"{query}를 찾을 수 없습니다.")
        return
    print(f"\n[query] {query}")
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f" {id_to_word[i]}: {similarity[i]}")

        count += 1
        if count >= top:
            return


def ppmi(C: np.ndarray, verbose=False, eps=1e-8):
    """양의 정보상호량 계산 함수

    빈도수를 이용하는 방식에서는 the와 같은 관사가 최다 빈도이기 때문에 높은 유사도를 가지는 경우가 많다.
    이런 상황을 방지하기 위해서 단어가 단독으로 출현하는 횟수를 고려한 pmi 지수를 사용한다.
    이때 pmi지수는 동시발생 횟수가 0일 때 음의 무한대 값을 가지기 때문에 max(0, pmi)를 한 ppmi를 사용한다.

    Args:
        C (np.ndarray): 동시발생행렬
        verbose (bool, optional): 출력문을 자세히 볼 것인지 설정. Defaults to False.
        eps ([type], optional): 0으로 나누기 에러 방지. Defaults to 1e-8.

    Returns:
        np.ndarray: 입력 동시발생행렬과 같은 크기의 ppmi 행렬
    """
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]

    pbar = tqdm(total=total)
    pbar.set_description(f"PPMI 계산: ")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / S[j] * S[i] + eps)
            M[i, j] = max(0, pmi)
            if verbose:
                pbar.update(1)
    pbar.close()
    return M


def create_contexts_target(corpus: np.ndarray, window_size=1):
    """말뭉치를 맥락 리스트와 타겟 리스트로 변환

    Args:
        corpus (np.ndarray): 정수 인코딩 된 말뭉치
        window_size (int, optional): 주변의 몇 개 단어를 맥락으로 볼 것인지 설정. Defaults to 1.

    Returns:
        np.ndarray: contexts
        np.ndarray: target
    """
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


def convert_one_hot(corpus: np.ndarray, vocab_size: int):
    """정수 인코딩된 말 뭉치를 원 핫 인코딩하기

    Args:
        corpus (np.ndarray): 정수 인코딩된 말뭉치
        vocab_size (int): 단어 개수

    Returns:
        np.ndarray: 원 핫 인코딩된 말뭉치
    """
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot
