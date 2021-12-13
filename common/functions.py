# -*- coding: utf-8 -*-
import numpy as np


def softmax(x):
    """Softmax 함수

    y_k = exp(s_k) / Sum(exp(s_i))
    """
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    """CrossEntropy 오차 함수

    L = - Sum(t_k, log(y_k))

    N개의 데이터를 포함한 미니배치를 고려한 식

    L = - (1/N) Sum( Sum(t_nk * log(y_nk)) )
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if t.size == y.size:
        # 각 클래스 별 확률 중 가장 큰 값의 인덱스를 정답 레이블 인덱스로 사용
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
