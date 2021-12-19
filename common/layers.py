# -*- coding: utf-8 -*-
"""layers.py

공통 변수
    params: 가중치 리스트
    grads: 기울기 리스트

공통 메소드
    forward: 순전파
    backward: 역전파
"""
from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import cross_entropy_error, softmax


class MatMul:
    """행렬곱 노드 층

    곱셈 노드의 역전파: 순전파 시의 입력을 서로 바꿔 기울기와 곱해준다.
        e.g z = x X y 일 때 dx/dz = y, dz/dy = x 이다.
            dL/dx = dL/dx dx/dz = dL/dx y
            dL/dy = dL/dy dy/dz = dL/dy x

    위를 행렬로 확장할 수도 있다.
    dL/dx = dL/dy W^T
    dL/dW = x^T dL/dy
    """

    def __init__(self, W) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        (W,) = self.params
        out = np.matmul(x, W)
        self.x = x
        return out

    def backward(self, dout):
        """dx = 기울기 * W^T

        dW = x^T * 기울기
        """
        (W,) = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Sigmoid:
    """Sigmoid 활성화 함수 층

    Sigmoid 함수의 미분 구하는 방법
    참고: 책의 부록 A.1
    참고: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e

    backward: sigmoid 함수의 미분 * 기울기 = y(1-y) * 기울기
    """

    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        """dx = y(1-y) X dout"""
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    """Affine 층 (완전연결계층)

    완전연결계층에 의한 변환은 기하학에서 affine변환에 해당
    """

    def __init__(self, W, b) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        # 편향 b를 더할 때 numpy의 broadcast 기능이 적용됨
        # 이는 계산 그래프에서 Repeat 노드에 해당함 (Sum 노드로 역전파)
        out = np.matmul(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        # Repeat 노드의 역전파 -> Sum 노드
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class SoftmaxWithLoss:
    """SoftmaxWithLoss 층

    Softmax - Cross Entropy Error - L

    Sigmoid 함수의 출력이
    """

    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.y = None  # softmax 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


class SigmoidWithLoss:
    """SodmoidWithLoss 층

    Sodmoid - Cross Entropy Error - L

    Sigmoid 함수의 출력이
    """

    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dout / batch_size
        return dx


class Embedding:
    """Embedding 층

    One-hot 벡터와 가중치 행렬의 곱의 연산량을 줄이기 위하여
    특정 단어의 분산 표현을 추출해주는 층

    메모리 사용량과 불필요한 계산을 줄일 수 있다.
    """

    def __init__(self, W) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        (W,) = self.params
        if GPU:
            idx = idx.get()
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        (dW,) = self.grads
        dW[...] = 0

        # for i, word_id in enumerate(self.idx):
        #     dW[word_id] += dout[i]
        if GPU:
            import cupyx

            cupyx.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None
