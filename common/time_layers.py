# -*- coding: utf-8 -*-
"""time_layers.py

시계열 데이터를 다루는 층 모음

공통 변수
    params: 가중치 리스트
    grads: 기울기 리스트

공통 메소드
    forward: 순전파
    backward: 역전파
"""
from common.functions import softmax
from common.layers import Embedding
from common.np import *  # import numpy as np


class RNN:
    """RNN 층

    RNN의 순전파 수식을 그대로 구현 한 것
    .. math::
        h_t = tanh(h_{t-1}W_h + x_tW_x + b)
    """

    def __init__(self, Wx, Wh, b) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        # h_prev: 이전 RNN 계층으로부터 받아온 입력
        Wx, Wh, b = self.params
        # 가중치 Wh, Wx 학습
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # tanh의 미분 = 1 - tanh^2(x)
        dt = dh_next * (1 - h_next ** 2)
        # Repeat의 역전파
        db = np.sum(dt, axis=0)
        # Matmlu의 역전파
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        return dx, dh_prev


class TimeRNN:
    """TimeRNN 층

    T개의 시간을 다루기 위하여 RNN 층 T개를 사용하는 계층
    """

    def __init__(self, Wx, Wh, b, stateful=False) -> None:
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype="f")

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype="f")

        # T개의 RNN 층 쌓기
        for t in range(T):
            # Wx, Wh, b는 RNN 층에서 학습됨
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype="f")
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):
            layer = self.layers[t]
            # 순전파 시 출력은 2개로 분기하기 때문에 역전파 시 합산한 기울기가 전해진다.
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        # TimeRNN 계층 내의 RNN 계층들은 모두 같은 기울기를 사용함
        # 따라서 TimeRNN 계층에서의 기울기는 내부 RNN 계층들의 기울기를 모두 더한 것
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        # Seq2Seq에서 사용
        self.dh = dh

        return dxs


class TimeEmbedding:
    """TimeEmbedding 층 - T개의 Embedding 층을 이용하는 층

    forward 시에는 미니배치내의 N개의 xs를 입력으로 하여 학습함.
    이때 xs는 T개의 각 시간축에 해당하는 입력이 들어있음.

    backward 시에는 forward의 역순으로 Embedding 층의 backward 실행함.
    이때 가중치를 공유했으므로 backward된 가중치들을 모두 합산함.
    """

    def __init__(self, W) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype="f")
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    """TimeAffine 충 - 행렬 계산으로 최적화된 버전

    행렬 계산으로 한 번에 처리하게 되면 T개의 Affine 계층을 이용하는 방식보다 효율적이다.

    - N: 미니배치의 크기
    - T: 벡터의 개수 (trucated 단위)
    - D: 벡터의 차원 (Hidden 노드 개수)
    - V: 어휘의 개수

    (N, T, D) 크기의 행렬을 (N * T, D)로 변환하여 (D, V) 크기의 행렬 W와 계산한다.
    """

    def __init__(self, W, b) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N * T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class TimeSoftmaxWithLoss:
    """TimeSoftmaxWithLoss 층

    T개의 SoftmaxWithLoss 계층의 손실을 합산하여 평균한 최종 손실을 구하는 층
    """

    def __init__(self) -> None:
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        # 정답 레이블이 원합 벡터인 경우 숫자로 바꿔줌
        if ts.ndim == 3:
            ts = ts.argmax(axis=2)

        mask = ts != self.ignore_label

        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        # ignore_label에 해당하는 데이터는 손실을 0으로 설정하여 무시함
        ls *= mask
        # T개의 softmax 층의 손실을 합산
        loss = -np.sum(ls)
        # T로 나누어 데이터 1개당 평균 손실 계산
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        # ignore_label에 해당하는 데이터는 기울기를 0으로 설정함
        dx *= mask[:, np.newaxis]

        # 원래 형태로 복원
        dx = dx.reshape((N, T, V))
        return dx
