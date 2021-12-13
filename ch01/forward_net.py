# -*- coding: utf-8 -*- 
import numpy as np


class Sigmoid:
    def __init__(self) -> None:
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    """완전연결계층에 의한 변환은 기하학에서 affine변환에 해당"""

    def __init__(self, W, b) -> None:
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]

        # 모든 가중치를 리스트로 모으기
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)
