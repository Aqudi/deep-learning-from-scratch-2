# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
from common.np import *
from common.layers import Embedding
from negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    """CBOW (SimpleCBOW 개선 버전)

    - one-hot 벡터의 확장성 해결을 위해 context와 target을 단어 id로 받음
    - Softmax와 CrossEntropyError의 확장성 해결을 위해 NegativeSamplingLoss 도입
    """

    def __init__(self, vocab_size, hidden_size, window_size, corpus) -> None:
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        # 계층 생성
        self.in_layers = []
        for _ in range(2 * window_size):
            layer = Embedding(W_in)  # Embedding 계층 사용
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 단어의 분산 표현 저장
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
