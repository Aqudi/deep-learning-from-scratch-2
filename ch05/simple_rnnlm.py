# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import numpy as np
from common.time_layers import *


class SimpleRnnlm:
    """RNN으로 만든 조건부 언어 모델

    언어 모델은 문장 (단어 시퀸스)에 확률을 부여한다.

    특히, 조건부 언어 모델은 이전에 출현한 단어들의 시퀸스가 출현할 확률을 사전 확률로 하고,
    사후 확률인 이후에 출현할 단어의 확률을 계산하는 모델이다.
    """

    def __init__(self, vocab_size, wordvec_size, hidden_size) -> None:
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 가중치 초기화
        embed_W = (rn(V, D) / 100).astype("f")
        # - RNN 계층과 Affine 계층에는 Xavier 초깃값을 사용함
        # - 이는 이전 계층의 노드의 루트를 표준편차로 한 분포로 값들을 초기화하는 것임
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype("f")
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype("f")
        rnn_b = np.zeros(H).astype("f")
        affine_W = (rn(H, V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")

        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            # Truncated BPTT로 학습한다고 가정하여 stateful을 True로 설정
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b),
        ]

        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        # 각 층들을 잘 정의해놨기 때문에 순차적으로 호출하면 된다.
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        # 마찬가지로 각 층들이 잘 정의되어 있기 때문에 순차적으로 호출하면 된다.
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        """RNN layer의 reset_state 함수를 호출해준다."""
        self.rnn_layer.reset_state()
