# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    """EmbeddingDot 층

    Embeded 값에 가중치 값을 곱해주는 층
    """

    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    """UnigramSampler 층

    Unigram이란 1개의 연속된 단어이다. 2개의 연속된 단어는 bigram, 3개의 연속된 단어는
    trigram이라고 불린다. UnigramSampler는 각 단어를 기준으로 확률 분포를 만들어 단어를
    추출하는 sampler를 의미한다.

    말 뭉치에서 각 단어가 출현하는 횟수를 구하여 확률 분포로 나타내고, 이 확률 분포를 토대로
    단어를 샘플링한다. 이때 출현 확률이 낮은 단어를 버리지 않기 위하여 0.75와 같은 power를
    제곱함으로써 원래 확률이 낮은 단어의 확률을 살짝 높여준다.
    """

    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        # 출현 확률이 낮은 단어를 버리지 않기 위함
        # 0.75 제곱을 함으로써 원래 확률이 낮은 단어의 확률을 살짝 높여준다.
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(
                    self.vocab_size, size=self.sample_size, replace=False, p=p
                )
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # replace=True로 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(
                self.vocab_size,
                size=(batch_size, self.sample_size),
                replace=True,
                p=self.word_p,
            )
        return negative_sample


class NegativeSamplingLoss:
    """NegativeSamplingLoss 층

    Softmax와 CrossEntropyError를 사용하는 다중 분류 문제에서 Sigmoid와 CrossEntropyError를
    사용하는 긍부정 이진 분류 문제로 바꿔준다.

    학습 시 라벨이 1인 정답에 해당하는 단어 벡터와의 스코어를 계산하여 긍정적인 예시들을 학습하고,
    negative sampling 기법을 이용하여 뽑아낸 라벨이 0인 오답에 해당하는 단어 벡터와의 스코어를
    계산하여 부정적인 예시들을 학습한다.

    결과적으로 손실함수의 계산을 간단히 하고, 일부 단어와의 유사도만 비교하여 학습 시간을 단축할 수 있다.
    """

    def __init__(self, W, corpus, power=0.75, sample_size=5) -> None:
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        # 긍정적인 예 1개, 부정적인 예 sample_size 개만큼의 layer 생성
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적인 예 순전파 - label은 1
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적인 예 순전파 - label은 0
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negatvie_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negatvie_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for (l0, l1) in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            # Repeat 노드의 역전파 - Sum 노드
            dh += l1.backward(dscore)
        return dh


if __name__ == "__main__":
    # UnigramSampler 동작 확인
    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    power = 0.75
    sample_size = 2
    sampler = UnigramSampler(corpus, power, sample_size)
    target = np.array([1, 3, 0])
    negative_sample = sampler.get_negative_sample(target)
