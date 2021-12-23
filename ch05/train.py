# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from common.trainer import RNNlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 하이퍼파리미터 설정
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100

# 데이터 읽기 (1000개만)
corpus, word_to_id, id_to_word = ptb.load_data("train")
corpus_size = 10000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus)) + 1

xs = corpus[:-1]
ts = corpus[1:]
data_size = len(xs)
print(f"말뭉치 크기: {corpus_size}, 어휘 수: {vocab_size}")

# 학습 시 사용하는 변수들
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 모델 생성
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RNNlmTrainer(model, optimizer)

# 학습 시작
trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot()
