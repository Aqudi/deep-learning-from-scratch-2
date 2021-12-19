# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from simple_skip_gram import SimpleSkipGram
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

# 전처리
text = "You say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

# CBOW
models = [
    ("CBOW", SimpleCBOW(vocab_size, hidden_size)),
    ("SkipGram", SimpleSkipGram(vocab_size, hidden_size)),
]
optimizer = Adam()
trainers = []
for name, model in models:
    trainers.append(Trainer(model, optimizer))

results = []
for (name, model), trainer in zip(models, trainers):
    trainer.fit(contexts, target, max_epoch, batch_size)

# word embedding 확인
for (name, model), trainer in zip(models, trainers):
    trainer.plot()
    word_vecs = model.word_vecs
    print(f"{name} 결과")
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
