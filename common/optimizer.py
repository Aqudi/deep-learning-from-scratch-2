# -*- coding: utf-8 -*-
"""optimizer.py

공통 메소드
update(params, grads)
    params: 가중치 리스트
    grads: 기울기 리스트
"""
import numpy as np


class SGD:
    """확률적 경사하강법(Stochastic Gradient Descent)"""

    def __init__(self, lr=0.01) -> None:
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
