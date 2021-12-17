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


class Momentum:
    """Momentum (SGD 느린 속도와 local minima 문제를 개선한 optimizer)

    SGD에 관성의 개념을 적용하여 이전 이동거리[v]와 관성 계수[momentum]에 따라
    매개변수를 업데이트한다.

    이를 통해 더욱 빠르게 학습을 할 수 있으며 gradient가 0인 곳에서도 관성에 의해
    매개변수를 업데이트할 수 있다.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            # parameter가 이전에 이동한 거리에 관성을 부여하고, 속도를 업데이트한다.
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            # 해당 속도만큼 parameter 업데이트
            params[i] += self.v[i]


class AdaGrad:
    """AdaGrad (Adaptive Gradient)

    지속적으로 변하던 parameter는 최적값에 가까워졌을 것이고,
    한 번도 변하지 않은 parameter는 더 큰 변화를 줘야한다.

    Adagrad에서는 이를 위해 gradient^2의 합을 이동거리의 척도로 사용하여
    이 수치가 크다면 상대적으로 적은 변화를 주고, 작다면 큰 변화를 준다.
    """

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            # gradient^2 값의 합을 저장
            self.h[i] += grads[i] * grads[i]
            # 지금까지 parameter가 변화한 만큼 더 적게 update
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class RMSprop:
    """RMSprop

    Adagrad의 문제점을 계산하기 위해 지수이동평균을 적용하여
    학습의 최소 step을 유지할 수 있도록 만들었다.

    Adagrad에서 사용한 이동거리의 척도에 decay_rate (forgetting factor)를
    적용하여 학습이 진행됨에 따라 학습속도가 지속적으로 줄어들도록 한다.
    """

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] *= self.decay_rate
            self.h[i] += (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)


class Adam:
    """Adam (Adaptive Moment Estimation)

    RMSProp과 Momentum 기법을 합친 optimizer이다.

    Momentum에서는 관성계수와 함께 계산된 관성으로 parameter를 update 하지만
    Adam에서는 기울기 값과 기울기의 제곱값의 지수이동평균을 이용해 step 변화량을 조절한다.

    논문: http://arxiv.org/abs/1412.6980v8
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1

        lr_t = (
            self.lr
            * np.sqrt(1.0 - self.beta2 ** self.iter)
            / (1.0 - self.beta1 ** self.iter)
        )

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
