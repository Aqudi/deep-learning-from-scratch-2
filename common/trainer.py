# coding: utf-8
import sys

sys.path.append("..")
from common.np import *  # import numpy as np
import time
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        print(max_iters)

        start_time = time.time()
        for epoch in range(max_epoch):
            # 1. 데이터 섞어서 미니배치 생성
            idx = np.random.permutation(data_size)
            x = x[idx]
            t = t[idx]
            for iters in range(max_iters):
                batch_x = x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = t[iters * batch_size : (iters + 1) * batch_size]

                # 2. 기울기 계산, 3. 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                # 공유된 가중치를 하나로 모음
                params, grads = remove_duplicate(model.params, model.grads)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                # 학습 경과 출력 (평가)
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(
                        f"| epoch {epoch+1} | 반복 {iters+1} / {max_iters} | 시간 {elapsed_time}[s] | 손실 {avg_loss:.2f}"
                    )
                    self.loss_list.append(avg_loss)
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label="train")
        plt.xlabel(f"iters (x{self.eval_interval})")
        plt.ylabel("loss")
        plt.show()


def remove_duplicate(params: list, grads: list):
    """중복되는 가중치를 하나로 모아 그 가중치에 대응하는 기울기를 더한다.

    여러 계층에서 같은 가중치를 공유하여 params 리스트에 같은 가중치가 여러개 존재한다.
    이로 인해 기존 구현에서는 Adam이나 Momentum같은 옵티마이저의 본래 동작과 달라진다.
    """
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif (
                    params[i].ndim == 2
                    and params[j].ndim == 2
                    and params[i].T.shape == params[j].shape
                    and np.all(params[i].T == params[j])
                ):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg:
                    break
            if find_flg:
                break

        if not find_flg:
            break

    return params, grads
