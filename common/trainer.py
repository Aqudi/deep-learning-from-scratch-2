# coding: utf-8
import sys


sys.path.append("..")
from common.np import *  # import numpy as np
from common.util import clip_grads
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


class RNNlmTrainer:
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x: np.ndarray, t: np.ndarray, batch_size: int, time_size: int):
        """미니 배치 제공 함수

        기존 Trainer 클래스와 다르게 Truncated BPTT를 하기 위해 배치를 미니배치로 나눠주고,
        해당 배치들을 순차적으로 제공하기 위해 offset을 조정해주는 작업이 필요하다.

        여기서 시간을 기준으로 자른 미니배치들은 time layer 내에서 시간을 고려한 계산을 할 때 사용된다.

        Args:
            x (np.ndarray): 데이터
            t (np.ndarray): one-hot 인코딩된 정답 라벨
            batch_size (int): 몇 개의 배치로 나눌 것인지
            time_size (int): truncated 할 기준 time 수

        Returns:
            np.ndarray: x의 미니배치
            np.ndarray: t의 미니배치
        """
        # 미니 배치 획득
        batch_x = np.empty((batch_size, time_size), dtype="i")
        batch_t = np.empty((batch_size, time_size), dtype="i")

        # 각 미니배치에서 샘플을 읽을 시작 위치를 계산
        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1

        return batch_x, batch_t

    def fit(
        self,
        xs: np.ndarray,
        ts: np.ndarray,
        max_epoch: int = 10,
        batch_size: int = 20,
        time_size: int = 35,
        max_grad: int = None,
        eval_interval: int = 20,
    ):
        """RNNlm을 학습시키는 함수

        기존 Trainer 클래스의 fit함수와 다르게 get_batch함수를 사용하여 batch를 가져오고,
        perplexity를 평가에서 사용한다.

        Args:
            xs (np.ndarray): 데이터
            ts (np.ndarray): 정답 라벨
            max_epoch (int, optional): 최대 epoch. Defaults to 10.
            batch_size (int, optional): 배치 크기. Defaults to 20.
            time_size (int, optional): truncated BPTT 단위. Defaults to 35.
            max_grad (int, optional): 최대 기울기 제한. Defaults to None.
            eval_interval (int, optional): perplexity 평가 주기. Defaults to 20.
        """
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                # Truncated BPTT를 하기위한 미니배치를 획득
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 기울기 계산, 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                # 공유된 가중치를 하나로 모음
                params, grads = remove_duplicate(model.params, model.grads)
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                # RNNlm은 퍼플렉서티 평가한다.
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(
                        "| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f"
                        % (
                            self.current_epoch + 1,
                            iters + 1,
                            max_iters,
                            elapsed_time,
                            ppl,
                        )
                    )
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label="train")
        plt.xlabel(f"iters (x{self.eval_interval})")
        plt.ylabel("perplexity")
        plt.show()
