# coding: utf-8
import sys

sys.path.append("..")
import numpy as np
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
                optimizer.update(model.params, model.grads)

                total_loss += loss
                loss_count += 1

                # 학습 경과 출력
                if (iters + 1) % 10 == 0:
                    avg_loss = total_loss / loss_count
                    print(
                        f"| epoch {epoch+1} | 반복 {iters+1} / {max_iters} | 손실 {avg_loss:.2f}"
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
