# coding: utf-8
from common.config import np, GPU


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, _ = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        # softmax
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

        # cross entropy error
        batch_size = t.shape[0]
        log_y = np.log(np.clip(y, 1e-6, 1))
        loss = -np.sum(t * log_y) / batch_size

        self.y = y
        self.t = t
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = dout * (self.y - self.t) / batch_size
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # sigmoidの出力
        self.t = None  # 教師データ

    def forward(self, x, t):
        # sigmoid
        y = 1 / (1 + np.exp(-x))

        # cross entropy error
        batch_size = t.shape[0]
        log_y = np.log(np.clip(y, 1e-6, 1))
        log_not_y = np.log(np.clip(1-y, 1e-6, 1))
        loss = -np.sum(t*log_y + (1-t)*log_not_y) / batch_size

        self.y = y
        self.t = t
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        # もしくは
        # for i, word_id in enumerate(self.idx):
        #     dW[word_id] += dout[i]

        return None
