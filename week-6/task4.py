import torch
import torch.nn.functional as F
import random

with open("names.txt", "r") as file:
    content = file.read()

words = content.splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 8

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size

        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight

        if self.bias is not None:
            self.out += self.bias

        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNormWrong:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)

            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

class Embedding:

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]

class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)

        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out

    def parameters(self):
        return []

class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def create_model(batch_norm):
    return Sequential([
        Embedding(len(itos), 10),
        FlattenConsecutive(2),
        Linear(20, 200, bias=False),
        batch_norm(200),
        Tanh(),
        FlattenConsecutive(2),
        Linear(400, 200, bias=False),
        batch_norm(200),
        Tanh(),
        FlattenConsecutive(2),
        Linear(400, 200, bias=False),
        batch_norm(200),
        Tanh(),
        Linear(200, len(itos))
    ])

def train_model(batch_norm):
    torch.manual_seed(42)

    model = create_model(batch_norm)

    with torch.no_grad():
        model.layers[-1].weight *= 0.1

    parameters = model.parameters()

    for p in parameters:
        p.requires_grad = True

    max_steps = 200000
    batch_size = 32

    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        for p in parameters:
            p.grad = None

        loss.backward()

        lr = 0.1 if i < 150000 else 0.01

        for p in parameters:
            p.data += -lr * p.grad

        if i % 10000 == 0:
            print(f"{batch_norm.__name__}: {i:7d}/{max_steps:7d}: {loss.item():.4f}")

    for layer in model.layers:
        layer.training = False

    with torch.no_grad():
        logits = model(Xdev)
        dev_loss = F.cross_entropy(logits, Ydev).item()

    return dev_loss

x = torch.randn(32, 4, 200)

wrong_mean = x.mean(0, keepdim=True)
correct_mean = x.mean((0, 1), keepdim=True)

print("Girdi şekli:", x.shape)
print("Yanlış ortalama şekli:", wrong_mean.shape)
print("Doğru ortalama şekli:", correct_mean.shape)

print()
print("Yanlış yöntemde sadece batch ekseni üzerinden ortalama alınıyor.")
print("Bu durumda her zaman konumu için farklı bir ortalama kalıyor.")
print("Girdi B x T x C biçiminde olduğu için BatchNorm istatistikleri B ve T eksenlerinin ikisi üzerinden alınmalıdır.")
print("Bu nedenle doğru eksenler (0, 1)'dir ve her kanal için tek bir ortalama ile varyans elde edilir.")

wrong_loss = train_model(BatchNormWrong)
correct_loss = train_model(BatchNorm1d)

print()
print("Düzeltmeden önce dev loss:", wrong_loss)
print("Düzeltmeden sonra dev loss:", correct_loss)
print("Fark:", wrong_loss - correct_loss)