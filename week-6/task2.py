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

random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

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
                dim = (0,1)
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

class Flatten:

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
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

def build_dataset(words, block_size):
    X, Y = [], []

    for w in words:
        context = [0] * block_size

        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    return torch.tensor(X), torch.tensor(Y)

def train_model(block_size):
    Xtr, Ytr = build_dataset(words[:n1], block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size)

    torch.manual_seed(42)

    n_embd = 10
    n_hidden = 200

    model = Sequential([
        Embedding(len(itos), n_embd),
        Flatten(),
        Linear(n_embd * block_size, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, len(itos))
    ])

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
            print(f"{block_size=}, {i:7d}/{max_steps:7d}: {loss.item():.4f}")

    for layer in model.layers:
        layer.training = False

    with torch.no_grad():
        logits = model(Xdev)
        dev_loss = F.cross_entropy(logits, Ydev).item()

    parameter_count = sum(p.nelement() for p in parameters)

    return parameter_count, dev_loss

params3, dev3 = train_model(3)
params8, dev8 = train_model(8)

print("Bağlam 3")
print("Parametre sayısı:", params3)
print("Dev loss:", dev3)

print("Bağlam 8")
print("Parametre sayısı:", params8)
print("Dev loss:", dev8)

print("Parametre artışı:", params8 - params3)
print("Parametre artış yüzdesi:", 100 * (params8 - params3) / params3)
print("Loss düşüşü:", dev3 - dev8)