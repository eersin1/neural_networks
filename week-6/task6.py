import torch
import torch.nn.functional as F
import random

with open("turkish_names.txt", "r", encoding="utf-8") as file:
    content = file.read()

words = [word.strip().lower() for word in content.splitlines() if word.strip()]

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

class Flatten:

    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []

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

def train_model(model, block_size):
    Xtr, Ytr = build_dataset(words[:n1], block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], block_size)

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
            print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")

    for layer in model.layers:
        layer.training = False

    with torch.no_grad():
        logits = model(Xdev)
        dev_loss = F.cross_entropy(logits, Ydev).item()

    return model, dev_loss

torch.manual_seed(42)

turkish_mlp = Sequential([
    Embedding(len(itos), 10),
    Flatten(),
    Linear(10 * 3, 200, bias=False),
    BatchNorm1d(200),
    Tanh(),
    Linear(200, len(itos))
])

turkish_mlp, mlp_loss = train_model(turkish_mlp, 3)

torch.manual_seed(42)

turkish_wavenet = Sequential([
    Embedding(len(itos), 24),
    FlattenConsecutive(2),
    Linear(24 * 2, 128, bias=False),
    BatchNorm1d(128),
    Tanh(),
    FlattenConsecutive(2),
    Linear(128 * 2, 128, bias=False),
    BatchNorm1d(128),
    Tanh(),
    FlattenConsecutive(2),
    Linear(128 * 2, 128, bias=False),
    BatchNorm1d(128),
    Tanh(),
    Linear(128, len(itos))
])

turkish_wavenet, wavenet_loss = train_model(turkish_wavenet, 8)

print()
print("Türkçe bağlam 3 MLP dev loss:", mlp_loss)
print("Türkçe bağlam 8 WaveNet dev loss:", wavenet_loss)
print("Loss düşüşü:", mlp_loss - wavenet_loss)
print("Yüzdesel düşüş:", 100 * (mlp_loss - wavenet_loss) / mlp_loss)

print()
print("Üretilen isimler:")

g = torch.Generator().manual_seed(2147483647)

for _ in range(20):
    out = []
    context = [0] * 8

    while True:
        logits = turkish_wavenet(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()

        context = context[1:] + [ix]
        out.append(ix)

        if ix == 0:
            break

    print("".join(itos[i] for i in out))
