import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


with open("names.txt", "r") as file:
    content = file.read()

words = content.splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi["."] = 0
itos = {i:s for s,i in stoi.items()}

g = torch.Generator().manual_seed(2147483647)
block_size = 3
X, Y = [], []

for w in words:
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # if __name__ == "__main__":
            # print("".join(itos[i] for i in context), "--->", itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27, 2), generator=g)

emb = C[X]