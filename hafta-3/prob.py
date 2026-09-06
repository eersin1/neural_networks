from count import N, itos
import torch

g = torch.Generator().manual_seed(2147483647)
P = N.float()
P /= P.sum(1, keepdim=True)


for i in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))

