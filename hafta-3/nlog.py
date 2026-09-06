from count import words, stoi, N
import torch

P = (N+1).float()
P /= P.sum(1, keepdim=True)

log_likelihood = 0
n=0

for w in words[:5]:
    #for w in ["andrejq"]:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

nll = -log_likelihood
print(f"{nll/n}")