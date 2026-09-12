from value import Value, draw_dot
import torch

# neuron 
x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
bias = Value(6.8813735870195432, label="bias")
x1w1 = x1 * w1; x1w1.label="x1w1"
x2w2 = x2 * w2; x2w2.label="x2w2"
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1 + x2w2"
n = x1w1x2w2 + bias; n.label = "n"
e = (2*n).exp()
o = (e - 1) / (e + 1)

"""
# backward
o.backward()
dot = draw_dot(o)
dot.render("graph", view=True, cleanup=True)
"""

"""
# numerical derivative
h = 0.0001
w1_ngradient = (((2*(x1*(w1+h)+x2*w2 + bias)).exp() - 1) / ((2*(x1*(w1+h)+x2*w2 + bias)).exp() + 1) - o) / h
print(w1_ngradient)
x1_ngradient = (((2*((x1+h)*w1+x2*w2 + bias)).exp() - 1) / ((2*((x1+h)*w1+x2*w2 + bias)).exp() + 1) - o) / h
print(x1_ngradient)
w2_ngradient = (((2*(x1*w1+x2*(w2+h) + bias)).exp() - 1) / ((2*(x1*w1+x2*(w2+h) + bias)).exp() + 1) - o) / h
print(w2_ngradient)
x2_ngradient = (((2*(x1*w1+(x2+h)*w2 + bias)).exp() - 1) / ((2*(x1*w1+(x2+h)*w2 + bias)).exp() + 1) - o) / h
print(x2_ngradient)
bias_ngradient = (((2*(x1*w1+x2*w2 + bias+h)).exp() - 1) / ((2*(x1*w1+x2*w2 + bias+h)).exp() + 1) - o) / h
print(bias_ngradient)
"""

#pytorch
x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.8813735870195432]).double(); b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)

print(o.data.item())
o.backward()
print("x1", x1.grad.item())
print("w1", w1.grad.item())
print("x2", x2.grad.item())
print("w2", w2.grad.item())
