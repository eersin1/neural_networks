from value import Value, draw_dot

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e = a*b
e.label = "e"
d = e + c
d.label = "d"
f = Value(-2.0, label="f")
L = d * f
L.label = "L"

# manual backprop
L.grad = 1.00
d.grad = -2.00
f.grad = 4.00
c.grad = -2.00
e.grad = -2.00
a.grad = 6.00
b.grad = -4.00
'''
dot = draw_dot(L)
dot.render("graph", view=True, cleanup=True)
'''

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
o = n.tanh()

# neuron manual backprop
o.grad = 1.0
n.grad = 1 - o.data**2
x1w1x2w2.grad = 0.5
bias.grad = 0.5
x2w2.grad = 0.5
x1w1.grad = 0.5
w2.grad = 0.0
x2.grad = w2.data * x2w2.grad
w1.grad = x1.data * x1w1.grad
x1.grad = w1.data * x1w1.grad

dot = draw_dot(o)
dot.render("graph", view=True, cleanup=True)
