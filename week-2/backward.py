from value import Value, draw_dot

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

o.backward()

dot = draw_dot(o)
dot.render("graph", view=True, cleanup=True)

'''
# multivariable case test

a = Value(3.0, label="a")
b = a + a ; b.label="b"
b.backward()
dot = draw_dot(b)
dot.render("graph", view=True, cleanup=True)
'''

