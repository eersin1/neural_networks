import math
import matplotlib.pyplot as plt

def sigmoid(n):
    return 1/(1+math.exp(-n))

def weighted(inputs, weights):
    total = 0
    for i in range(len(inputs)):
        total += inputs[i]*weights[i]
    return total

def neuron(inputs, weights, bias):
    return sigmoid(weighted(inputs,weights)+bias)

def loss(actuals):
    total = 0
    targets = [
        0.5,
        0,
        0,
        0
    ]
    for i in range(len(targets)):
        total += (actuals[i] - targets[i])**2
    return total/len(actuals)
    

def main():
    neurons = [
        {"b": -2, "w": [-3.81, 1.89, 6.23, 4.56]},
        {"b": 3, "w": [-5.39, 0.25, 2.34, 9.85]},
        {"b": -1, "w": [3.01, 4.82, 1.23, 4.23]},
        {"b": -10, "w": [4.32, 1.87, 6.46, 7.88]},
    ]
    x = [0.84, 0.43, 0.02, 0.65]
    b_new = [-17, -12, -7, -6, -5, -4, -3, -2, -1, -0.75, -0.7009, -0.5, -0.25, 0, 1, 2, 3, 8, 13]
    y_plot = []
    for i in range(len(b_new)):
        neurons[0]["b"] = b_new[i]
        actuals = [neuron(x, n["w"], n["b"]) for n in neurons]
        y_plot.append(loss(actuals))
    plt.plot(b_new, y_plot)
    plt.title("Change in Loss")
    plt.xlabel("Neuron 1 Bias")
    plt.ylabel("Loss")
    plt.show()

main()