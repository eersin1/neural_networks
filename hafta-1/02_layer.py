import math

def sigmoid(n):
    return 1/(1+math.exp(-n))

def weighted(inputs, weights):
    total = 0
    for i in range(len(inputs)):
        total += inputs[i]*weights[i]
    return total

def neuron(inputs, weights, bias):
    return sigmoid(weighted(inputs,weights)+bias)

def main():
    neurons = [
        {"b": -2, "w": [-3.81, 1.89, 6.23, 4.56]},
        {"b": 3, "w": [-5.39, 0.25, 2.34, 9.85]},
        {"b": -1, "w": [3.01, 4.82, 1.23, 4.23]},
        {"b": -10, "w": [4.32, 1.87, 6.46, 7.88]},
    ]
    x = [0.84, 0.43, 0.02, 0.65]
    for i,n in enumerate(neurons):
        print(f"Neuron {i}: {neuron(x, n["w"], n["b"])}")


main()