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
    
# too large/small biases makes the graident descent stuck
def main():
    neurons = [
        {"b": -2, "w": [-3.81, 1.89, 6.23, 4.56]},
        {"b": 3, "w": [-5.39, 0.25, 2.34, 9.85]},
        {"b": -1, "w": [3.01, 4.82, 1.23, 4.23]},
        {"b": -10, "w": [4.32, 1.87, 6.46, 7.88]},
    ]
    x = [0.84, 0.43, 0.02, 0.65]
    slope = 1
    while not -0.0001 < slope < 0.0001:
        actuals = [neuron(x, n["w"], n["b"]) for n in neurons]
        neurons[0]["b"] = neurons[0]["b"] + 0.00001
        increment = [neuron(x, n["w"], n["b"]) for n in neurons]
        slope = (loss(increment) - loss(actuals)) / 0.00001
        neurons[0]["b"] = neurons[0]["b"] - 0.25*slope
        print(neurons[0]["b"])
    print(neurons[0]["b"])

main()