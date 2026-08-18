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
    b = 10
    x = [0.84, 0.43]
    w = [-3.81, 1.8]
    print(neuron(x,w,b))

main()