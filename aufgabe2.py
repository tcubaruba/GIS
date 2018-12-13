from random import uniform
from math import exp
from math import tanh



def create_network(n_inp, n_hidden, n_out):
    network = list()
    #putting random weights in layer
    hidden_layer = [{'weights': [uniform(-1.0,1.0) for i in range(n_inp)]} for i in range(n_hidden)]
    output_layer = [{'weights': [uniform(-1.0,1.0)  for i in range(n_hidden)]} for i in range(n_out)]
    network.append(hidden_layer)
    network.append(output_layer)
    return network

#def activation function with sigmoid function
def activate(weight, inp, fun):
    sum_nodes = weight[-1]
    for i in range(len(weight)-1):
        sum_nodes += inp[i]*weight[i]
    if(fun=="sigmoid"):
        sigmoid = 1.0/(1.0+exp(-sum_nodes))
        return sigmoid
    elif(fun=="tanh"):
        return tanh(sum_nodes)
    elif(fun=="rectifler"):
        return abs(sum_nodes*(sum_nodes>0))
    else:
        return 0


def forward(network, inp, fun):
    for i in range(2):
        if i==0:
            hidden_output = []
            for node in network[0]:
                node['output'] = activate(node['weights'], inp, fun)
                hidden_output.append(node['output'])
        else:
            outter_output = []
            for node in network[1]:
                node['output'] = activate(node['weights'], hidden_output, fun)
                outter_output.append(node['output'])
    return outter_output

def error(network, y):
    error = 0
    for node in network[1]:
        y_c = node['output']
        error = ((y - y_c)**2)/2
    return error

def print_network(network, y):
    print("hidden layer:")
    print(network[0])
    print ("output layer:")
    print(network[1])
    print()
    print("OUTPUT")
    for node in network[1]:
        print(node['output'])
    print()
    print("Y: %s" %y)
    print("SQUARED ERROR: %s" % error(network, y))




x = [0.5, 0.2, 0.8]
y = 0.5
print()
print()
print("NEURAL NETWORK WITH SIGMOID:")

network1 = create_network(3,2,1)
forward(network1, x, "sigmoid")
print_network(network1,y)
print()
print()
print("NEURAL NETWORK WITH TANH:")

network2 = create_network(3,2,1)
forward(network2, x, "tanh")
print_network(network2,y)
print()
print()
print("NEURAL NETWORK WITH RECTIFLER:")

network3 = create_network(3,2,1)
forward(network3, x, "rectifler")
print_network(network3,y)



