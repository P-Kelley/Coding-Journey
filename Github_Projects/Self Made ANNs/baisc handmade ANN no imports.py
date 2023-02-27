input = [1.2 ,3.2 , 4.1, 5.4]
weights = [[14, 1.5, 2.3, 4.1], [1,2,3,4], [4,3,5,6]]
bias = [2, 1, 3]
'''#single neuron
output = input[0] * weights[0] + input[1] * weights[1] + input[2] * weights[2] + input[3] * weights[3] + bias
print(max(0, output))
'''
#Single layer of three neurons
layer_output = []
for neuron_weight, neuron_bias in zip(weights,bias):
    neuron_output = 0
    for n_input, weight in zip(input, neuron_weight):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)


