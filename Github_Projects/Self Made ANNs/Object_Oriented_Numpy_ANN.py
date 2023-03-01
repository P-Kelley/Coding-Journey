import numpy as np
import math


#This is a temp data generator for testing purposes
def create_data(points, classes):
    X = np.zeros((points * classes, 2))
    Y = np.zeros(points * classes, dtype = 'uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        Y[ix] = class_number
    return X,Y 


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities



layerlist = []
#defining hidden layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, activationfunction):
        self.af = activationfunction
        self.weights = np.random.random([n_inputs, n_neurons])
        self.biases = np.zeros(n_neurons)
        self.derivatives = np.zeros((n_inputs,n_neurons))
        layerlist.append(self)
        #The derivatives we are refering to is the derivative of the loss function with respect to weight or bias so dL/dW . We perform this for each weight and bias.
        #We can brute force finding this derivative by running a prediction of the network, then changing it very slightly and running the prediction again to see the effect on the loss function.
        #If we perform this for each weight and bias in the network, we can then see how the loss is effected by each weight and bias for the current input, and move each one very slightly in the correct direction
        #This is very inefficent for large networks, so we will not do this here. Instead we will use chain rule to simplify dL/dW = dL/dA * dA/dZ * dZ/dW where A is the activation function and Z is the output of one neuron (Z = W * input + bias)
        #So for standard neural networks dZ/dW = input (note this is not always true for other types of neural networks such as convultion networks). dA/dZ depends on the type of activation function dL/dA depends on the type of loss function used

        
    def forward(self, inputs):
        #Legacy function used in random movement network
        self.output = np.dot(inputs, self.weights) + self.biases
    def forward_ReLU(self, inputs):
        #The derivative (dA/dZ) for this activation function relative to input is 0 for input < 0 and 1 for input > 0. Technically it is undefined at 0 but we will say that it is 0
        self.activations = inputs
        active = np.dot(inputs, self.weights) + self.biases
        relu = Activation_ReLU()
        relu.forward(active)
        self.output = relu.output
    def forward_Softmax(self, inputs):
        #The derivative for this function is a bit complicated because it is a vector function, so it takes in multiple inputs and gives a single vector output. As a result, we have to take the Jacobian (fancy vector derivative)
        #This means to take the derivative for a specific input i, and take it in regards to another input j. j and i are independent. 
        self.activations = inputs
        active = np.dot(inputs, self.weights) + self.biases
        soft = Activation_Softmax()
        soft.forward(active)
        self.output = soft.output

    def back_ReLU(self):
        #backout is dAdZ, should return 1 for x > 0 and 0 for x <= 0
        #note, we change values of 0 to 1e-7 to prevent division by 0 errors
        
                
    
        back = self.output
        for i in range(len(self.output)):
            for n in range(len(self.output[0])):
                if back[i][n] <= 0:
                    back[i][n] = 0
                else:
                    back[i][n] = 1

        dLdW = np.dot(self.activations.T, back)
        return dLdW
        


    
    def back_Softmax(self,truth):
        dLdZ = self.output - truth
        a = np.dot(self.activations.T, dLdZ)
        return a




class runModel:
    def __init__(self, input):
        for i,layer in enumerate(layerlist):
            if layer.af.lower() == "relu":
                layer.forward_ReLU(input)
                input = layer.output
            elif layer.af.lower() == 'softmax':
                layer.forward_Softmax(input)
                input = layer.output
            else:
                raise "Activation function not understood at layer {}".format(i)
        self.output = layerlist[i].output

        
    def backpropagate(self,learnrate, truth):
        #Goes from output layer to input layer and adjust weights and biases in correct direction
        for i,layer in enumerate(reversed(layerlist)):
            i = len(layerlist) - i

            if layer.af.lower() == 'relu':
                step = learnrate * np.negative(layer.back_ReLU())
                layer.weights = layer.weights + step
            elif layer.af.lower() == 'softmax':
                step = learnrate * np.negative(layer.back_Softmax(truth))
                layer.weights = layer.weights + step
            
            

class Loss:
    def calculate(self,output, truth):
        sample_losses = self.forward(output,truth)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    #This function is L = sum(-y_true * ln(y_result)) in classification problems y_true is 0 for all values except one, so it becomes L = -ln(y_result) for the neuron corresponding to the expected truth
    # The derivative becomes a simple dL/dA = -1/y_result.  y_result is clipped to avoid division by 0 errors
    def forward(self, y_predicted, y_true):
        samples = len(y_predicted)
        y_pred_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)
        
        
        correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)

        neg_log_liklyhood = -np.log(correct_confidences)
        return neg_log_liklyhood

