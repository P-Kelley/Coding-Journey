'''This is one method of calculating a loss function. It is best used in classification problems as it allows for many simplifications. 
The equation is written L = sum( -y_predicted * log(y_result) ) . The sum is over all neurons in a batch of outputs. 
For a classification problem our predicted outputs will be all 0 except for our expected class will be 1. 
This simplifies the equation to be L = -log(y_result) which is a very simple equation for a loss function. 
Note: this is a natural log, (ln in math). Comp sci just uses log and assumaes it as ln

Here is a simple example of how the categotical loss function simplifies
import math

softmax_output = [0.8, 0.1, 0.2]

target_output = [1,0,0]
#This implies that target class is 0 because 1 is in the 0th position


loss = -(math.log(softmax_output[0]) * target_output[0] + 
        math.log(softmax_output[1]) * target_output[1] + 
        math.log(softmax_output[1]) * target_output[2] )

note this becomes the same as loss = -math.log(softmax_output[0])


Using numpy this can be implemented easier with class targets, an array containing the values where the expected value is
Ex: [0, 1, 1] Would mean that for the first set of data the target is at index 0, second and third set of data the target is at index 1
Ex:
softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_targets = [0,1,1]
softmax_outputs[[0, 1, 2], class_targets] 

The [0,1,2] array in the last softmax_outputs line shows what arrays in softmax outputs should line up with what target
Ex:
softmax_outputs[[0,1,2], [0,0,0]] 
this would give the 0th element of the first 3 arrays in softmax_outputs
'''
import numpy as np
softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])
class_targets = [0,1,1]
loss = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
print(loss)
ave_batch_loss = np.mean(loss)

# We should clip the predicted outputs because if an output is 0 the log is infinity, which messes up our data so we clip a very small number
cliped_outputs = np.clip(softmax_outputs, 1e-7, 1 - 1e-7)
class_targets = [0,1,1]
loss_clipped = -np.log(cliped_outputs[range(len(softmax_outputs)), class_targets])
print(loss)
ave_batch_loss_clipped = np.mean(loss)