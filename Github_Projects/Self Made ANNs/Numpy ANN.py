import numpy as np
input = [[1.2 ,3.2 , 4.1, 5.4],
         [2, 5, -1, 2],
         [-1.5, 2.7, 3.3, -0.8]]
weights1 = [[0, 0, 0, 1], 
            [1,2,3,4], 
            [4,3,5,6]]
bias1 = [1, 2, 3]


'''
output = np.dot(weights, np.array(input).T) + bias
This is not useful because it results in one neurons output in one array rather than one batch of input into one array
We want the layers output from a batch, not one neurons output from all the batches
'''
output1 = np.dot(input, np.array(weights1).T) + bias1
print(output1)



weights2 = [[-0.3, 2, -1], 
            [1, 0, 0], 
            [4, -3, 5.2]]

bias2 = [1, 4, 5]

output2 = np.dot(output1, np.array(weights2).T) + bias2 

print(output2)