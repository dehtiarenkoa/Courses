#realising XOR function(not-and(or, and)): w5 = [0,1,1] #5,1,3 (or); #w6 = [-1.5,1,1] #6,2,4 (and);#w9 = [2, -1, -1] #9,7,8 not-and

import numpy as np
input = [[0,0],[0,1],[1,0],[1,1]]
w5 = [0,1,1] #5,1,3 or
w6 = [-1.5,1,1] #6,2,4 and
w9 = [2, -1, -1] #9,7,8 not-and
#w6 = [2,-1,-1] #6,2,4 not-and
#w9 = [-1.5, 1, 1] #9,7,8 and
def perceptron(w, previous):
    return 1 if w[0] + w[1]*previous[0]+w[2]*previous[1] >0 else 0
"""input = np.array([[0,0],[0,1],[1,0],[1,1]])
print(input.shape)
print(input[:,0])
result = [perceptron(w9,[perceptron(w5, input[i]), perceptron(w6, input[i])]) for i in range(len(input))]"""
result =[]
for i in range(len(input)):
    n=perceptron(w5, input[i])
    m=perceptron(w6, input[i])
    print(n,m)
    result.append(perceptron(w9, [n, m]))
print(result)