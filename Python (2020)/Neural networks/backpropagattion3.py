import numpy as np
n = 1
n_l = 3
n_input = 3
weights_1 = np.array([[8.0,7.0,8.0],[10.0,10.0,9.0]])
weights_2 = np.array([[10.0, 9.0]])
x = np.array([[0],[1],[2]])
b_1 = np.array ([[0,0,0]] )
b_2 = np.array ([[0,0]] )
y_1 = np.array([[0],[0]])
y_2 = 0
y = 1
#deltas = np.zeros((weights.shape[1],weights.shape[0]))
activations_1 = np.zeros_like(weights_1)
sums = np.array([[0],[1],[2]])

def get_error(deltas, sums, weights):
    delta_n = ((deltas.dot(weights))*sigmoid_prime(sums)).mean(axis = 0)
    return delta_n
def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))
def sigmoid_prime (sums):
    return np.array([[sigmoid(x)*(1-sigmoid(x)) for x in z] for z in sums])
def forward_activation ():
   ######n_input = weights_1.shape[0]
    for i in range(n_input):
        global y_1, y_2, sum_1, sum_2
        sum_1 = weights_1.dot(x)
        #print("sum_1: ", sum_1.shape)
        #y_1 = np.array (w1*x[i][1] for w1 in weights_1[i])
        y_1 = np.array([sigmoid(s1) for s1 in sum_1])
        print("y_1: ", y_1.shape)
        sum_2 = weights_2.dot(y_1)
        #global y_2
        y_2 = sigmoid(sum_2)
        print("y_2: ", y_2)

def back_propagation():
    forward_activation()
    nabla_L_J = y_2 - y
    delta_L = nabla_L_J * sigmoid_prime(sum_2)
    print(delta_L)
    delta_2 = (weights_2*delta_L)* sigmoid_prime(sum_2)
    print("delta_2", delta_2)
    print("delta_2.shape", delta_2.shape)
    delta_1  = ((weights_1.T).dot((delta_2.T)))*(sigmoid_prime(sum_1).T)
    print(delta_1)


print(b_2.shape)
print(x.shape)
print(weights_1.shape)
print(weights_2.shape)
back_propagation()


#print(sigmoid_prime(sums).shape)
"""
y_1:  (2, 1)
y_2:  [[0.99999999]]
y_1:  (2, 1)
y_2:  [[0.99999999]]
y_1:  (2, 1)
y_2:  [[0.99999999]]
[[-3.13913279e-17]]
delta_2 [[-1.75879219e-24 -1.58291297e-24]]
delta_2.shape (1, 2)
[[-3.06824752e-33 -2.06738940e-35]
 [-2.88776238e-33 -1.94577826e-35]
 [-2.90581089e-33 -1.95793937e-35]]"""