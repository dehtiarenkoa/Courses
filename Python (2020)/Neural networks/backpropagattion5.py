import numpy as np
n = 1
n_l = 3
n_input = 3
weights_1 = np.array([[0.7,0.2,0.7],[0.8,0.3,0.6]])
weights_2 = np.array([[0.2, 0.4]])
x = np.array([[0],[1],[1]])
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
def sigmoid_prime (x):
    return sigmoid(x)*(1-sigmoid(x))
    #return np.array([[sigmoid(x)*(1-sigmoid(x)) for x in z] for z in sums])

def maxy(x):
    return x if x>0 else 0
def maxy_prime(x):
    return int(x>0)
    #return 1 if x>0 else 0
    #return np.array([[(1 if x>0 else 0) for x in z] for z in sums])

def forward_activation ():
   ######n_input = weights_1.shape[0]
    for i in range(n_input):
        global y_1, y_2, sum_1, sum_2
        sum_1 = weights_1.dot(x)
        print("sum_1: ", sum_1)
        #y_1 = np.array (w1*x[i][1] for w1 in weights_1[i])
        m=maxy(sum_1[0][0])
        s=sigmoid(sum_1[1][0])
        print("maxy ", m)
        print("sy ", s)
        y_1 = np.array([[m],[s]])
        print("y_1: ", y_1)
        sum_2 = weights_2.dot(y_1)
        print("sum_2: ", sum_2)
        #global y_2
        y_2 = sigmoid(sum_2)
        print("y_2: ", y_2)

def back_propagation():
    forward_activation()
    nabla_L_J = y_2 - y
    print("nabla_L_J: ", nabla_L_J)
    delta_L = nabla_L_J * sigmoid_prime(sum_2)
    print("delta_L: ",delta_L)
    temp = np.array([[maxy_prime(sum_1[0][0])], [sigmoid_prime(sum_1[1][0])]])
    print("temp", temp)
    #delta_2 = (weights_2*delta_L)* sigmoid_prime(sum_2)
    delta_2 = (weights_2 * delta_L) * (temp.T)
    print("delta_2", delta_2)
    print("delta_2.shape", delta_2.shape)
    #(weights_1.dot(x.T)).T
    #delta_1  = ((weights_1.T).dot((delta_2.T)))*(temp.T)
    #print("delta_1",delta_1)
    #print("delta_1.shape", delta_1.shape)
    #dJ_dW = delta_2*(weights_1.T)
    #dJ_dW = delta_2*(weights_1.T)
    #vmaxy = np.vectorize(maxy)
    #dJ_dW = vmaxy((weights_1.dot(x)).T)*delta_2
    #print(weights_1.T)
    dJ_dW = x.dot(delta_2)
    print(dJ_dW)
#!!!!!!!! the right answer is -0.01829328 -0.00751855. (delta_2).
# because [a1]=[x] ==>> dJ_dW = x.dot(delta_2)
#[[ 0.          0.        ]
#[-0.01829328 -0.00751855]
#[-0.01829328 -0.00751855]]


print(b_2.shape)
print(x.shape)
print(weights_1.shape)
print(weights_2.shape)
back_propagation()


#print(sigmoid_prime(sums).shape)
"""
sum_1:  [[0.9]
 [0.9]]
maxy  0.8999999999999999
sy  0.7109495026250039
y_1:  [[0.9      ]
 [0.7109495]]
y_2:  [[0.61405267]]
sum_1:  [[0.9]
 [0.9]]
maxy  0.8999999999999999
sy  0.7109495026250039
y_1:  [[0.9      ]
 [0.7109495]]
y_2:  [[0.61405267]]
sum_1:  [[0.9]
 [0.9]]
maxy  0.8999999999999999
sy  0.7109495026250039
y_1:  [[0.9      ]
 [0.7109495]]
y_2:  [[0.61405267]]
delta_L:  [[-0.09146642]]
delta_2 [[-0.00433536 -0.00867072]]
delta_2.shape (1, 2)
temp [[1.        ]
 [0.20550031]]
[[-0.00997133 -0.00204911]
 [-0.00346829 -0.00071273]
 [-0.00823719 -0.00169274]]"""
# answer= -0.01829328 -0.00751855
# instead of the -0.00823719 -0.00169274 or -0.00576603 -0.00101565