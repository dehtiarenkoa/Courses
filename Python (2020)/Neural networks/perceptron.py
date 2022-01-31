# it doesnt work

import numpy as np

y = np.array([[1],[1],[0]])
x = np.array([[1, 0.3], [0.4, 0.5], [0.7, 0.8]])
# x = np.array([[1,1,1],[1, 0.4, 0.7],[0.3, 0.5, 0.8]])
# x = np.array([[1, 0.4, 0.7],[0.3, 0.5, 0.8]])
# x = np.array([[1, 0.4, 0.7],[0.3, 0.5, 0.8]])
w = np.array([[0], [0], [0]])
print("y: ", y)
print("x: ", x.shape)
print("x: ", "\n",x)
print("w: ", w.shape)
ones = np.ones((3,1))
x=np.hstack((ones, x))
print("x: \n",x)
#result = False
#while not result:
y1 = w*x
print("y1: \n",y1)
# y2 = np.ones((3,1))
# print("y2: \n",y2)
y2 = np.array([y1.sum(axis=0)]).T
print("y2: \n",y2)
y3 = y-y2
print("y3: \n",y3)
"""w1 = y3/x
print("w1: \n",w1)
 [[1.         1.         3.33333333]
 [1.         2.5        2.        ]
 [0.         0.         0.        ]] 
y4 = w1*x
print("y4: \n",y4)
y2 = np.array([y4.sum(axis=0)]).T
print("y2: \n",y2)"""

r = np.empty((6, 2))
print("r: \n",r)