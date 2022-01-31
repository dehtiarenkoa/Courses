import numpy as np
import timeit

X = np.array([[1, 60], [1, 50], [1, 75]])
y = np.array([10, 7, 12])

step1 = X.T.dot(X)
step2 = np.linalg.inv(step1)
step3 = step2.dot(X.T)
b = step3.dot(y)
print("!! only this variant is right !! ", b)

"""
li=len(X)
t=np.ones(li)
u=timeit.timeit('''import numpy as np 
X = np.array([[1, 60], [1, 50], [1, 75]])
li=len(X)
np.ones(li)''', number=100)
print("time: ", u, " ! ")
print("t.shape: ", t.shape)
"""

X1 = np.array([[60], [50], [75]])
#print(X1.shape)
#t1=np.ones(len(X1))
#A = np.vstack([t1, X1]).T

X2=X1.reshape(3,1)
print(X1.shape)
#b = np.ones((X2.shape[0], X2.shape[1] + 1) if X2.shape[1] else X2.len)
b = np.ones((X2.shape[0], X2.shape[1] + 1))
b[:, :-1] = X2
print("b: ", b)
# b=X - !!!! it works to receive the right coefficients
b1 = np.linalg.lstsq(b, y, rcond=None)[0]
print(b1)

"""
import matplotlib.pyplot as plt
_ = plt.plot(x, y, 'o', label='Original data', markersize=10)
_ = plt.plot(x, m*x + c, 'r', label='Fitted line')
_ = plt.legend()
plt.show()
"""