import numpy as np
import random
"""a = np.array([[1,2,3], [4,5,6]])
b = np.zeros((5,))
a = np.array([1, 2, 3], float)
a[:,np.newaxis]
d=a[:,np.newaxis].shape
d=range(6)
print(a)
print(d.__doc__)

a = np.array([1,2,3], float)
b = np.array([4,5], float)
print(a)
print(b)
print(a+b)

a = np.array([1, np.NaN, np.Inf], float)
print(a)
g=np.isinf(a)
print(g)

a = np.array([[0, 1], [2, 3]], float)
b = np.array([2, 3], float)
c = np.array([[1, 1], [4, 0]], float)
print(np.dot(a, b))
print(np.dot(b, a))
print(b)

------------
a = np.eye(3, 4, k=0)
b=np.eye(3, 4, k=1)
a=2*a
b=a+b
print(b)
-------------

w = np.array(random.sample(range(1000), 12))
w = w.reshape((2,2,3))
w = w.reshape(())
print(w)
---------



x_shape = tuple(map(int, input().split()))
X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)
y_shape = tuple(map(int, input().split()))
Y = np.fromiter(map(int, input().split()), np.int).reshape(y_shape)
sh1 = "2 3"
arr1 = "8 7 7 14 4 6"
sh2 = "4 3"
arr2 = "5 5 1 5 2 6 3 3 9 1 4 6"
[[ 82  96 108  78]
 [ 96 114 108  66]]
=====================

sh1 = "2 3"
arr1 = "5 9 9 10 8 9"
sh2 = "3 4"
arr2 = "6 11 3 5 4 5 3 2 5 8 2 2"
x_shape = tuple(map(int, sh1.split()))
X = np.fromiter(map(int, arr1.split()), np.int).reshape(x_shape)
y_shape = tuple(map(int, sh2.split()))
Y = np.fromiter(map(int, arr2.split()), np.int).reshape(y_shape)
# here goes your solution; X and Y are already defined!
try:
    print(X.dot(Y.T))
except:
    print("matrix shapes do not match")
"""

from urllib.request import urlopen
f = urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')
gg = np.loadtxt(f, usecols=None, skiprows=1, delimiter=",")
                     # dtype={'names': ('a', 'b', 'c', 'd'),
                       #      'formats': ('f4', 'f4', 'f4', 'f4')})
x=gg.mean(axis=0)
print(x)