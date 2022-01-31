import urllib
from urllib import request
import numpy as np

# fname = input()  # read file name from stdin
fname = "https://stepic.org/media/attachments/lesson/16462/boston_houses.csv"
f = urllib.request.urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with
# here goes your solution
# "medv","crim","zn","chas","nox","rm","dis"
print(data.shape)
y = data[:,:1]  # "medv"
print("medv: ", y.shape)
X0 = data[:,1:]
print("X: ", X0.shape)
#print("X: ", X)

X= np.copy(data)
X[:,:1] = np.ones_like(y)
print("X: ", X)
step1 = X.T.dot(X)
step2 = np.linalg.inv(step1)
step3 = step2.dot(X.T)
b = step3.dot(y)
"""
print("b: ", type(b))
d = map(str, b)
print("d: ", type(d))
s = " ".join(d)
s1 = np.array2string(b, precision=6, separator=' ')
s2 = np.dtype('U')
s2 = b
print("b: ", type(b))
s3 = ("%.2f" % x for x in b)
s4 = b.astype('|S10')
print(*str(b))
for i in b:
    print(i, end=" ")
"""
print("!! !! ", (" ".join(map(str, b.flatten()))))
# right answer : -3.65580428507 -0.216395502369 0.0737305981755 4.41245057691 -25.4684487841 7.14320155075 -1.30108767765

"""
z = np.array([1,3])
print("z.shape: ", z.shape)
#z.shape:  (2,)
z1 = np.array([[1],[3]])
print("z1.shape: ", z1.shape)
#z1.shape:  (2, 1)
"""