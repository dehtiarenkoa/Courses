# it works!

import numpy as np

y = np.array([[1],[1],[0]])
x = np.array([[1, 0.3], [0.4, 0.5], [0.7, 0.8]])
w = np.array([[0], [0], [0]])
print("y: ", "\n", y)
print("x: ", x.shape)
print("x: ", "\n",x)
print("w: ", "\n", w)
ones = np.ones((3,1))
x=np.hstack((ones, x))
print("x: \n",x)
"""
y1=np.array([x.sum(axis=0)])
y1 = np.sum(x, axis=1)
print("y1: \n", y1)
y1=w.T*y1
y1=y1.T
print("y1: \n", y1)
"""
def alg(examples, w):

    def predict(a, w):
        #a = np.array([e])
        #e.reshape((3,1))
        print("a.shape: ", a.shape)
        # print("e.shape: ", e.shape)
        print("w.shape: ", w.shape)
        a= w*a
        s=sum(a.T)
        print("sum(a): ", s)
        print("a.shape: ", a.shape)
        return 1 if sum(a.T) > 0 else 0

    step =0
    perfect = False
    while not perfect:
        step += 1
        print("step: ", step)
        perfect = True
        #for e in examples:
        i=0
        while i<3:
            print("i: ", i)
            e = np.array([x[i,:]])
            print("e: ", e, type(e))

            p = predict(e, w)
            print("predict: \n", p)
            if p != y[i]:
                perfect = False
                if p == 0:
                    w = w + e
                    print("w = w + e")
                if p == 1:
                    w = w - e
                    print("w = w - e")
            print("w: \n", w)
            i+=1

    return w

print(alg(x, w.T))


"""
i=0
predicted = np.ones_like(y)
while i<predicted.shape[0]:
    predicted[i][0] = 0 if y1[i][0]>0 else 1
    print(f"predicted[{i}][0]: \n", predicted[i][0])
    i += 1

i=0
perfect = False
while not perfect or i<predicted.shape[0]:
    perfect=True
    print("i: ", i)
    if y[i][0]!=predicted[i][0]:
        perfect = False
        print("perfect: ", perfect)
        if predicted[i][0]==0:
            w[i][0]=w[i][0]+y1[i][0]
        perfect = True
        if predicted[i][0] == 1:
            w[i][0] = w[i][0] - y1[i][0]
        perfect = True
    i+=1
print("w: \n",w)

"""



