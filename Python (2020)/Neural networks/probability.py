import math

p = 0.55
n = 50
q = 0.45
m_list = list(range(26, n+1))


def c(m):
    n = 50
    return math.factorial(n)/(math.factorial(m)*math.factorial(n-m))

p25 = c(25)*(p**25)*((1-p)**25)
print(p25)
p_list = [c(m)*(p**m)*((1-p)**(n-m)) for m in m_list]
s = sum(p_list)+0.5*p25
print(f"{s:.6}")

# 2.36301e-13
