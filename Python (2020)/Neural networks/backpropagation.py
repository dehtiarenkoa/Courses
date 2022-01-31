import numpy as np
n = 3
n_l = 4
np.random.seed(42)
deltas = np.random.random((n,n_l+1))
sums = np.random.random((n,n_l))
weights = np.random.random((n_l+1, n_l))

def get_error(deltas, sums, weights):
    delta_n = ((deltas.dot(weights))*sigmoid_prime(sums)).mean(axis = 0)
    return delta_n
def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))
def sigmoid_prime (sums):
    return np.array([[sigmoid(x)*(1-sigmoid(x)) for x in z] for z in sums])

result = get_error(deltas, sums, weights)
print(result)
"""
    Итак, мы знаем, как посчитать «назад» ошибку из l+1 l+1 l+1 слоя в l l l-й. Чтобы это знание не утекло куда подальше, давайте сразу его запрограммируем.
    Заодно вспомним различия между .dot и *.

Напишите функцию, которая, используя набор ошибок δl+1 \delta^{l+1} δl+1 для n n n примеров, матрицу весов Wl+1 W^{l+1} Wl+1 и набор значений сумматорной функции
на l l l-м шаге для этих примеров, возвращает значение ошибки δl \delta^l δl на l l l-м слое сети.

Сигнатура: get_error(deltas, sums, weights), где deltas — ndarray формы (nnn, nl+1n_{l+1}nl+1​), содержащий в iii-й строке значения ошибок для iii-го примера из в
ходных данных, sums — ndarray формы (nnn, nln_lnl​), содержащий в iii-й строке значения сумматорных функций нейронов lll-го слоя для iii-го примера из входных данных,
weights — ndarray формы (nl+1n_{l+1}nl+1​, nln_lnl​), содержащий веса для перехода между lll-м и l+1l+1l+1-м слоем сети.
Требуется вернуть вектор δl\delta^lδl — ndarray формы (nln_lnl​, 1);
мы не проверяем размер (форму) ответа, но это может помочь вам сориентироваться. Все нейроны в сети — сигмоидальные.
Функции sigmoid и sigmoid_prime уже определены.

Обратите внимание, в предыдущей задаче мы работали только с одним примером, а сейчас вам на вход подаётся несколько.
Не забудьте учесть этот факт и просуммировать всё, что нужно. И разделить тоже. Подсказка: J=1n∑i=1n12∣y^(i)−y(i)∣2  ⟹ 
 ∂J∂θ=1n∑i=1n∂∂θ(12∣y^(i)−y(i)∣2)
J = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left\vert\hat y ^{(i)} - y^{(i)}\right\vert^2 \implies \frac{\partial J}{\partial \theta} =
= \frac{1}{n}\sum_{i=1}^n \frac{\partial }{\partial \theta}\left(\frac{1}{2}\left\vert\hat y ^{(i)} - y^{(i)}\right\vert^2\right)
J=n1​∑i=1n​21​∣∣∣​y^​(i)−y(i)∣∣∣​2⟹∂θ∂J​=n1​∑i=1n​∂θ∂​(21​∣∣∣​y^​(i)−y(i)∣∣∣​2) для любого параметра θ \theta θ, который не число примеров.
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    
    def get_error(deltas, sums, weights):
    # here goes your code
    delta_n = ((deltas.dot(weights))*sigmoid_prime(sums)).mean(axis = 0)
    return delta_n"""
