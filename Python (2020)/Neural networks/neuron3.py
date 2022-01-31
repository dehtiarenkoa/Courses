import numpy as np
import random
def sigmoid(x):
    """сигмоидальная функция, работает и с числами, и с векторами (поэлементно)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    """производная сигмоидальной функции, работает и с числами, и с векторами (поэлементно)"""
    return sigmoid(x) * (1 - sigmoid(x))


class Neuron:

    def __init__(self, weights, activation_function=sigmoid, activation_function_derivative=sigmoid_prime):
        """
        weights - вертикальный вектор весов нейрона формы (m, 1), weights[0][0] - смещение
        activation_function - активационная функция нейрона, сигмоидальная функция по умолчанию
        activation_function_derivative - производная активационной функции нейрона
        """

        assert weights.shape[1] == 1, "Incorrect weight shape"

        self.w = weights
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward_pass(self, single_input):
        """
        активационная функция логистического нейрона
        single_input - вектор входов формы (m, 1),
        первый элемент вектора single_input - единица (если вы хотите учитывать смещение)
        """

        result = 0
        for i in range(self.w.size):
            result += float(self.w[i] * single_input[i])
        return self.activation_function(result)

    def summatory(self, input_matrix):
        """
        Вычисляет результат сумматорной функции для каждого примера из input_matrix.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вектор значений сумматорной функции размера (n, 1).
        """
        # Этот метод необходимо реализовать

        return input_matrix.dot(self.w) #+1

    def activation(self, summatory_activation):
        """
        Вычисляет для каждого примера результат активационной функции,
        получив на вход вектор значений сумматорной функций
        summatory_activation - вектор размера (n, 1),
        где summatory_activation[i] - значение суммматорной функции для i-го примера.
        Возвращает вектор размера (n, 1), содержащий в i-й строке
        значение активационной функции для i-го примера.
        """
        # Этот метод необходимо реализовать

        return np.array([sigmoid(a) for a in summatory_activation], dtype=float)

    def vectorized_forward_pass(self, input_matrix):
        """
        Векторизованная активационная функция логистического нейрона.
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных.
        Возвращает вертикальный вектор размера (n, 1) с выходными активациями нейрона
        (элементы вектора - float)
        """
        return self.activation(self.summatory(input_matrix))

    def SGD(self, X, y, batch_size, learning_rate=0.1, eps=1e-6, max_steps=200):
        """
        Внешний цикл алгоритма градиентного спуска.
        X - матрица входных активаций (n, m)
        y - вектор правильных ответов (n, 1)

        learning_rate - константа скорости обучения
        batch_size - размер батча, на основании которого
        рассчитывается градиент и совершается один шаг алгоритма

        eps - критерий остановки номер один: если разница между значением целевой функции
        до и после обновления весов меньше eps - алгоритм останавливается.
        Вторым вариантом была бы проверка размера градиента, а не изменение функции,
        что будет работать лучше - неочевидно. В заданиях используйте первый подход.

        max_steps - критерий остановки номер два: если количество обновлений весов
        достигло max_steps, то алгоритм останавливается

        Метод возвращает 1, если отработал первый критерий остановки (спуск сошёлся)
        и 0, если второй (спуск не достиг минимума за отведённое время).
        """

        # Этот метод необходимо реализовать

        pass

    def update_mini_batch(self, X, y, learning_rate, eps):
        """
        X - матрица размера (batch_size, m)
        y - вектор правильных ответов размера (batch_size, 1)
        learning_rate - константа скорости обучения
        eps - критерий остановки номер один: если разница между значением целевой функции
        до и после обновления весов меньше eps - алгоритм останавливается.

        Рассчитывает градиент (не забывайте использовать подготовленные заранее внешние функции)
        и обновляет веса нейрона. Если ошибка изменилась меньше, чем на eps - возвращаем 1,
        иначе возвращаем 0.
        """
        # Этот метод необходимо реализовать

        result = 0
        error = J_quadratic(self, X, y)
        #print ("error1: ", error)
        nabla = compute_grad_analytically(self, X, y)
        self.w = self.w - learning_rate * nabla
        #print("self.w\n",self.w)
        error2 = J_quadratic(self, X, y)
        diff = abs(error - error2)
        #print("error2: ", error2)
        if diff < eps:
            result = 1
        return result

#it works, but i dont agreee, I like my below one variant
        result = 0
        while result != 1:
            error = J_quadratic(self, X, y)
            #print ("error1: ", error)
            nabla = compute_grad_analytically(self, X, y)
            self.w = self.w - learning_rate * nabla
            #print("self.w\n",self.w)
            error2 = J_quadratic(self, X, y)
            diff = abs(error - error2)
            #print("error2: ", error2)
            if diff < eps:
                result = 1
        return result
"""w:  [[-0.22264542]
 [-0.45730194]
 [ 0.65747502]
 [-0.28649335]
 [-0.43813098]]
"""

"""
it doesnt work i dont know why....
        result = 0
        while result != 1:
            error = J_quadratic(self, X, y)
            #print ("error1: ", error)
            nabla = compute_grad_analytically(self, X, y)
            self.w = self.w - learning_rate * nabla
            #print("self.w\n",self.w)
            error2 = J_quadratic(self, X, y)
            diff = abs(error - error2)
            #print("error2: ", error2)
            if diff < eps:
                result = 1
        return result
"""

def J_quadratic(neuron, X, y):
    """
    Оценивает значение квадратичной целевой функции.
    Всё как в лекции, никаких хитростей.

    neuron - нейрон, у которого есть метод vectorized_forward_pass, предсказывающий значения на выборке X
    X - матрица входных активаций (n, m)
    y - вектор правильных ответов (n, 1)

    Возвращает значение J (число)
    """

    assert y.shape[1] == 1, 'Incorrect y shape'

    return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)


def J_quadratic_derivative(y, y_hat):
    """
    Вычисляет вектор частных производных целевой функции по каждому из предсказаний.
    y_hat - вертикальный вектор предсказаний,
    y - вертикальный вектор правильных ответов,

    В данном случае функция смехотворно простая, но если мы захотим поэкспериментировать
    с целевыми функциями - полезно вынести эти вычисления в отдельный этап.

    Возвращает вектор значений производной целевой функции для каждого примера отдельно.
    """

    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'

    return (y_hat - y) / len(y)


def compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative):
    """
    Аналитическая производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для примеров из матрицы X
    J_prime - функция, считающая производные целевой функции по ответам

    Возвращает вектор размера (m, 1)
    """

    # Вычисляем активации
    # z - вектор результатов сумматорной функции нейрона на разных примерах

    z = neuron.summatory(X)
    y_hat = neuron.activation(z)

    # Вычисляем нужные нам частные производные
    dy_dyhat = J_prime(y, y_hat)
    dyhat_dz = neuron.activation_function_derivative(z)

    # осознайте эту строчку:
    dz_dw = X

    # а главное, эту:
    grad = ((dy_dyhat * dyhat_dz).T).dot(dz_dw)

    # можно было написать в два этапа. Осознайте, почему получается одно и то же
    # grad_matrix = dy_dyhat * dyhat_dz * dz_dw
    # grad = np.sum(, axis=0)

    # Сделаем из горизонтального вектора вертикальный
    grad = grad.T

    return grad


np.random.seed(42)
n = 10
m = 5

X = 20 * np.random.sample((n, m)) - 10
"""print("X: ",X)
X:  [[-2.50919762  9.01428613  4.63987884  1.97316968 -6.87962719]
 [-6.88010959 -8.83832776  7.32352292  2.02230023  4.16145156]
 [-9.58831011  9.39819704  6.64885282 -5.75321779 -6.36350066]
 [-6.3319098  -3.91515514  0.49512863 -1.36109963 -4.1754172 ]
 [ 2.23705789 -7.21012279 -4.15710703 -2.67276313 -0.87860032]
 [ 5.70351923 -6.00652436  0.28468877  1.84829138 -9.07099175]
 [ 2.15089704 -6.58951753 -8.69896814  8.97771075  9.31264066]
 [ 6.16794696 -3.90772462 -8.04655772  3.68466053 -1.19695013]
 [-7.5592353  -0.0964618  -9.31222958  8.18640804 -4.82440037]
 [ 3.25044569 -3.76577848  0.40136042  0.93420559 -6.30291089]]"""
y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]
#print("y: ",y)
w = 2 * np.random.random((m, 1)) - 1
"""
 To sample Unif[a, b), b > a multiply the output of random_sample by (b-a) and add a:
(b - a) * random_sample() + a
"""
print("w: ",w)
"""
[[-0.22264542]
 [-0.45730194]
 [ 0.65747502]
 [-0.28649335]
 [-0.43813098]]"""
neuron = Neuron(w)
g=neuron.update_mini_batch(X, y, 0.1, 1e-5)
print("neuron.w \n", neuron.w)
result = np.array([[-0.22368982], [-0.45599204], [ 0.65727411], [-0.28380677], [-0.43011026]])
print("neuron.w-result\n", neuron.w-result)
"""
[[ 0.0010444 ]
 [-0.0013099 ]
 [ 0.00020091]
 [-0.00268658]
 [-0.00802072]]
"""
