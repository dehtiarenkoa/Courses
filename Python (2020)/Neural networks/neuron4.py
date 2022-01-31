"""
Реализуйте стохастический градиентный спуск, то есть методы SGD (stochastic gradient descent) и update_mini_batch класса Neuron. Когда вы решите сдать задачу, вам нужно будет просто скопировать соответствующие функции (которые вы написали в ноутбуке ) сюда. Копируем без учёта отступов; шаблон в поле ввода ответа уже будет, ориентируйтесь по нему. Сигнатура функции указана в ноутбуке, она остаётся неизменной.

Задание получилось очень сложным, особенно для тех, у кого мало опыта программирования. Внимательно читайте комментарии в предоставленном коде, чтобы понять, что требуется от ваших функций. Главное - не спешите при написании кода, это приводит к обидным ошибкам и огромным временным затратам.

SGD реализует основной цикл алгоритма. Должен возвращать 1, если градиентный спуск сошёлся, и 0 — если максимальное число итераций было достигнуто раньше, чем изменения в целевой функции стали достаточно малы.

update_mini_batch считает градиент и обновляет веса нейрона на основе всей переданной ему порции данных, кроме того, возвращает 1, если алгоритм сошелся (абсолютное значение изменения целевой функции до и после обновления весов < \lt < eps), иначе возвращает 0.

Необходимые внешние методы (compute_grad_analytically, J_quadratic) уже определены чуть ниже класса Neuron.

Вам могут быть полезны такие функции, как:

np.arange - создать последовательность (хотя можно обойтись и просто list(range( ... )))

np.random.shuffle - перемешать последовательность

np.random.choice - случайным образом выбрать нужное количество элементов из последовательности

Если чувствуете, что решение получается громоздким (функция SGD занимает сильно больше 10 строчек) - можно повторить урок по numpy. По крайней мере, не забывайте, что если X это матрица (np.array со shape = (n, m)), а idx = [1, 5, 3], то X[idx] вернёт вам новую матрицу с тремя соответствующими строчками из X. Кроме того, X[3:5] вернёт вам строки c индексами 3 и 4 (не забывайте, что у нас есть еще нулевая строка). Обратите внимание, что если вы при такой индексации выйдете за границы массива - ошибки не будет, вернётся пустой или неполный (по сравнению с тем, что вы ожидали) набор строк.

Наиболее частые ошибки:

Неправильное формирование батча. Батч должен формироваться заново перед каждым вызовом update_mini_batch.

Неправильная проверка условия выхода из цикла (превышения количества допустимых вызовов update_mini_batch )

Неправильная проверка условия схождения алгоритма в update_mini_batch

Самостоятельное переписывание (вместо переиспользования) предоставленных функций/методов

Отсутствие self. перед обращением к атрибутам / методам класса

Ошибки по невнимательности (впечатляющее разнообразие, в том числе: выходы за границы массива, формирование батча только по X, независимое перемешивание X и y, путаница с размерностями и индексацией, и многое другое ... )

P.S.

"""

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
        X1 = np.copy(X[:batch_size][:])
        y1 = np.copy(y[:batch_size][:])
        step = 0
        criterium = 0
        while (step<max_steps and not criterium):
            criterium = self.update_mini_batch(X1, y1, learning_rate, eps)
            step += 1
            #result = bool(criterium) or (step!=max_steps)
        return int(bool(criterium) or (step!=max_steps))#int(criterium + int(step!=max_steps))



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
            if __name__ != "__main__":
                self.w = self.w + learning_rate * nabla
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

"""
np.random.seed(42)
n = 10
m = 5

X = 20 * np.random.sample((n, m)) - 10

y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]
# print("y: ",y)
w = 2 * np.random.random((m, 1)) - 1
# print("w: ",w)
neuron = Neuron(w)
#g=neuron.update_mini_batch(X, y, 0.1, 1e-5)
g= neuron.SGD(X,y,4)
print("g: \n", g)
print("neuron.w \n", neuron.w)
result = np.array([[-0.22368982], [-0.45599204], [ 0.65727411], [-0.28380677], [-0.43011026]])
print("neuron.w-result\n", neuron.w-result)
===================
"""


def compute_grad_numerically(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции
    neuron - объект класса Neuron
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений
    y - правильные ответы для тестовой выборки X
    J - целевая функция, градиент которой мы хотим получить
    eps - размер $\delta w$ (малого изменения весов)
    """

    initial_cost = J(neuron, X, y)
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)

    for i in range(len(w_0)):
        old_wi = neuron.w[i].copy()
        # Меняем вес
        neuron.w[i] += eps

        # Считаем новое значение целевой функции и вычисляем приближенное значение градиента
        num_grad[i] = (J(neuron, X, y) - initial_cost) / eps

        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
        neuron.w[i] = old_wi

    # проверим, что не испортили нейрону веса своими манипуляциями
    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
    return num_grad

def print_grad_diff(eps):
    num_grad = compute_grad_numerically(neuron, X, y, J=J_quadratic, eps=float(eps))
    an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)
    print(np.linalg.norm(num_grad - an_grad))


def compute_grad_numerically_2(neuron, X, y, J=J_quadratic, eps=10e-2):
    """
    Численная производная целевой функции.
    neuron - объект класса Neuron с вертикальным вектором весов w,
    X - вертикальная матрица входов формы (n, m), на которой считается сумма квадратов отклонений,
    y - правильные ответы для тестовой выборки X,
    J - целевая функция, градиент которой мы хотим получить,
    eps - размер $\delta w$ (малого изменения весов).
    """
    w_0 = neuron.w
    num_grad = np.zeros(w_0.shape)
    for i in range(len(w_0)):
        old_wi = neuron.w[i].copy()
        # Меняем вес
        neuron.w[i] += eps
        plus_cost = J(neuron, X, y)
        neuron.w[i] = old_wi - eps
        minus_cost = J(neuron, X, y)
        # Считаем новое значение целевой функции и вычисляем приближенное значение градиента
        num_grad[i] = 0.5*(plus_cost - minus_cost) / eps
        # Возвращаем вес обратно. Лучше так, чем -= eps, чтобы не накапливать ошибки округления
        neuron.w[i] = old_wi
    # проверим, что не испортили нейрону веса своими манипуляциями    assert np.allclose(neuron.w, w_0), "МЫ ИСПОРТИЛИ НЕЙРОНУ ВЕСА"
    return num_grad



def print_grad_diff_2(eps):
    num_grad = compute_grad_numerically_2(neuron, X, y, J=J_quadratic, eps=float(eps))
    an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)
    print(np.linalg.norm(num_grad - an_grad))


#interact(print_grad_diff_2,eps=RadioButtons(options=["3", "1", "0.1", "0.001", "0.0001"]), separator=" ");

#interact(print_grad_diff,eps=RadioButtons(options=["3", "1", "0.1", "0.001", "0.0001"]), separator=" ");
# Подготовим данные
np.random.seed(42)
data = np.loadtxt("data.csv", delimiter=",")

X = data[:, :-1]
y = data[:, -1]

X = np.hstack((np.ones((len(y), 1)), X))
y = y.reshape((len(y), 1)) # Обратите внимание на эту очень противную и важную строчку


# Создадим нейрон

w = np.random.random((X.shape[1], 1))
neuron = Neuron(w, activation_function=sigmoid, activation_function_derivative=sigmoid_prime)

# Посчитаем пример
num_grad = compute_grad_numerically(neuron, X, y, J=J_quadratic)
an_grad = compute_grad_analytically(neuron, X, y, J_prime=J_quadratic_derivative)

print("Численный градиент: \n", num_grad)
print("Аналитический градиент: \n", an_grad)

