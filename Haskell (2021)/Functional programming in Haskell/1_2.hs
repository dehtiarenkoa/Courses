import Data.Function

{--
Before tasks:
https://stepik.org/lesson/8417/step/2?discussion=103521&unit=1555
Каков тип будет у flip id

https://stepik.org/lesson/8417/step/2?discussion=103521&reply=103669&unit=1555
(flip :: (a -> b -> c) -> b -> a -> c) (id :: d -> d)
// т.к. id это первый агрумент, то получаем соотношение на типы: d -> d = a -> (b -> c)
// со скобками: d -> d = a -> (b -> c),
// доп шаг: под учитывая, то, что на в правой и левой стороне одинаковый kind * -> *
//                    получаем, что каждая * должна быть равная соответсвующей (е
// d = b -> c, a = (b -> c)
// подставляем
(flip :: ((b->c) -> b -> c) -> b -> (b -> c) -> c) (id :: (b->c) -> (b->c)) =
flip id :: b -> (b -> c) -> c
```
проверяем:
ghci
GHCi, version 7.10.2: http://www.haskell.org/ghc/ :? for help
Prelude> :t flip id
flip id :: b -> (b -> c) -> c

https://stepik.org/lesson/8417/step/2?discussion=103521&reply=103771&unit=1555
Бонус. Если определить оператор "точка", как
(.) = flip id
то на Haskell можно писать в объектно-ориентированном стиле :)
"Hello world".reverse
2.sqrt
[1..5].map (\ x -> x * 2).filter (\ x -> x > 5)

--}

--2.1.3
{--


Напишите функцию трех аргументов getSecondFrom, 
полиморфную по каждому из них, которая полностью игнорирует первый и третий аргумент, 
а возвращает второй. Укажите ее тип.

GHCi> getSecondFrom True 'x' "Hello"
'x'
GHCi> getSecondFrom 'x' 42 True 
42

--}
getSecondFrom :: a->b->c->b
getSecondFrom x y z = y

--2.1.4
{--
Сколько разных всегда завершающихся функций с типом a -> a -> b -> a -> a можно реализовать?
Две функции одинаковой арности считаются разными, если существует набор значений их аргументов, 
на котором они дают разные результирующие значения.
--}

--1.2.7
{--
В модуле Data.Function определена полезная функция высшего порядка
on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
on op f x y = f x `op` f y
Она принимает четыре аргумента: бинарный оператор с однотипными аргументами (типа b), 
функцию f :: a -> b, возвращающую значение типа b, и два значения типа a. 
Функция on применяет f дважды к двум значениям типа a и передает результат в бинарный оператор.
Используя on можно, например, записать функцию суммирования квадратов аргументов так:
sumSquares = (+) `on` (^2)
Функция multSecond, перемножающая вторые элементы пар, реализована следующим образом
multSecond = g2 `on` h2
g2 = undefined
h2 = undefined
Напишите реализацию функций g2 и h2.
GHCi> multSecond ('A',2) ('E',7)
14
--}

multSecond = g2 `on` h2
g2 x y= x*y
h2 = snd

--2.1.9

{--
Реализуйте функцию on3, имеющую семантику, схожую с on, но принимающую в качестве 
первого аргумента трехместную функцию:
on3 :: (b -> b -> b -> c) -> (a -> b) -> a -> a -> a -> c
on3 op f x y z = undefined
Например, сумма квадратов трех чисел может быть записана с использованием on3 так
GHCi> let sum3squares = (\x y z -> x+y+z) `on3` (^2)
GHCi> sum3squares 1 2 3
14
--}

on3 :: (b -> b -> b -> c) -> (a -> b) -> a -> a -> a -> c
on3 op f x y z = op (f x) (f y) (f z)

--2.2.3

{--
Функция одной переменной doItYourself выбирает наибольшее из переданного ей аргумента и числа 42, 
затем возводит результат выбора в куб и, наконец, вычисляет логарифм по основанию 2 от полученного числа. 
Эта функция реализована в виде:
doItYourself = f1 . g1 . h1
Напишите реализации функций f1, g1 и h1. Постарайтесь сделать это в бесточечном стиле.
f1 = undefined
g1 = undefined
h1 = undefined
--}

doItYourself = f1 . g1 . h1
f1 = logBase 2
g1 x = x^3
h1 = max 42

--2.2.5
{--
Сколько разных всегда завершающихся функций с типом a -> (a,b) -> a -> (b,a,a) можно реализовать?
9
--
https://stepik.org/lesson/12398/step/5?discussion=2386721&unit=2828
Сколько двух-символьных комбинаций, можно построить из 10 цифр (от 0 до 9)? Ответ: 10^2
--}

--2.2.9

{--
В модуле Data.Tuple стандартной библиотеки определена функция swap :: (a,b) -> (b,a), 
переставляющая местами элементы пары:
GHCi> swap (1,'A')
('A',1)
Эта функция может быть выражена в виде:
swap = f (g h)
где f, g и h — некоторые идентификаторы из следующего набора:
curry uncurry flip (,) const
Укажите через запятую подходящую тройку f,g,h.
----
swap :: (a, b) -> (b, a)
const :: a -> b -> a
(,) :: a -> b -> (a, b)
flip :: (a -> b -> c) -> b -> a -> c
curry :: ((a, b) -> c) -> a -> b -> c
uncurry :: (a -> b -> c) -> (a, b) -> c

--
https://stepik.org/lesson/12398/step/9?discussion=118367&reply=118368&unit=2828
Функция curry позволяет функциям, работающим с двухэлементными кортежами (парами), 
работать как функциям, принимающим два аргумента. 
Функция uncurry позволяет функциям, принимающих два аргумента, 
работать с двухэлементными кортежами (парами).

Prelude Data.Tuple Data.Function> :type uncurry curry
uncurry curry :: ((a, b) -> c, a) -> b -> c
-- curry ((a, b) -> c) -> a -> b -> c
-- uncurry (a -> b -> c) -> (a, b) -> c
-- uncurry curry
-- d=((a, b) -> c)
-- e = b->c
-- curry= d->a->e
-- uncurry curry = uncurry d->a->e = (d, a)->e = ((a, b) -> c, a)->b->c
--done!

Prelude Data.Tuple Data.Function> :type curry uncurry
<interactive>:1:7: error:
    * Couldn't match type `(a, b)' with `a1 -> b1 -> c'
      Expected type: (a, b) -> (a1, b1) -> c
        Actual type: (a1 -> b1 -> c) -> (a1, b1) -> c
    * In the first argument of `curry', namely `uncurry'
      In the expression: curry uncurry

Prelude Data.Tuple Data.Function> :t curry (,) uncurry
curry (,) uncurry  :: b1 -> b2 -> (((a -> b3 -> c) -> (a, b3) -> c, b1), b2)

Prelude Data.Tuple Data.Function> :t (,) uncurry
(,) uncurry :: b1 -> ((a -> b2 -> c) -> (a, b2) -> c, b1)

Prelude Data.Tuple Data.Function> :t curry id
curry id :: a -> b -> (a, b)

++?? *Main> :t uncurry const
uncurry const :: (c, b) -> c
--const :: a -> b -> a
--uncurry :: (a -> b -> c) -> (a, b) -> c


Prelude Data.Tuple Data.Function> :t uncurry (flip const)
uncurry (flip const) :: (b, c) -> c

Prelude Data.Tuple Data.Function> :t flip const
flip const :: b -> c -> c

Prelude Data.Tuple Data.Function> :t flip curry
flip curry :: a -> ((a, b) -> c) -> b -> c

Prelude Data.Tuple Data.Function> :t uncurry flip
uncurry flip :: (a -> b -> c, b) -> a -> c

Prelude Data.Tuple Data.Function> :t uncurry (flip curry)
uncurry (flip curry) :: (a, (a, b) -> c) -> b -> c

Prelude Data.Tuple Data.Function> :t flip curry
flip curry :: a -> ((a, b) -> c) -> b -> c

Prelude Data.Tuple Data.Function> :t curry (uncurry flip)
curry (uncurry flip) :: (a -> b -> c) -> b -> a -> c

Prelude Data.Tuple Data.Function> :t (,) (uncurry flip)
(,) (uncurry flip) :: b1 -> ((a -> b2 -> c, b2) -> a -> c, b1)

Prelude Data.Tuple Data.Function> :t curry flip
<interactive>:1:7: error:
    * Couldn't match type `(a, b)' with `a1 -> b1 -> c'
      Expected type: (a, b) -> b1 -> a1 -> c
        Actual type: (a1 -> b1 -> c) -> b1 -> a1 -> c
    * In the first argument of `curry', namely `flip'
      In the expression: curry flip

+Prelude Data.Tuple Data.Function> :t curry const
curry const :: a -> b1 -> b2 -> (a, b1)

Prelude Data.Tuple Data.Function> :t curry (const flip)
curry (const flip) :: a1 -> b1 -> (a2 -> b2 -> c) -> b2 -> a2 -> c
*Main> :t swap
swap :: a1 -> b1 -> (a2 -> b2 -> c) -> b2 -> a2 -> c

Prelude Data.Tuple Data.Function> :t curry (flip const)
curry (flip const) :: a -> b -> c -> c
*Main> :t swap
swap :: a -> b -> c -> c

????? Prelude Data.Tuple Data.Function> :t flip (curry const)
flip (curry const) :: b1 -> a -> b2 -> (a, b1)  
*Main> :t swap
swap :: b1 -> a -> b2 -> (a, b1)

Prelude> :t (,) (flip curry)
(,) (flip curry) :: b1 -> (a -> ((a, b2) -> c) -> b2 -> c, b1)
*Main> :t swap
swap :: b1 -> (a -> ((a, b2) -> c) -> b2 -> c, b1)

Prelude> :t uncurry fst
uncurry fst :: ((b1 -> c, b2), b1) -> c
--uncurry :: (a -> b -> c) -> (a, b) -> c
--fst ::(a, b)->a
???

Prelude> :t curry fst
curry fst :: c -> b -> c

Prelude> :t fst
fst :: (a, b) -> a

Prelude> :t const uncurry
const uncurry :: b1 -> (a -> b2 -> c) -> (a, b2) -> c

Prelude> :t const curry
const curry :: b1 -> ((a, b2) -> c) -> a -> b2 -> c

Prelude> :t const flip
const flip :: b1 -> (a -> b2 -> c) -> b2 -> a -> c

Prelude> :t uncurry const flip
<interactive>:1:15: error:
    * Couldn't match expected type `(c, b1)'
                  with actual type `(a0 -> b0 -> c0) -> b0 -> a0 -> c0'
    * Probable cause: `flip' is applied to too few arguments
      In the second argument of `uncurry', namely `flip'
      In the expression: uncurry const flip

Prelude> :t uncurry (const flip)
uncurry (const flip) :: (b1, a -> b2 -> c) -> b2 -> a -> c

(,) const flip
  :: (a1 -> b1 -> a1, (a2 -> b2 -> c) -> b2 -> a2 -> c)

Prelude> :t (,) (const flip)
(,) (const flip)
  :: b1 -> (b2 -> (a -> b3 -> c) -> b3 -> a -> c, b1)
*Main> :t swap
swap :: b1 -> (b2 -> (a -> b3 -> c) -> b3 -> a -> c, b1)

Prelude> :t const ((,) flip)
const ((,) flip)
  :: b1 -> b2 -> ((a -> b3 -> c) -> b3 -> a -> c, b2)  
*Main> :t swap
swap :: b1 -> b2 -> ((a -> b3 -> c) -> b3 -> a -> c, b2)

Prelude> :t flip ((,) const)
<interactive>:1:7: error:
    * Couldn't match type `(a0 -> b0 -> a0, b1)' with `b -> c'
      Expected type: b1 -> b -> c
        Actual type: b1 -> (a0 -> b0 -> a0, b1)
    * Possible cause: `(,)' is applied to too many arguments
      In the first argument of `flip', namely `((,) const)'
      In the expression: flip ((,) const)

Prelude> :t (,) (flip curry)
(,) (flip curry) :: b1 -> (a -> ((a, b2) -> c) -> b2 -> c, b1)
*Main> :t swap
swap :: b1 -> (a -> ((a, b2) -> c) -> b2 -> c, b1)

*Main> :t const  (flip curry)
const  (flip curry) :: b1 -> a -> ((a, b2) -> c) -> b2 -> c
*Main> :t swap
swap :: b1 -> a -> ((a, b2) -> c) -> b2 -> c

++++
*Main> :t flip (,)
flip (,) :: b -> a -> (a, b)

+++++++++++++++++++++++++++++++++++++
*Main> :t uncurry (flip (,))
uncurry (flip (,)) :: (b, a) -> (a, b)
--}
cu :: ((a, b) -> c) -> a -> b -> c
cu = curry
un :: (a -> b -> c) -> (a, b) -> c
un = uncurry
fl :: (a -> b -> c) -> b -> a -> c
fl = flip
pa :: a -> b -> (a, b)
pa = (,)
co :: a -> b -> a
co = const

--fu = [fl, pa, co, un, cu]
f = un
g = fl
h = cu
swap = f (g h)
--res = swap (True, 3)
