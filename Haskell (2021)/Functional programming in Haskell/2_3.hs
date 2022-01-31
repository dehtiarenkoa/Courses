-- 2.3.7
{-
Реализуйте класс типов Printable, предоставляющий один метод toString — функцию одной переменной, 
которая преобразует значение типа, являющегося представителем Printable, в строковое представление.
Сделайте типы данных Bool и () представителями этого класса типов, обеспечив следующее поведение:
GHCi> toString True
"true"
GHCi> toString False
"false"
GHCi> toString ()
"unit type"
-}


class Printable a where
  toString :: a -> [Char] 

instance Printable Bool where
    toString True = "true"
    toString False = "false"

instance Printable () where
    toString () = "unit type"


--2.3.9

{-
Сделайте тип пары представителем класса типов Printable, реализованного вами в предыдущей задаче, 
обеспечив следующее поведение:
GHCi> toString (False,())
"(false,unit type)"
GHCi> toString (True,False)
"(true,false)"
Примечание. Объявление класса типов Printable и представителей этого класса для типов () и  Bool
 заново реализовывать не надо — они присутствуют в программе, вызывающей ваш код.
-}

instance (Printable a, Printable b) => Printable (a, b) where
  toString (a,b) = "("++ toString a ++ "," ++ toString b ++ ")"


--2.4.3
{-
Пусть существуют два класса типов KnownToGork и KnownToMork, 
которые предоставляют методы stomp (stab) и doesEnrageGork (doesEnrageMork) соответственно:

class KnownToGork a where
    stomp :: a -> a
    doesEnrageGork :: a -> Bool

class KnownToMork a where
    stab :: a -> a
    doesEnrageMork :: a -> Bool

Класса типов KnownToGorkAndMork является расширением обоих этих классов, 
предоставляя дополнительно метод stompOrStab:

class (KnownToGork a, KnownToMork a) => KnownToGorkAndMork a where
    stompOrStab :: a -> a

Задайте реализацию по умолчанию метода stompOrStab, которая 
вызывает метод stomp, если переданное ему значение приводит в ярость Морка; 
вызывает stab, если оно приводит в ярость Горка 
и вызывает сначала stab, а потом stomp, если оно приводит в ярость их обоих. 
Если не происходит ничего из вышеперечисленного, 
метод должен возвращать переданный ему аргумент.
-}

class KnownToGork a where
  stomp :: a -> a
  doesEnrageGork :: a -> Bool

class KnownToMork a where
  stab :: a -> a
  doesEnrageMork :: a -> Bool

class (KnownToGork a, KnownToMork a) => KnownToGorkAndMork a where
  stompOrStab :: a -> a
  stompOrStab a
    | doesEnrageGork a && not (doesEnrageMork a) = stab a 
    | doesEnrageMork a && not (doesEnrageGork a) = stomp a 
    | doesEnrageMork a &&  doesEnrageGork a = stomp (stab a)
    | otherwise = a


-- 2.4.5
{-
Имея функцию ip = show a ++ show b ++ show c ++ show d 
определите значения a, b, c, d так, чтобы добиться следующего поведения:
GHCi> ip
"127.224.120.12"
-}
ip = show a ++ show b ++ show c ++ show d
a = 127.22
b = 4.12
c = 0.1
d = 2

--2.4.7
{-
Реализуйте класс типов
class SafeEnum a where
  ssucc :: a -> a
  spred :: a -> a
обе функции которого ведут себя как succ и pred стандартного класса Enum, 
однако являются тотальными, то есть не останавливаются с ошибкой 
на наибольшем и наименьшем значениях типа-перечисления соответственно, 
а обеспечивают циклическое поведение. 
Ваш класс должен быть расширением ряда классов типов стандартной библиотеки, 
так чтобы можно было написать реализацию по умолчанию его методов, 
позволяющую объявлять его представителей без необходимости писать 
какой бы то ни было код. Например, для типа Bool должно быть достаточно 
написать строку
instance SafeEnum Bool
и получить возможность вызывать
GHCi> ssucc False
True
GHCi> ssucc True
False
-}
class (Bounded a, Enum a, Ord a, Bounded a) => SafeEnum a where
  ssucc :: a -> a
  ssucc a  
      | a < maxBound = succ a 
      | otherwise = minBound
  --  where 
  --     maj = maxBound -- :: a
  --     mij = minBound -- :: a
  spred :: a -> a
  spred a
    | a > minBound = pred a
    | otherwise = maxBound
  --  where 
  --     maj = maxBound -- :: a
  --     mij = minBound -- :: a

  --2.4.9
  {-
  
Напишите функцию с сигнатурой:
avg :: Int -> Int -> Int -> Double
вычисляющую среднее значение переданных в нее аргументов:
GHCi> avg 3 4 8
5.0
  -}
avg :: Int -> Int -> Int -> Double
avg a b c = (fromIntegral a + fromIntegral b + fromIntegral c) / 3 

--2.5.3
{-
Предположим, что стандартные функции определены следующим образом:
id x = x
const x y = x
max x y = if x <= y then y else x
infixr 0 $
f $ x = f x
Сколько редексов имеется в следующем выражении
const $ const (4 + 5) $ max 42
Примечание. Мы определили шаг вычислений как подстановку тела функции 
вместо ее имени с заменой всех ее формальных параметров на фактически 
переданные ей выражения. Редексом при этом мы называем подвыражение, 
над которым можно осуществить подобный шаг.
=3
-}

--2.5.5
{-
Сколько шагов редукции потребуется, чтобы вычислить значение функции value, 
если используется ленивая стратегия вычислений с механизмом разделения?
bar x y z = x + y
foo a b = bar a a (a + b)
value = foo (3 * 10) (5 - 2)
Примечание. Подстановку тела функции value вместо value не считайте.
=4
-}

--2.5.11
{-
При вычислении каких из перечисленных ниже функций использование seq предотвратит 
нарастание количества невычисленных редексов при увеличении значения первого аргумента:

foo 0 x = x
foo n x = let x' = foo (n - 1) (x + 1)
          in x' `seq` x'

bar 0 f = f
bar x f = let f' = \a -> f (x + a)
              x' = x - 1
          in f' `seq` x' `seq` bar x' f'

baz 0 (x, y) = x + y
baz n (x, y) = let x' = x + 1
                   y' = y - 1
                   p  = (x', y')
                   n' = n - 1
               in p `seq` n' `seq` baz n' p

quux 0 (x, y) = x + y
quux n (x, y) = let x' = x + 1
                    y' = y - 1
                    p  = (x', y')
                    n' = n - 1
                in x' `seq` y' `seq` n' `seq` quux n' p

=quux
-}

