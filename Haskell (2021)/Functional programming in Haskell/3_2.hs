--{-# LANGUAGE InstanceSigs #-}

--3.2.3
-- Напишите функцию readDigits, принимающую строку и возвращающую пару строк.
-- Первый элемент пары содержит цифровой префикс исходной строки, а второй - ее оставшуюся часть.
-- GHCi> readDigits "365ads"
-- ("365","ads")
-- GHCi> readDigits "365"
-- ("365","")
-- В решении вам поможет функция isDigit из модуля Data.Char.
import Data.Char as Dc
import Data.Set (fromList, toList)
import Data.List

readDigits :: String -> (String, String)
readDigits [] = ([],[])
readDigits s = span Dc.isDigit s

--3.2.4
-- Реализуйте функцию filterDisj, принимающую два унарных предиката и список, 
-- и возвращающую список элементов, удовлетворяющих хотя бы одному из предикатов.
-- GHCi> filterDisj (< 10) odd [7,8,10,11,12]
-- [7,8,11]
filterDisj :: (a -> Bool) -> (a -> Bool) -> [a] -> [a]
filterDisj p1 p2 [] = []
filterDisj p1 p2 [a] = [a | p1 a || p2 a]
filterDisj p1 p2 (a : as)
 |p1 a || p2 a = a : filterDisj p1 p2 as
 |otherwise =filterDisj p1 p2 as

-- + https://stepik.org/lesson/12321/step/4?discussion=336938&thread=solutions&unit=2785
-- filterDisj :: (a -> Bool) -> (a -> Bool) -> [a] -> [a]
-- filterDisj p1 p2 = filter (\x -> p1 x || p2 x)

-- https://stepik.org/lesson/12321/step/4?discussion=1348707&thread=solutions&unit=2785
-- filterDisj :: (a -> Bool) -> (a -> Bool) -> [a] -> [a]
-- filterDisj p1 p2 x = [n | n <- x, p1 n || p2 n]

--3.2.5
-- Напишите реализацию функции qsort. Функция qsort должная принимать на вход список элементов 
-- и сортировать его в порядке возрастания с помощью сортировки Хоара: 
-- для какого-то элемента x изначального списка (обычно выбирают первый) 
-- делить список на элементы меньше и не меньше x, и потом запускаться рекурсивно на обеих частях.
-- GHCi> qsort [1,3,2,5]
-- [1,2,3,5]
-- Разрешается использовать только функции, доступные из библиотеки Prelude.
qsort :: Ord a => [a] -> [a]
qsort [] = []
qsort [a] = [a]
qsort (a : as) = qsort (filter (< a) as) ++ [a] ++ qsort (filter (>= a) as)

--3.2.7
-- Напишите функцию squares'n'cubes, принимающую список чисел,
-- и возвращающую список квадратов и кубов элементов исходного списка.
-- GHCi> squares'n'cubes [3,4,5]
-- [9,27,16,64,25,125]
squares'n'cubes :: Num a => [a] -> [a]
squares'n'cubes = concatMap (\x->[x^2,x^3])

--3.2.8
-- Воспользовавшись функциями map и concatMap, определите функцию perms, 
-- которая возвращает все перестановки, которые можно получить из данного списка, в любом порядке.
-- GHCi> perms [1,2,3]
-- [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
-- Считайте, что все элементы в списке уникальны, и что для пустого списка имеется одна перестановка.

-- perms :: [a] -> [[a]]
-- perms []= [[]]
-- perms [a, b] = [[a, b], [b, a]]
-- perms (a : as) = map (\x -> a:x) (perms as)
---- *Main> perms [1,2,3]
-- [[1,2,3],[1,3,2]]
-- ~>perms(1:[2,3]) = map (1:) (perms [2,3]) = map (1:) [[2,3],[3,2]] = [[1,2,3],[1,3,2]]
---- *Main> perms [1,2,3,4]
-- [[1,2,3,4],[1,2,4,3]]
-- ~>perms(1:[2,3,4]) = map (1:) (perms [2,3,4]) = map (1:) (map (2:) (perms [3,4])) = 
 -- = map (1:) (map (2:) [[3,4],[4,3]] ) = map (1:) [[2,3,4],[2,4,3]] = [[1,2,3,4],[1,2,4,3]]

-- perms :: [a] -> [[a]]
-- perms [] = [[]]
-- perms [a, b] = [[a, b], [b, a]]
-- perms (a : as) = map (a :) (perms as) ++ map (++[a]) (perms as)
---- * Main> perms [1,2,3]
-- [[1,2,3],[1,3,2],[2,3,1],[3,2,1]]
-- ~> perms (1 : [2,3]) = map (1 :) (perms [2,3]) ++ map (++[1]) (perms [2,3]) =
 -- = map (1:) [[2,3][3,2]] ++ map (++[1]) [[2,3][3,2]] = [[1,2,3],[1,3,2],[2,3,1],[3,2,1]]
---- *Main> perms [1,2,3,4]
-- [[1,2,3,4],[1,2,4,3],[1,3,4,2],[1,4,3,2],[2,3,4,1],[2,4,3,1],[3,4,2,1],[4,3,2,1]]
-- ~> map (1 :) (perms [2,3,4])) ++ map (++[1]) (perms [2,3,4]) = 
 -- = map (1 :) (map (2 :)(perms [3,4])) ++ map (++[1]) (map (2 :)(perms [3,4])) =
 -- = map (1 :) (map (2 :)[[3,4][4,3]]++ map (++[2])[[3,4][4,3]]) ++ map (++[1]) (map (2 :)[[3,4][4,3]]++ map (++[2])[[3,4][4,3]]) =
 -- = map (1 :) ([[2,3,4][2,4,3]]++[[3,4,2][4,3,2]]) ++ map (++[1]) ([[2,3,4][2,4,3]]++[[3,4,2][4,3,2]]) =
 -- = [[1,2,3,4][1,2,4,3][1,3,4,2][1,4,3,2]]++[[2,3,4,1][2,4,3,1][3,4,2,1][4,3,2,1]] =
 -- = [[1,2,3,4],[1,2,4,3],[1,3,4,2],[1,4,3,2],[2,3,4,1],[2,4,3,1],[3,4,2,1],[4,3,2,1]]


-- perms :: [a] -> [[a]]
-- perms [] = [[]]
-- perms [a, b] = [[a, b], [b, a]]
-- perms (a : as) 
--  |length as == 2 = map (a :) (perms as) 
--  |otherwise = map (a :) (perms as) ++ map (++[a]) (perms as)
-- *Main> perms [1,2,3]
-- [[1,2,3],[1,3,2]]
-- *Main> perms [1,2,3,4]
-- [[1,2,3,4],[1,2,4,3],[2,3,4,1],[2,4,3,1]]

-- perms [] = [[]]
-- perms [a, b] = [[a, b], [b, a]]
-- --d = [[]]
-- --f a b [d0] = [(b:a:[d0])++
-- perms (a:as) = f [a] [] [as] where 
--  -- f :: [t] [t] [t] -> [[t]]
--   f _ b [] = [[]]
--   --f [a] b [] = [[a]]
--   f [a] b d = map (\x -> b ++ [a] ++ x) d ++ f (head d) (b ++ [a]) (tail d)


-- perms :: [a] -> [[a]]
-- perms []= [[]]
-- perms [a, b] = [[a, b], [b, a]]
-- perms [a, b, c] = let
--   f [a, b, c] = [a : head (perms [b, c]), a : last (perms [b, c])]
--   in f [a, b, c] ++ f [ b, a, c] ++ f [c, a, b]
-- perms (a : as) = map (a :) (perms as) ++ map ((head as):) (perms (a:(tail as)))
-- *Main> perms [1,2,3]
-- [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
-- *Main> perms [1,2,3,4]
-- [[1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],[2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],
-- [2,4,1,3],[2,4,3,1]]
-- *Main> perms [1,2,3,4,5]
-- [[1,2,3,4,5],[1,2,3,5,4],[1,2,4,3,5],[1,2,4,5,3],[1,2,5,3,4],[1,2,5,4,3],[1,3,2,4,5],[1,3,2,5,4],[1,3,4,2,5],
-- [1,3,4,5,2],[1,3,5,2,4],[1,3,5,4,2],[2,1,3,4,5],[2,1,3,5,4],[2,1,4,3,5],[2,1,4,5,3],[2,1,5,3,4],[2,1,5,4,3],
-- [2,3,1,4,5],[2,3,1,5,4],[2,3,4,1,5],[2,3,4,5,1],[2,3,5,1,4],[2,3,5,4,1]]


-- perms :: [a] -> [[a]]
-- perms [] = [[]]
-- perms [a, b] = [[a, b], [b, a]]
-- perms [a, b, c] =
--   let f [a, b, c] = [a : head (perms [b, c]), a : last (perms [b, c])]
--    in f [a, b, c] ++ f [b, a, c] ++ f [c, a, b]
-- perms (a : as) =  let
--     fu (a : as) b
--      | length as>1 = map (a :) (perms (b++as)) ++ fu (head as : tail as) (b++[a])
--      | length as==1 = map (a :) (perms (b ++ as)) ++ fu [head as] (b ++ [a])
--      | otherwise =  map (a :) (perms b) ++ fu [] (b++[a])
--   in fu (a : as) []
-- *Main> perms [1,2,3,4,5]
-- [[1,2,3,4,5],[1,2,3,5,4],[1,2,4,3,5],[1,2,4,5,3],[1,2,5,3,4],[1,2,5,4,3],[1,3,2,4,5],[1,3,2,5,4],[1,3,4,2,5],
-- [1,3,4,5,2],[1,3,5,2,4],[1,3,5,4,2],[1,4,2,3,5],[1,4,2,5,3],[1,4,3,2,5],[1,4,3,5,2],[1,4,5,2,3],[1,4,5,3,2],
-- [1,5,2,3,4],[1,5,2,4,3],[1,5,3,2,4],[1,5,3,4,2],[1,5,4,2,3],[1,5,4,3,2]*** Exception: 3_2.hs:
-- (139,5)-(142,57): Non-exhaustive patterns in function fu
-- *Main> perms [1,2,3,4]
-- [[1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],[2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],[2,4,1,3],
-- [2,4,3,1],[3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],[4,1,2,3],[4,1,3,2],[4,2,1,3],[4,2,3,1],
-- [4,3,1,2],[4,3,2,1]*** Exception: 3_2.hs:(139,5)-(142,57)


-- WORKS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
perms :: [a] -> [[a]]
perms [] = [[]]
perms [a, b] = [[a, b], [b, a]]
perms [a, b, c] =
  let f [a, b, c] = [a : head (perms [b, c]), a : last (perms [b, c])]
   in f [a, b, c] ++ f [b, a, c] ++ f [c, a, b]
perms (a : as) =  let
    fu [] b = []
    fu (a : as) b = map (a :) (perms (b ++ as)) ++ fu as (b ++ [a])
  in fu (a : as) []

-- *Main> perms [1,2,3,4]
-- [[1,2,3,4],[1,2,4,3],[1,3,2,4],[1,3,4,2],[1,4,2,3],[1,4,3,2],[2,1,3,4],[2,1,4,3],[2,3,1,4],[2,3,4,1],
-- [2,4,1,3],[2,4,3,1],[3,1,2,4],[3,1,4,2],[3,2,1,4],[3,2,4,1],[3,4,1,2],[3,4,2,1],[4,1,2,3],[4,1,3,2],
-- [4,2,1,3],[4,2,3,1],[4,3,1,2],[4,3,2,1]]
-- *Main> perms [1,2,3,4,5]
-- [[1,2,3,4,5],[1,2,3,5,4],[1,2,4,3,5],[1,2,4,5,3],[1,2,5,3,4],[1,2,5,4,3],[1,3,2,4,5],[1,3,2,5,4],
-- [1,3,4,2,5],[1,3,4,5,2],[1,3,5,2,4],[1,3,5,4,2],[1,4,2,3,5],[1,4,2,5,3],[1,4,3,2,5],[1,4,3,5,2],
-- [1,4,5,2,3],[1,4,5,3,2],[1,5,2,3,4],[1,5,2,4,3],[1,5,3,2,4],[1,5,3,4,2],[1,5,4,2,3],[1,5,4,3,2],
-- [2,1,3,4,5],[2,1,3,5,4],[2,1,4,3,5],[2,1,4,5,3],[2,1,5,3,4],[2,1,5,4,3],[2,3,1,4,5],[2,3,1,5,4],
-- [2,3,4,1,5],[2,3,4,5,1],[2,3,5,1,4],[2,3,5,4,1],[2,4,1,3,5],[2,4,1,5,3],[2,4,3,1,5],[2,4,3,5,1],
-- [2,4,5,1,3],[2,4,5,3,1],[2,5,1,3,4],[2,5,1,4,3],[2,5,3,1,4],[2,5,3,4,1],[2,5,4,1,3],[2,5,4,3,1],
-- [3,1,2,4,5],[3,1,2,5,4],[3,1,4,2,5],[3,1,4,5,2],[3,1,5,2,4],[3,1,5,4,2],[3,2,1,4,5],[3,2,1,5,4],
-- [3,2,4,1,5],[3,2,4,5,1],[3,2,5,1,4],[3,2,5,4,1],[3,4,1,2,5],[3,4,1,5,2],[3,4,2,1,5],[3,4,2,5,1],
-- [3,4,5,1,2],[3,4,5,2,1],[3,5,1,2,4],[3,5,1,4,2],[3,5,2,1,4],[3,5,2,4,1],[3,5,4,1,2],[3,5,4,2,1],
-- [4,1,2,3,5],[4,1,2,5,3],[4,1,3,2,5],[4,1,3,5,2],[4,1,5,2,3],[4,1,5,3,2],[4,2,1,3,5],[4,2,1,5,3],
-- [4,2,3,1,5],[4,2,3,5,1],[4,2,5,1,3],[4,2,5,3,1],[4,3,1,2,5],[4,3,1,5,2],[4,3,2,1,5],[4,3,2,5,1],
-- [4,3,5,1,2],[4,3,5,2,1],[4,5,1,2,3],[4,5,1,3,2],[4,5,2,1,3],[4,5,2,3,1],[4,5,3,1,2],[4,5,3,2,1],
-- [5,1,2,3,4],[5,1,2,4,3],[5,1,3,2,4],[5,1,3,4,2],[5,1,4,2,3],[5,1,4,3,2],[5,2,1,3,4],[5,2,1,4,3],
-- [5,2,3,1,4],[5,2,3,4,1],[5,2,4,1,3],[5,2,4,3,1],[5,3,1,2,4],[5,3,1,4,2],[5,3,2,1,4],[5,3,2,4,1],
-- [5,3,4,1,2],[5,3,4,2,1],[5,4,1,2,3],[5,4,1,3,2],[5,4,2,1,3],[5,4,2,3,1],[5,4,3,1,2],[5,4,3,2,1]]

--https://stepik.org/lesson/12321/step/8?discussion=337724&thread=solutions&unit=2785
-- perms :: [a] -> [[a]]
-- perms [] = [[]]
-- perms [x] = [[x]]
-- perms (x : xs) = concatMap (insertElem x) (perms xs)
--   where
--     insertElem x [] = [[x]]
--     insertElem x yss@(y : ys) = (x : yss) : map (y :) (insertElem x ys)

--https://stepik.org/lesson/12321/step/8?discussion=517672&thread=solutions&unit=2785
-- perms :: [a] -> [[a]]
-- perms [] = [[]]
-- perms x = concatMap (\n -> map ((x !! n) :) (perms (take n x ++ drop (n + 1) x))) [0 .. (length x - 1)]

--3.2.10
{-
Реализуйте функцию delAllUpper, удаляющую из текста все слова, целиком состоящие из символов в верхнем регистре. 
Предполагается, что текст состоит только из символов алфавита и пробелов, знаки пунктуации, цифры и т.п. отсутствуют.
GHCi> delAllUpper "Abc IS not ABC"
"Abc not"
Постарайтесь реализовать эту функцию как цепочку композиций, аналогично revWords из предыдущего видео.
-}
-- import Data.Char
delAllUpper :: String -> String
--delAllUpper s = unwords (filter (all isUpper) (words s)) 
--delAllUpper s = unwords ( filter (not.(all isUpper)) (words s) ) --- Works!
delAllUpper = unwords . filter (not . all isUpper) . words -- Works!!!

--https://stepik.org/lesson/12321/step/10?discussion=477997&thread=solutions&unit=2785
-- delAllUpper = unwords . filter (any isLower) . words


--3.2.12
-- Напишите функцию max3, которой передаются три списка одинаковой длины и 
--которая возвращает список той же длины, содержащий на k-ой позиции наибольшее 
--значение из величин на этой позиции в списках-аргументах.
-- GHCi> max3 [7,2,9] [3,6,8] [1,8,10]
-- [7,8,10]
-- GHCi> max3 "AXZ" "YDW" "MLK"
-- "YXZ"
max3 :: Ord a => [a] -> [a] -> [a] -> [a]
max3 [] [] [] = []
max3 (e : es) (b : bs) (c : cs) = maximum [e, b, c] : max3 es bs cs -- Works!!!!
--max3 (e : es) (b : bs) (c : cs) = zipWith3 maximum [e, b, c] (e : es) (b : bs) (c : cs) : max3 es bs cs

--https://stepik.org/lesson/12321/step/12?discussion=438331&thread=solutions&unit=2785
-- max3 :: Ord a => [a] -> [a] -> [a] -> [a]
-- max3 = zipWith3 ((max .) . max)

--https://stepik.org/lesson/12321/step/12?discussion=870846&thread=solutions&unit=2785
-- max3 :: Ord a => [a] -> [a] -> [a] -> [a]
-- max3 = zipWith3 (\a b c -> a `max` b `max` c)

--https://stepik.org/lesson/12321/step/12?discussion=999455&thread=solutions&unit=2785
-- max3 :: Ord a => [a] -> [a] -> [a] -> [a]
-- max3 x = zipWith max . zipWith max x

--https://stepik.org/lesson/12321/step/12?discussion=1773020&thread=solutions&unit=2785
-- max3 :: Ord a => [a] -> [a] -> [a] -> [a]
-- max3 = zipWith3 (\x y z -> maximum [x, y, z])

--3.3.2
{-
Реализуйте c использованием функции zipWith функцию fibStream, возвращающую бесконечный список чисел Фибоначчи.
GHCi> take 10 $ fibStream
[0,1,1,2,3,5,8,13,21,34]
-}

fibStream :: [Integer]
fibStream = 0 : 1 : zipWith (+) fibStream (tail fibStream)

--https://coderoad.ru/18157582/Haskell-%D0%93%D0%B5%D0%BD%D0%B5%D1%80%D0%B0%D1%82%D0%BE%D1%80-%D0%A1%D0%BF%D0%B8%D1%81%D0%BA%D0%BE%D0%B2

-- fibStream = let 
--   a = 0 : 1: 1: [last a+ last (init a)]
--   in 0:1:zipWith (+) a (tail a)

--fbs :: [Integer]
--fbs = 0:1:[last fbs]

--nat = 0:[last nat+1]

--3.3.5
{-
Предположим, что функция repeat, была бы определена следующим образом:
repeat = iterate repeatHelper
определите, как должна выглядеть функция repeatHelper.
-}
repeat = iterate repeatHelper
repeatHelper x =  x
--iterate f x = x : iterate f (f x)

--3.3.7
{-
Пусть задан тип Odd нечетных чисел следующим образом:
data Odd = Odd Integer 
  deriving (Eq, Show)
Сделайте этот тип представителем класса типов Enum.
GHCi> succ $ Odd (-100000000000003)
Odd (-100000000000001)
Конструкции с четным аргументом, типа Odd 2, считаются недопустимыми и не тестируются. Примечание. 
Мы еще не знакомились с объявлениями пользовательских типов данных, однако, скорее всего, 
приведенное объявление не вызовет сложностей. Здесь объявляется тип данных Odd с конструктором Odd. 
Фактически это простая упаковка для типа Integer. Часть deriving (Eq, Show) указывает компилятору, 
чтобы он автоматически сгенерировал представителей соответствующих классов типов для нашего типа 
(такая возможность имеется для ряда стандартных классов типов). Значения типа Odd можно конструировать 
следующим образом:
GHCi> let x = Odd 33
GHCi> x
Odd 33
и использовать конструктор данных Odd в сопоставлении с образцом:
addEven :: Odd -> Integer -> Odd
addEven (Odd n) m | m `mod` 2 == 0 = Odd (n + m)
                  | otherwise      = error "addEven: second parameter cannot be odd"
-}

--https://stepik.org/lesson/8328/step/7?discussion=344613&unit=1476
-- Выписала для быстрой самопроверки, может кому пригодится.
-- test0 = succ (Odd 1) == (Odd 3)
-- test1 = pred (Odd 3) == (Odd 1)
-- -- enumFrom
-- test2 = (take 3 $ [Odd 1 ..]) == [Odd 1,Odd 3,Odd 5]
-- -- enumFromTo
-- -- -- По возрастанию
-- test3 = (take 3 $ [Odd 1..Odd 7]) == [Odd 1,Odd 3,Odd 5]
-- -- -- По убыванию
-- test4 = (take 3 $ [Odd 7..Odd 1]) == []
-- -- enumFromThen
-- -- -- По возрастанию
-- test5 = (take 3 $ [Odd 1, Odd 3 ..]) == [Odd 1,Odd 3,Odd 5]
-- -- -- По убыванию
-- test6 = (take 3 $ [Odd 3, Odd 1 ..]) == [Odd 3,Odd 1,Odd (-1)]
-- -- enumFromThenTo
-- -- -- По возрастанию
-- test7 =([Odd 1, Odd 5 .. Odd 7]) == [Odd 1,Odd 5]
-- -- -- По убыванию
-- test8 =([Odd 7, Odd 5 .. Odd 1]) == [Odd 7,Odd 5,Odd 3,Odd 1]
-- -- -- x1 < x3 && x1 > x2
-- test9 =([Odd 7, Odd 5 .. Odd 11]) == []
-- -- -- x1 > x3 && x1 < x2
-- test10 =([Odd 3, Odd 5 .. Odd 1]) == []
-- allTests = zip [0..] [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10]


data Odd = Odd Integer
 deriving (Eq,Show)
instance Enum Odd where
   --succ, pred ::(Num a) => a -> a
   succ (Odd x)= Odd (x + 2)
   pred (Odd x) = Odd (x - 2) --if x > 2 then Odd (x - 2) else error "pred error"
  --toEnum :: Integer -> a
  -- fromEnum :: a -> Int
   --enumFrom :: Num a => a -> [a] -- [n..]
   enumFrom (Odd x) = map Odd [x1, (x1+2) ..] where
     x1 = if odd x then x else x + 1
   --enumFromThen :: Num a => a -> a -> [a] -- [n,n’..]
   enumFromThen (Odd x) (Odd y) = map Odd [x1, y1 ..]
     where
       x1 = if odd x then x else x+1
       y1 = if odd y then y else y+1
   --enumFromTo :: Num a => a -> a -> [a] -- [n..m]
   enumFromTo (Odd x) (Odd y) = map Odd [x1, (x+2) .. y1]
    where
      x1 = if odd x then x else x + 1
      y1 = if odd y then y else y - 1
   --enumFromThenTo :: Num a => a -> a -> a -> [a] -- [n,n’..m]
   enumFromThenTo (Odd x1) (Odd x2) (Odd y) = map Odd [x01, x02 .. y1]
    where
      x01 = if odd x1 then x1 else x1 + 1
      x02 = if odd x2 then x2 else x2 + 1
      y1 = if odd y then y else y -1

  -- toEnum x = if odd x then Odd (toInteger x) else error "error d"
  -- fromEnum x = integerToInt (toInteger x) where
  --     integerToInt :: Integer -> Int
  --     integerToInt = fromIntegral
  --succ 1 = 3

  -- succ c = Odd (toInteger1 c + 2) --toEnum . (+ 2) . fromEnum
  --  where
  --     toInteger1 :: Odd -> Integer
  --     toInteger1 x =  x:: Integer

  --  enumFrom x = map toEnum [fromEnum x1 ..] where
  --    x1 = if odd (fromEnum x) then x else pred x
  -- enumFromThen x y = map toEnum [x1, fromEnum y ..]  where
  --    --y1 = if odd (fromEnum y) then fromEnum y else fromEnum (pred y)
  --   x1 = if odd (fromEnum x) then fromEnum x else fromEnum (pred x)

test0 = succ (Odd 1) == (Odd 3)
test1 = pred (Odd 3) == (Odd 1)
-- enumFrom
test2 = (take 3 $ [Odd 1 ..]) == [Odd 1, Odd 3, Odd 5]
-- enumFromTo
-- -- По возрастанию
test3 = (take 3 $ [Odd 1 .. Odd 7]) == [Odd 1, Odd 3, Odd 5]
-- -- По убыванию
test4 = (take 3 $ [Odd 7 .. Odd 1]) == []
-- enumFromThen
-- -- По возрастанию
test5 = (take 3 $ [Odd 1, Odd 3 ..]) == [Odd 1, Odd 3, Odd 5]
-- -- По убыванию
test6 = (take 3 $ [Odd 3, Odd 1 ..]) == [Odd 3, Odd 1, Odd (-1)]
-- enumFromThenTo
-- -- По возрастанию
test7 = ([Odd 1, Odd 5 .. Odd 7]) == [Odd 1, Odd 5]
-- -- По убыванию
test8 = ([Odd 7, Odd 5 .. Odd 1]) == [Odd 7, Odd 5, Odd 3, Odd 1]
-- -- x1 < x3 && x1 > x2
test9 = ([Odd 7, Odd 5 .. Odd 11]) == []
-- -- x1 > x3 && x1 < x2
test10 = ([Odd 3, Odd 5 .. Odd 1]) == []
allTests = zip [0 ..] [test0, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10]
{-
class Enum a where
  -- | the successor of a value.  For numeric types, 'succ' adds 1.
  succ :: a -> a

  -- | the predecessor of a value.  For numeric types, 'pred' subtracts 1.
  pred :: a -> a

  -- | Convert from an 'Int'.
  toEnum :: Int -> a

  -- | Convert to an 'Int'.
  -- It is implementation-dependent what 'fromEnum' returns when
  -- applied to a value that is too large to fit in an 'Int'.
  fromEnum :: a -> Int

  -- | Used in Haskell's translation of @[n..]@ with @[n..] = enumFrom n@,
  --   a possible implementation being @enumFrom n = n : enumFrom (succ n)@.
  --   For example:
  --
  --     * @enumFrom 4 :: [Integer] = [4,5,6,7,...]@
  --     * @enumFrom 6 :: [Int] = [6,7,8,9,...,maxBound :: Int]@
  enumFrom :: a -> [a]

  -- | Used in Haskell's translation of @[n,n'..]@
  --   with @[n,n'..] = enumFromThen n n'@, a possible implementation being
  --   @enumFromThen n n' = n : n' : worker (f x) (f x n')@,
  --   @worker s v = v : worker s (s v)@, @x = fromEnum n' - fromEnum n@ and
  --   @f n y
  --     | n > 0 = f (n - 1) (succ y)
  --     | n < 0 = f (n + 1) (pred y)
  --     | otherwise = y@
  --   For example:
  --
  --     * @enumFromThen 4 6 :: [Integer] = [4,6,8,10...]@
  --     * @enumFromThen 6 2 :: [Int] = [6,2,-2,-6,...,minBound :: Int]@
  enumFromThen :: a -> a -> [a]

  -- | Used in Haskell's translation of @[n..m]@ with
  --   @[n..m] = enumFromTo n m@, a possible implementation being
  --   @enumFromTo n m
  --      | n <= m = n : enumFromTo (succ n) m
  --      | otherwise = []@.
  --   For example:
  --
  --     * @enumFromTo 6 10 :: [Int] = [6,7,8,9,10]@
  --     * @enumFromTo 42 1 :: [Integer] = []@
  enumFromTo :: a -> a -> [a]

  -- | Used in Haskell's translation of @[n,n'..m]@ with
  --   @[n,n'..m] = enumFromThenTo n n' m@, a possible implementation
  --   being @enumFromThenTo n n' m = worker (f x) (c x) n m@,
  --   @x = fromEnum n' - fromEnum n@, @c x = bool (>=) (<=) (x > 0)@
  --   @f n y
  --      | n > 0 = f (n - 1) (succ y)
  --      | n < 0 = f (n + 1) (pred y)
  --      | otherwise = y@ and
  --   @worker s c v m
  --      | c v m = v : worker s c (s v) m
  --      | otherwise = []@
  --   For example:
  --
  --     * @enumFromThenTo 4 2 -6 :: [Integer] = [4,2,0,-2,-4,-6]@
  --     * @enumFromThenTo 6 8 2 :: [Int] = []@
  enumFromThenTo :: a -> a -> a -> [a]

  succ = toEnum . (+ 1) . fromEnum

  pred = toEnum . (subtract 1) . fromEnum

  -- See Note [Stable Unfolding for list producers]
  {-# INLINEABLE enumFrom #-}
  enumFrom x = map toEnum [fromEnum x ..]

  -- See Note [Stable Unfolding for list producers]
  {-# INLINEABLE enumFromThen #-}
  enumFromThen x y = map toEnum [fromEnum x, fromEnum y ..]

  -- See Note [Stable Unfolding for list producers]
  {-# INLINEABLE enumFromTo #-}
  enumFromTo x y = map toEnum [fromEnum x .. fromEnum y]

  -- See Note [Stable Unfolding for list producers]
  {-# INLINEABLE enumFromThenTo #-}
  enumFromThenTo x1 x2 y = map toEnum [fromEnum x1, fromEnum x2 .. fromEnum y]
-}

--https://stepik.org/lesson/8328/step/7?discussion=339292&thread=solutions&unit=1476
-- instance Enum Odd where
--   toEnum i = Odd (toInteger i)
--   fromEnum (Odd n) = fromEnum n

--   succ (Odd n) = Odd (n + 2)
--   pred (Odd n) = Odd (n -2)

--   enumFrom (Odd n) = map Odd [n, n + 2 ..]
--   enumFromTo (Odd n) (Odd m) = map Odd [n, n + 2 .. m]
--   enumFromThen (Odd n) (Odd n') = map Odd [n, n' ..]
--   enumFromThenTo (Odd n) (Odd n') (Odd m) = map Odd [n, n' .. m]

--3.3.9
{-
Пусть есть список положительных достоинств монет coins, отсортированный по возрастанию. 
Воспользовавшись механизмом генераторов списков, напишите функцию change, которая разбивает 
переданную ей положительную сумму денег на монеты достоинств из списка coins всеми возможными способами. 
Например, если coins = [2, 3, 7]:
GHCi> change 7
[[2,2,3],[2,3,2],[3,2,2],[7]]
Примечание. Порядок монет в каждом разбиении имеет значение, то есть наборы [2,2,3] и [2,3,2] — различаются.
Список coins определять не надо.
-}

--https://stepik.org/lesson/8328/step/9?discussion=806761&unit=1476
-- Почувствуйте разницу:

-- * Main> [ x:y | x <- [1,2,3], y <- [] ]

-- []

-- * Main> [ x:y | x <- [1,2,3], y <- [[]] ]

-- [[1],[2],[3]]

--https://stepik.org/lesson/8328/step/9?discussion=112606&reply=112974&unit=1476
-- "Could not deduce (a1 ~ Integer)
--   from the context (Ord a, Num a)
--     bound by the type signature for
--               change :: (Ord a, Num a) => a -> [[a]]    ......"
-- Проблема в том, что, когда вы просто определяете `coins` как список чисел в GHCi
-- (если у вас GHC достаточно старой версии), вы получаете список не произвольного числового типа,
-- а именно список `Integer`. Почему так происходит — отдельная долгая (и довольно непонятная) история.
-- Суть в том, что вам надо явно указать, что вы хотите список из элементов произвольного числового типа,
-- апример, используя синатксис с фигурными скобками и точками с запятой:
-- λ> let {coins :: Num a => [a]; coins = [2, 3, 7]}
-- λ> :t coins
-- coins :: Num a => [a]


-- perms :: [a] -> [[a]]
-- perms [] = [[]]
-- perms [x] = [[x]]
-- perms (x : xs) = concatMap (insertElem x) (perms xs)
--   where
--     insertElem x [] = [[x]]
--     insertElem x yss@(y : ys) = (x : yss) : map (y :) (insertElem x ys)

change :: (Ord a, Num a) => a -> [[a]]
--coins = [2, 3, 7]
coins :: Num a => [a]
coins = [2, 3, 7]
--change s = [[(coins !! 0) .. (coins !! (length coins -1)] | x <- coins, y <- coins, sx <- [[]], sum (x : sx) <= s]

-- change s = [x : (y : (z : sx)) | x <- coins, y <- coins, z <- coins, sx <- [[]], sum (x : (y : (z : sx))) == s]
-- *Main> change 7
-- [[2,2,3],[2,3,2],[3,2,2]]

-- * Main> [(x:(y:z))|x<-coins, (y:z)<-[coins]] where
-- * Main> change 7
-- [[2,2,3,7],[3,2,3,7],[7,2,3,7]]
-- *Main> [(x:[y])|x<-coins, y<-coins]
-- [[2,2],[2,3],[2,7],[3,2],[3,3],[3,7],[7,2],[7,3],[7,7]]

--change s = [x : (y : sx) | x <- coins, y <- coins, sx <- [[]], sum (x : sx) <= s]

-- * Main> change 7

-- [[2,2],[2,3],[2,7],[3,2],[3,3],[3,7],[7,2],[7,3],[7,7]]

-- change s = [(x : xs) | x <- coins, xs <- [[]], sum (x : xs) <= s] where
--   helper  [] =[]
--   helper (x:xs)  = x:(helper xs)
--u=[[],[4],[3,4],[2,3,4]]

--hel [0..(length coins)]
--[[],[3],[2,3],[1,2,3]]
--change s = [(x : xs) | x <- coins, xs <- [[],[3],[2,3],[1,2,3]], sum (x : xs) <= s]

-- * Main> change 7

-- [[2],[2,3],[2,2,3],[3],[3,3],[7]]

--hel [] =[]
--hel [f] = [f]
--hel (x : xs) =  hel xs ++ [xs]
--hel (x : xs) =  hel xs ++ [xs]
--hel (x : xs) = ([x] : (hel xs)) -- ++ [xs]


-- change s = [x : y | x <- coins, y  <- h coins, sum (x : y) <= s]
-- --h [] = [[]]
-- h [t] = [[t]]
-- h c@(a:b) = [x1 : y1| x1 <- c, y1 <- [a]: h b]
-- --* Main> change 7
-- -- [[2,2,2],[2,3,2],[3,2,2]]
-- -- *Main> h (tail coins)
-- -- [[3,3],[3,7],[7,3],[7,7]]

-- change s = [x : y | x <- coins, y <- h coins, sum (x : y) <= s]
-- h [] = [[]]
-- --h [a] = [[a]]
-- h c = [x1 : y1 | x1 <- c, y1 <- [head c] : h (tail c)]
-- *Main> change 7
--[[2,2,2],[2,3,2],[3,2,2]]
-- * Main> h (tail coins)
--[[3,3],[3,7,7],[3,7],[7,3],[7,7,7],[7,7]]

-- change s = [x : y | x <- coins, y <- h coins, sum (x : y) <= s]
-- h [] = [[]]
-- --h [a] = [[a]]
-- h c = [x1 : y1 | x1 <- c, y1 <- []:[head c] : h (tail c)]
-- *Main> change 7
-- [[2,2],[2,2,2],[2,2,3],[2,3],[2,3,2],[3,2],[3,2,2],[3,3]]
-- *Main> h (tail coins)
-- [[3],[3,3],[3,7],[3,7,7],[3,7],[7],[7,3],[7,7],[7,7,7],[7,7]]

-- change s = [x : y | x <- coins, y <- []:h coins, sum (x : y) <= s]
-- h [] = [[]]
-- --h [a] = [[a]]
-- h c = [x1 : y1 | x1 <- c, y1 <- []:[head c] : h (tail c)]
-- *Main> change 7
-- [[2],[2,2],[2,2,2],[2,2,3],[2,3],[2,3,2],[3],[3,2],[3,2,2],[3,3],[7]]
-- *Main> h (tail coins)
-- [[3],[3,3],[3,7],[3,7,7],[3,7],[7],[7,3],[7,7],[7,7,7],[7,7]]

-- change s = [x : y | x <- coins, y <- [] : h coins, sum (x : y) == s]
-- h [] = [[]]
-- --h [a] = [[a]]
-- h c = [x1 : y1 | x1 <- c, y1 <- [] : [head c]: h (tail c)]
-- *Main> change 7
-- [[2,2,3],[2,3,2],[3,2,2],[7]]
-- *Main> change 10
-- [[2,2,3,3],[3,7],[7,3]]
-- *Main> change 70
-- []

-- change s = [x ++ y | x <- h coins, y <- [] : h coins, sum (x ++ y) == s]
-- h [] = [[]]
-- h c = [x1 : y1 | x1 <- c, y1 <- [] : [head c] : h (tail c)]
-- *Main> change 7
-- [[2,2,3],[2,3,2],[2,2,3],[2,3,2],[3,2,2],[3,2,2],[7]]
-- *Main> change 10
-- [[2,2,3,3],[2,2,3,3],[2,3,2,3],[2,3,3,2],[2,3,3,2],[3,7],[3,2,2,3],[3,2,3,2],[3,3,2,2],[3,7],[7,3],[7,3]]
-- *Main> change 70
-- []

-- [[2],[2,2],[2,2,3],[2,2,3,3],[2,2,2],[2,3],[2,3,3],[2,3,2],[2,7],[3],[3,2],[3,2,3],[3,2,2],[3,3],[3,3,3],[3,3,2],[3,7],[7],[7,2],[7,3]]
-- [[2],[2,2],[2,2,2],[2,2,3],[2,2,3,3],[2,3],[2,3,2],[2,3,3],[2,7],[3],[3,2],[3,2,2],[3,2,3],[3,3],[3,3,2],[3,3,3],[3,7],[7],[7,2],[7,3]]

-- change s = [x : y | x <- coins, y <- [] : h coins [], sum (x : y) == s]
-- h [] v = [[]]
-- h c v = [x1 : y1 | x1 <- c, y1 <- [] : [head c] : h (tail c ++ v) (v ++ [head c])]
-- *Main> change 7
-- [[2,2,3]Interrupted.

-- change s = [x : y | x <- coins, y <- [] : h coins [head coins] s, sum (x : y) == s]
-- h [] v s= [[]]
-- h c v s = [x1 : y1 | x1 <- c, y1 <- v : h (tail c) (v ++ [head (tail c)]) (s - head c), (s>=0) && (sum (x1 : y1) <= s)]
-- *Main> change 7
-- [[2,3,2],[3,2,2],[7]]
-- *Main> change 70
-- []
-- *Main> change 10
-- []

-- change s = [x : y | x <- coins, y <- concatMap Data.List.permutations (Data.List.subsequences coins), sum (x : y) == s]
-- *Main> change 7
-- [[2,2,3],[2,3,2],[7]]
-- *Main> change 10
-- [[3,7],[7,3]]
-- *Main> change 70
-- []
-- *Main> Data.List.subsequences coins
-- [[],[2],[3],[2,3],[7],[2,7],[3,7],[2,3,7]]
-- *Main> Data.List.permutations coins
-- [[2,3,7],[3,2,7],[7,3,2],[3,7,2],[7,2,3],[2,7,3]]
-- *Main> concatMap Data.List.permutations (Data.List.subsequences coins)
-- [[],[2],[3],[2,3],[3,2],[7],[2,7],[7,2],[3,7],[7,3],[2,3,7],[3,2,7],[7,3,2],[3,7,2],[7,2,3],[2,7,3]]
-- *Main> inits "abc"
-- ["","a","ab","abc"]
-- *Main> subsequences "abc"
-- ["","a","b","ab","c","ac","bc","abc"]

-- change s = [x : y | x <- coins, y <- h coins, sum (x : y) == s]
-- --h coins = Data.Set.toList (Data.Set.fromList (concatMap permutations (subsequences (coins ++ reverse coins))))
-- h coins = concatMap permutations (subsequences (coins ))

-- change s = [x : y | x <- coins, y <- h coins (s `div` head coins), sum (x : y) == s]
-- h :: [a] -> a -> [[a]]
-- h coins n = nub (concatMap permutations (subsequences coins ++ [replicate n (head coins)]))
--a lot of type errors

-- change s = [x : y | x <- coins, y <- nub (subsequences (nub (concat (nub (permutations (concat ([] : h2 coins s ))))))), sum (x : y) == s]
-- h :: Eq a => [a]  -> [[a]]
-- h coins = nub (concatMap permutations (subsequences coins))
-- h2 :: (Num a, Ord a) => [a] -> a -> [[a]]
-- h2 coins s = [take s (Data.List.repeat x) | x <- coins, sum (Data.List.repeat x) <= s]
-- reset break
-- h2 :: (Num a, Ord a) => [a] -> a -> [[a]]
-- h2 coins s = [concatMap (replicate 50) [x] | x <- coins, sum (concatMap (replicate 50) [x]) <= s]
-- h2 :: (Num a, Ord a) => [a] -> a -> [[a]]
-- h2 coins s = [x:y | x <- coins, y<-[[x]], sum (x : y) <= s]
-- *Main> h2 coins 7
-- [[2,2],[3,3]]
-- *Main> h2 coins 15
-- [[2,2],[3,3],[7,7]]
--h2 :: (Num a, Ord a) => [a] -> a -> [[a]]
-- h2 :: Integral a => [a] -> a -> [[a]]
-- h2 coins s = [[x] | x <- coins, _ <- [1 .. (s `div` x)]]
-- * Main> h2 coins 7
-- [[2],[2],[2],[3],[3],[7]]
-- *Main> []:h2 coins 7
-- [[],[2],[2],[2],[3],[3],[7]]
-- *Main> concat ([] : h2 coins 5)
-- [2,2,3]
-- *Main> concat ([] : h2 coins 6)
-- [2,2,2,3,3]
-- *Main> subsequences (nub (permutations (concat ([] : h2 coins 3))))
-- [[],[[2,3]],[[3,2]],[[2,3],[3,2]]]
-- *Main> subsequences (nub (permutations (concat ([] : h2 coins 5))))
-- [[],[[2,2,3]],[[3,2,2]],[[2,2,3],[3,2,2]],[[2,3,2]],[[2,2,3],[2,3,2]],[[3,2,2],[2,3,2]],[[2,2,3],[3,2,2],[2,3,2]]]
-- *Main> subsequences (concat (nub (permutations (concat ([] : h2 coins 3)))))
-- [[],[2],[3],[2,3],[3],[2,3],[3,3],[2,3,3],[2],[2,2],[3,2],[2,3,2],[3,2],[2,3,2],[3,3,2],[2,3,3,2]]


-- change s = [x : y | x <- coins, y <- [] : [h2 coins s], sum (x : y) == s]
-- h :: Eq a => [a] -> [[a]]
-- h coins = nub (concatMap permutations (subsequences coins))
-- h2 :: Integral a => [a] -> a -> [a]
-- h2 coins s = [x | x <- coins, _ <- [1 .. (s `div` x)]]

-- change s = [x : y | x <- coins, y <- nub (subsequences(h2 coins s ++ reverse (h2 coins s))), sum (x : y) == s]
-- h2 :: (Ord a, Num a) => [a] -> a -> [a]
-- h2 coins s = [x | x <- coins, _ <- [1 .. (s `div` x)]]
--Compilation error
-- main.hs:5:56:
--     Could not deduce (Integral a) arising from a use of ‘h2’
--     from the context (Ord a, Num a)
--       bound by the type signature for
--                  change :: (Ord a, Num a) => a -> [[a]]
--       at main.hs:4:11-38
--     Possible fix:
--       add (Integral a) to the context of
--         the type signature for change :: (Ord a, Num a) => a -> [[a]]
--     In the first argument of ‘(++)’, namely ‘h2 coins s’
--     In the first argument of ‘subsequences’, namely
--       ‘(h2 coins s ++ reverse (h2 coins s))’
--     In the first argument of ‘nub’, namely
--       ‘(subsequences (h2 coins s ++ reverse (h2 coins s)))’

-- +++ https://stepik.org/lesson/8328/step/9?discussion=2856747&unit=1476
-- Посмотрев решения понял, что вариантов реализации очень много, поэтому и подсказки сильно отличаются.
-- Единственное, что алгоритм относится к динамическому программированию - рекурсивный вызов той же функции, только с другим (меньшим) аргументом и условием выхода из рекурсии.
-- В целом, условие выхода из рекурсии - значение параметра функции меньше, чем минимальное значение coins.
-- change s | s < minimum coins = []
-- Именно такой ответ ожидает система. А не [[]].
-- Далее я решил, что неплохо было бы получать некий список списков и фильтровать их по необходимому условию. Тогда можно будет возвращать немного избыточный список.
--          | otherwise = [xs | xs <- listOfLists, sum xs == s]
-- Теперь самая важная часть. Как сказано выше, алгоритмы динамического программирования рекурсивны. Из-за чего нам нужно описать одну итерацию и запустить рекурсию.
-- Если на словах, то change 7 это:
--     [ результат change 5, где к каждому списку добавлен 2 ] ++
--     [ результат change 4, где к каждому списку добавлен 3 ] ++
--     [ результат change 0, где к каждому списку добавлен 7 ]
-- Из предыдущего урока стало понятно, что вышеперечисленное можно элегантно написать как:
-- [ n : y | n <- coins, y <- changeResult ]
-- Ну и тут пришло время использовать костыль. Дело в том, что при запуске change 0  результатом будет пустой список и тогда не пройдет итерация с нужным n. Описание тут.
-- Для этого я добавляю пустой список в голову каждого списка результата.
--  [] : (change $ s - n)

--- works, but its stolen from the previous comment :((
change s | s < minimum coins = []
         | otherwise = [xs | xs <- listOfLists, sum xs == s] where
           listOfLists = [n : y | n <- coins, y <- changeResult n] where
             changeResult n = [] : (change $ s - n)

-- or minimal variant 
-- change s
--   | s < minimum coins = []
--   | otherwise = [xs | xs <- [n : y | n <- coins, y <- [] : (change $ s - n)], sum xs == s]

-- https://stepik.org/lesson/8328/step/9?discussion=345421&thread=solutions&unit=1476
-- Почти всегда в рекурсивных решениях вместо того, 
-- чтобы вставлять дополнительные проверки по ходу дела, достаточно аккуратно обработать граничные случаи.
-- Ноль можно набрать ровно одним способом — надо вернуть ничего. Если же сумма отрицательная, 
-- то никаких способов её набрать из монет положительного номинала не существует.
-- change :: (Ord a, Num a) => a -> [[a]]
-- change n
--   | n < 0 = []
--   | n == 0 = [[]]
--   | otherwise = [x : xs | x <- coins, xs <- change (n - x)]

--https://stepik.org/lesson/8328/step/9?discussion=433325&thread=solutions&unit=1476
-- change 0 = [[]]
-- change s = [x : xs | x <- coins, s - x >= 0, xs <- change (s - x)]

--https://stepik.org/lesson/8328/step/9?discussion=481615&thread=solutions&unit=1476
-- ШАГ 1. Первым делом, с помощью генератора списков создаю список, размер которого равен размеру списка монеток (если монетка не превосходит сумму, которую нужно разменять). Этот список списков и породит последующие списки. На каждой позиции в этом списке стоит уникальная монетка. Пока одна.
-- т.е. если coins = [2,3,7] , а сумма к размену = 7, то инициализирующий список и будет = [ [2],[3],[7]]
-- ШАГ 2. Выращу этот список как дерево
-- Добавляю рекурсию: для каждого из полученных на первом шаге списков, вызываю change для неразмененного остатка.
-- т.е. для списка [2] - мне нужно еще сумму 5 разменять набором coins.
-- Для [7] - это 0.
-- и т.д.
-- ШАГ 3. concatMap слепила все варианты, которые выросли из первого списка
-- ШАГ 4. Кайфую.
-- change :: (Ord a, Num a) => a -> [[a]]
-- change a = concatMap (helper) [[x] | x <- coins, sum [x] <= a]
--   where
--     helper xs | sum xs == a = xs : []
--     helper xs | otherwise = map (++ xs) (change (a - sum xs))

--https://stepik.org/lesson/8328/step/9?discussion=381863&thread=solutions&unit=1476
-- change n
--   | n == 0 = [[]]
--   | n < 0 = []
--   | otherwise = concat [map (x :) (change (n - x)) | x <- coins, n - x >= 0]

--https://stepik.org/lesson/8328/step/9?discussion=338823&thread=solutions&unit=1476
-- change :: (Ord a, Num a) => a -> [[a]]
-- change n
--   | n <= 0 = [[]]
--   | otherwise = [coin : m | coin <- coins, coin <= n, m <- change (n - coin)]


-- *Lst> toster 4
-- [[2,2]]
-- *Lst> toster 5
-- [[3,2],[2,3]]
-- *Lst> toster 6
-- [[2,2,2],[3,3]]
-- *Lst> toster 7
-- [[3,2,2],[2,2,3],[2,3,2],[7]]
-- *Lst> toster 8
-- [[2,2,2,2],[3,3,2],[3,2,3],[2,3,3]]
-- *Lst> toster 10
-- [[2,2,2,2,2],[3,3,2,2],[3,2,2,3],[3,2,3,2],[2,2,3,3],[2,3,3,2],[2,3,2,3],[7,3],[3,7]]




-- --https :// elijahoyekunle . com / technology / 2018 / 10 / 15 / List - Filter - In - Haskell.html
-- type ArrFilter = Int -> [Int] -> [Int]
-- f :: ArrFilter
-- f _ [] = []
-- f n (x : xs) =
--   if x < n
--     then x : f n xs
--     else f n xs
-- f2 :: ArrFilter
-- f2 n arr = [num | num <- arr, num < n]
-- f3 :: ArrFilter
-- f3 n arr = filter (< n) arr
-- f4 :: ArrFilter
-- f4 n = filter (< n)
-- f5 :: ArrFilter
-- f5 = filter . (>)
-- --https :// elijahoyekunle . com / technology / 2018 / 10 / 05 / Haskell - Filter - Positions - In - List.html
-- type FilterPos = [Int] -> [Int]
-- f :: FilterPos
-- f lst = [lst !! n | n <- [0 .. length lst - 1], odd n]
-- f2 :: FilterPos
-- f2 (_ : x : xs) = x : f2 xs
-- f2 _ = []
-- f3 :: FilterPos
-- f3= map snd . filter (odd . fst) . zip [0 ..]
-- --f2 [2,4,6,8,10,12,14]
-- --[4,8,12]

--https://rsdn.org/forum/decl/4104888.all
--Как бы вы написали функцию для получения всех циклических перестановок списка?
rotations xs = zipWith const (iterate rotate xs) xs
  where
    rotate (y : ys) = ys ++ [y]
-- * Main> rotations coins
-- [[2,3,7],[3,7,2],[7,2,3]]

-- let rotations xs = tail $ zipWith (++) (tails xs) (inits xs)

--А функцию для получения всех перестановок списка (не используя permutations из Prelude)?
permutations' [] = [[]]
permutations' (x : xs) = permutations' xs >>= rotations . (x :)
-- *Main> permutations' coins
-- [[2,3,7],[3,7,2],[7,2,3],[2,7,3],[7,3,2],[3,2,7]]


--3.4.3
{-
Напишите реализацию функции concatList через foldr
GHCi > concatList [[1, 2], [], [3]]
[1, 2, 3]
-}
concatList :: [[a]] -> [a]
concatList  = foldr (++) []

--3.4.5
{-
Используя функцию foldr, напишите реализацию функции lengthList, вычисляющей количество элементов в списке.
GHCi> lengthList [7,6,5]
3
-}
lengthList :: [a] -> Int
lengthList  = foldr (\_ s -> s+1) 0

--3.4.6
{-
Реализуйте функцию sumOdd, которая суммирует элементы списка целых чисел, имеющие нечетные значения:
GHCi> sumOdd [2,5,30,37]
42
-}
sumOdd :: [Integer] -> Integer
sumOdd = foldr (\x s -> if odd x then s+x else s) 0

--3.4.8
{-
Какой функции стандартной библиотеки, суженной на списки, эквивалентно выражение foldr (:) []?
-}
--id

--3.4.9
{-
Какой функции стандартной библиотеки эквивалентно выражение foldr const undefined?
-}
--head

--3.5.3
{-
При каком значении переменной x следующие два выражения примут одно и то же значение (отличное от неопределенного)?
foldr (-) x [2,1,5]
foldl (-) x [2,1,5]
-}
--7

--3.5.8
{-
Реализуйте функцию meanList, которая находит среднее значение элементов списка, используя однократный вызов функции свертки.
GHCi> meanList [1,2,3,4]
2.5
Постобработка считается допустимой, то есть предполагаемая реализация функции meanList имеет вид
meanList = someFun . foldr someFoldingFun someIni
-}

meanList :: [Double] -> Double
meanList = someFun . foldr someFoldingFun someIni where
  someFun (a,b) = b/a
  someFoldingFun = \x (n, s) -> (n + 1, s + x)
  someIni = (0, 0)

--3.5.9
{-
Используя однократный вызов свертки, реализуйте функцию evenOnly, 
которая выбрасывает из списка элементы, стоящие на нечетных местах, оставляя только четные.
GHCi> evenOnly [1..10]
[2,4,6,8,10]
GHCi> evenOnly ['a'..'z']
"bdfhjlnprtvxz"
-}

--https://stackoverflow.com/questions/31789654/easy-way-to-init-a-boolean-list-in-haskell
--[even x | x <- [1..6]] -- [False,True,False,True,False,True]

evenOnly :: [a] -> [a]

evenOnly = fst . foldl f ([], False) where
  f :: ([a], Bool) -> a -> ([a], Bool)
  f  (s, b) x = if b then (s++[x], not b) else (s, not b)

-- *Main> evenOnly [1..10]
-- [2,4,6,8,10]
-- *Main> evenOnly [1..9]
-- [2,4,6,8]

--https://stepik.org/lesson/5790/step/9?discussion=738644&thread=solutions&unit=1136
-- evenOnly :: [a] -> [a]
-- evenOnly [] = []
-- evenOnly [_] = []
-- evenOnly (x1 : x2 : xs) = x2 : evenOnly xs

--https://stepik.org/lesson/5790/step/9?discussion=1352688&thread=solutions&unit=1136
-- evenOnly :: [a] -> [a]
-- evenOnly (x : y : t) = y : evenOnly t
-- evenOnly _ = []

-- 3.5.10
{-
Попробуйте добиться того, чтобы реализованная вами в прошлом задании функция evenOnly позволяла работать и с бесконечными списками.
То есть, например, запрос на первые три элемента бесконечного списка, возвращаемого этой функцией, примененной к списку всех 
натуральных чисел, должен завершаться:
GHCi> take 3 (evenOnly [1..])
[2,4,6]
-}

-- *Main> take 3 (evenOnly [1..])
-- Interrupted.

-- evenOnly2 :: [a] -> [a]
-- evenOnly2  = fst  . foldr f ([], False) 
--   where
--     f :: a -> ([a], Bool) -> ([a], Bool)
--     f x (s, b) = if b then (x:s, not b) else (s, not b)
-- *Main> take 3 (evenOnly2 [1..])
-- Interrupted.
-- *Main> take 3 (evenOnly2 [1..])
-- *** Exception: stack overflow
-- *Main> evenOnly2 [1..9]
-- [2,4,6,8]
-- *Main> evenOnly2 [1..10]
-- [1,3,5,7,9]

-- evenOnly2 :: [a] -> [a]
-- evenOnly2 =  fst  . foldr f ([], False) 
--   where
--     f :: a -> ([a], Bool) -> ([a], Bool)
--     f x ~(s, b) = if b then (x : s, not b) else (s, not b) 

-- evenOnly2 :: [a] -> [a]
-- evenOnly2 = fst. foldr f ([], False)
--   where
--     f :: a -> ([a], Bool) -> ([a], Bool)
--     f x ~(s, b) =  (x : s, not b) 
-- *Main> take 3 (evenOnly2 [1..])
-- [1,2,3]

-- evenOnly2 :: [a] -> [a]
-- evenOnly2 = fu ( foldr f ([], [bu]) w)
--   where
--     bu = odd (length w)
--     f :: a -> ([a], [Bool]) -> ([a], [Bool])
--     f x ~(s, b) =  (x : s, not (head b):b)
--     fu :: ([a], [Bool])->[a]
--     fu k = fi( q k) where 
--       q :: ([a],[Bool])->[(a, Bool)]
--       q k = zip (fst k) (snd k)
--       fi :: [(a, Bool)] -> [a]
--       fi p = [fst a | a <- p, snd a]
-- *Main> take 3 (evenOnly2 [1..])
-- Interrupted.

-- https://stackoverflow.com/questions/11106009/haskell-how-to-define-strict-tuples/11106072#11106072
-- makeStrict (a, b) = seq a (seq b (a, b))

-- ++ https://en.wikibooks.org/wiki/Haskell/Laziness
-- +https://wiki.haskell.org/Performance/Laziness
-- cleave :: [a] -> ([a],[a])
-- cleave = cleave' ([], [])
--    where
--      cleave' (eacc, oacc) [] = (eacc, oacc)
--      cleave' (eacc, oacc) [x] = (x : eacc, oacc)
--      cleave' (eacc, oacc) (x : x' : xs) = cleave' (x : eacc, x' : oacc) xs
-- * Main> take 3 (evenOnly2 [1..])
-- Interrupted.
evens [] = []
evens [x] = [x]
evens (x : _ : xs) = x : evens xs
odds [] = []
odds [x] = []
odds (_ : x : xs) = x : odds xs
cleave xs = (evens xs, odds xs)

evenOnly2 :: [a] -> [a]
evenOnly2 = snd .cleave
-- *Main> take 3 (evenOnly2 [1..])
-- [2,4,6]


-- https://stepik.org/lesson/5790/step/10?discussion=385917&thread=solutions&unit=1136
-- evenOnly :: [a] -> [a]
-- evenOnly = snd . foldr (\x ~(xs, ys) -> (x : ys, xs)) ([], [])

--https://stepik.org/lesson/5790/step/10?discussion=342766&thread=solutions&unit=1136
-- evenOnly :: [a] -> [a]
-- evenOnly (x : y : xs) = y : evenOnly xs
-- evenOnly _ = []

-- https://stepik.org/lesson/5790/step/10?discussion=978117&thread=solutions&unit=1136
-- evenOnly :: [a] -> [a]
-- evenOnly = snd . foldr (\a l -> let (xs, ys) = l in (a : ys, xs)) ([], [])

-- https://stepik.org/lesson/5790/step/10?discussion=347009&thread=solutions&unit=1136
-- evenOnly :: [a] -> [a]
-- evenOnly = snd . foldr f ([], [])
--   where
--     f x p = (x : snd p, fst p)

-- https://stepik.org/lesson/5790/step/10?discussion=1379576&thread=solutions&unit=1136
-- evenOnly :: [a] -> [a]
-- evenOnly = foldr (\(b, x) xs -> if b then x : xs else xs) [] . zip zig
-- zig = False : True : zig

--https://stepik.org/lesson/5790/step/10?discussion=580726&thread=solutions&unit=1136
-- evenOnly :: [a] -> [a]
-- evenOnly = foldr (\(n, x) l -> if (even n) then x : l else l) [] . zip [1 ..]


--3.6.3
{-
Напишите реализацию функции, возвращающей последний элемент списка, через foldl1.
lastElem :: [a] -> a
lastElem = foldl1 undefined
-}
lastElem :: [a] -> a
lastElem (s : ss) = foldl1 f (s : ss)
  where
    f :: a -> a -> a
    f x d = d
--f _ d = d

-- +++ foldr f ini [1,2,3] ~>> 1 `f` (2 `f` (3 `f` ini))
-- +++ foldl f ini [1,2,3] ~>> ((ini `f` 1) `f` 2) `f` 3

-- 3.6.10
{-
Используя unfoldr, реализуйте функцию, которая возвращает в обратном алфавитном порядке список символов, 
попадающих в заданный парой диапазон. Попадание символа x в диапазон пары (a,b) означает, что x >= a и x <= b.
revRange :: (Char,Char) -> [Char]
revRange = unfoldr g 
  where g = undefined
GHCi> revRange ('a','z')
"zyxwvutsrqponmlkjihgfedcba"
-}

-- https://stepik.org/lesson/6196/step/10?discussion=743603&unit=1229
-- То, что должна возвращать программа:
-- *Lst> revRange ('a','z')
-- "zyxwvutsrqponmlkjihgfedcba"
-- *Lst> revRange ('z','a')
-- ""
-- *Lst> revRange ('a','a')
-- "a"
-- *Lst> revRange ('z','z')
-- "z"

revRange :: (Char, Char) -> [Char]
revRange (a, b) = unfoldr g  b
  where
    --g :: (Char, Char) -> Maybe (Char, Char)
    --g :: (Char, Char) -> Maybe (Char, Char)
    g = \x -> if x < a then Nothing else Just (x, pred x)
-- *Main> revRange ('a','z')
--"zyxwvutsrqponmlkjihgfedcba"
-- *Main> revRange ('z','a')
-- "a`_^]\\[ZYXWVUTSRQPONMLKJIHGFEDCBA@?>=<;:9876543210/.-,+*)('&%$#\"! \US\RS\GS\FS\ESC\SUB\EM\CAN\ETB\SYN\NAK\DC4\DC3\DC2\DC1\DLE\SI\SO\r\f\v\n\t\b\a\ACK\ENQ\EOT\ETX\STX\SOH\NUL*** Exception: Prelude.Enum.Char.pred: bad argument
-- *Main> revRange ('a','a')
-- ""
-- * Main> revRange ('z','z')
-- ""
-- не работает на revRange ('\NUL', '\SOH')

-- https :// wiki . haskell . org / The_Fibonacci_sequence
--fibs = unfoldr (\(a,b) -> Just (a,(b,a+b))) (0,1)

-- https://www.cyberforum.ru/haskell/thread2360679.html

-- revRange :: (Char, Char) -> [Char]
-- revRange (a, b) = unfoldr fun (b, False)
--   where
--     fun (x, isEnd)
--       | x < a || isEnd = Nothing
--       | x > a = Just (x, (pred x, False))
--       | x == a = Just (x, (x, True))

-- revRange (a, b) = unfoldr fun b
--     where fun x 
--             | x < a = Nothing
--             | otherwise = Just (x, pred x)

-- import Data.Char
-- import Data.List
-- revRange :: (Char, Char) -> [Char]
-- revRange (a, b) = unfoldr fun $ ord b
--   where
--     a' = ord a
--     fun x
--       | x < a' = Nothing
--       | otherwise = Just (chr x, pred x)

-- task :: (Char, Char) -> String
-- task (x, y) = reverse $ [s | s <- [x .. y]]

-- https://stepik.org/lesson/6196/step/10?discussion=339607&thread=solutions&unit=1229
-- revRange :: (Char,Char) -> [Char]
-- revRange = unfoldr g where
--   g (begin, end)
--     | begin > end = Nothing
--     | otherwise   = Just (end, (begin, pred end))

-- https://stepik.org/lesson/6196/step/10?discussion=344485&thread=solutions&unit=1229
-- revRange :: (Char, Char) -> [Char]
-- revRange = unfoldr g
--   where
--     g (x, y) = if x > y then Nothing else Just (y, (x, pred y))