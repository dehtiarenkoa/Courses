{-# LANGUAGE MultiWayIf #-}
--3.1.3

{-}
Реализуйте функцию addTwoElements, которая бы добавляла два переданных ей значения в голову переданного списка.
GHCi> addTwoElements 2 12 [85,0,6]
[2,12,85,0,6]
-}
--{-# OPTIONS_GHC -Wno-overlapping-patterns #-}
addTwoElements :: a -> a -> [a] -> [a]
addTwoElements x y s = x: y: s

--3.1.4
{-

Реализуйте функцию nTimes, которая возвращает список, 
состоящий из повторяющихся значений ее первого аргумента. 
Количество повторов определяется значением второго аргумента этой функции.
GHCi> nTimes 42 3
[42,42,42]
GHCi> nTimes 'z' 5
"zzzzz"
-}
nTimes :: a -> Int -> [a]
nTimes a n =  helper a n 0 []
  where
      helper a n count list
         | n==count = list
         | otherwise = helper a n (count+1) (a:list)  -- also works:
-- | otherwise = (helper a n $! (count+1)) (a:list)

--3.1.6
sndHead = snd . head
sndHead1 ((,) x y : z) = x
sndHead2 ((:) ((,) _ x) y) = x
sndHead3 ((_, x) : _) = x
sndHead4 ((,) ((:) _ _) x) = x
sndHead5 ((,) y z : x) = x
sndHead6 ((,) y x : z) = x

--3.1.8
{-
Сформируйте список целых чисел, содержащий только те элементы исходного списка, значение которых нечетно.
GHCi> oddsOnly [2,5,7,10,11,12]
[5,7,11]
Для анализа четности можно использовать функции odd и even стандартной библиотеки.
-}
oddsOnly :: Integral a => [a] -> [a]
oddsOnly a = helper (length a) a [] 0 where
    helper le a b count
      | count == le = reverse b
    --   | count<le = if
    --     |odd h -> helper le t (h : b) (count+1)
    --     |even h -> helper le t b (count+1)
      | count<le = if odd h
          then helper le t (h : b) (count+1)
          else helper le t b (count+1)
      | otherwise = error "[Charerrrrror]"
     where
      h = head a
      t = tail a
      -- or (h:t) = a

{-
https://stepik.org/lesson/8326/step/8?discussion=338211&thread=solutions&unit=1474
oddsOnly :: Integral a => [a] -> [a]
oddsOnly [] = []
oddsOnly (x : xs)
  | odd x = x : oddsOnly xs
  | otherwise = oddsOnly xs

https://stepik.org/lesson/8326/step/8?discussion=343499&thread=solutions&unit=1474
oddsOnly :: Integral a => [a] -> [a]
oddsOnly = filter odd

https://stepik.org/lesson/8326/step/8?discussion=345799&thread=solutions&unit=1474
oddsOnly' :: Integral a => [a] -> [a]
oddsOnly' [] = []
oddsOnly' xs = [x | x <- xs, odd x]

https://stepik.org/lesson/8326/step/8?discussion=345799&thread=solutions&unit=1474
oddsOnly :: Integral a => [a] -> [a]
oddsOnly = \xs -> [x | x <- xs, odd x]

https://stepik.org/lesson/8326/step/8?discussion=1668617&thread=solutions&unit=1474
oddsOnly :: Integral a => [a] -> [a]
oddsOnly list = helper [] list
  where
    helper acc list
      | null list = reverse acc
      | otherwise = helper newAcc listTail
      where
        newAcc = if odd listHead then (listHead : acc) else acc
        (listHead : listTail) = list

https://stepik.org/lesson/8326/step/8?discussion=1046422&thread=solutions&unit=1474
oddsOnly :: Integral a => [a] -> [a]
--oddsOnly = filter odd
oddsOnly [] = []
oddsOnly (x : xs) = (++ oddsOnly xs) $ if odd x then [x] else []

https://stepik.org/lesson/8326/step/8?discussion=448282&thread=solutions&unit=1474
oddsOnly :: Integral a => [a] -> [a]
oddsOnly [] = []
oddsOnly (x : xs) = if odd x then x : oddsOnly xs else oddsOnly xs
-}

--3.1.10

{-
Реализуйте функцию isPalindrome, которая определяет, является ли переданный ей список палиндромом.
GHCi> isPalindrome "saippuakivikauppias"
True
GHCi> isPalindrome [1]
True
GHCi> isPalindrome [1, 2]
False
-}

isPalindrome :: Eq a => [a] -> Bool
isPalindrome a = helper a where
  helper a
   | null a || null (tail a)  = True
   | head a /= last a = False
   | otherwise = helper (tail (init a))

--3.1.12
{-
Составьте список сумм соответствующих элементов трех заданных списков. 
Длина результирующего списка должна быть равна длине самого длинного из заданных списков, 
при этом «закончившиеся» списки не должны давать вклада в суммы.
GHCi> sum3 [1,2,3] [4,5] [6]
[11,7,3]
-}

sum3 :: Num a => [a] -> [a] -> [a] -> [a]
sum3 a b c = helper a b c []
 where
  helper :: Num a => [a] -> [a] -> [a] -> [a] -> [a]
  helper [] [] [] s = s
  helper a b c s = helper at bt ct (s++[ah+bh+ch])
   where
     ah = if not (null a) then head a else 0
     bh = if not (null b) then head b else 0
     ch = if not (null c) then head c else 0
     at = if not (null a) && not (null (tail a)) then tail a else []
     bt = if not (null b) && not (null (tail b)) then tail b else []
     ct = if not (null c) && not (null (tail c)) then tail c else []
-- | otherwise = helper (tail a) (tail b) (tail c) (head a + head b + head c):s   

{-
https://stepik.org/lesson/8326/step/12?discussion=434731&thread=solutions&unit=1474
sum3 :: Num a => [a] -> [a] -> [a] -> [a]
sum3 xs ys zs = xs `sum2` ys `sum2` zs
  where
    sum2 [] bs = bs
    sum2 as [] = as
    sum2 (a : as) (b : bs) = (a + b) : sum2 as bs

https://stepik.org/lesson/8326/step/12?discussion=337277&thread=solutions&unit=1474
import Data.List
sum3 :: Num a => [a] -> [a] -> [a] -> [a]
sum3 a b c = map sum (transpose [a, b, c])

https://stepik.org/lesson/8326/step/12?discussion=385762&thread=solutions&unit=1474
sum3 :: Num a => [a] -> [a] -> [a] -> [a]
sum3 (a : as) (b : bs) (c : cs) = (a + b + c) : sum3 as bs cs
sum3 [] [] [] = []
sum3 a b c = sum3 (f a) (f b) (f c) where f [] = [0]; f arr = arr

-}


--3.1.13
{-
Напишите функцию groupElems которая группирует одинаковые элементы в списке (если они идут подряд) и возвращает список таких групп.
GHCi> groupElems []
[]
GHCi> groupElems [1,2]
[[1],[2]]
GHCi> groupElems [1,2,2,2,4]
[[1],[2,2,2],[4]]
GHCi> groupElems [1,2,3,2,4]
[[1],[2],[3],[2],[4]]
Разрешается использовать только функции, доступные из библиотеки Prelude.

groupElems :: Eq a => [a] -> [[a]]
groupElems [] = [[]]
groupElems [a] = [[a]]
groupElems (a : at) = groupElems [a] ++ groupElems at
--*Main> groupElems [1,2,3,2,4]
--[[1],[2],[3],[2],[4]]


groupElems :: Eq a => [a] -> [[a]]
groupElems a = [gp a]
gp [] = []
gp [a] = [a]
gp (a : at)
  | a == head at = gp [a] ++ gp at
  | otherwise = gp [a] ++ gp at
--*Main> groupElems [1,2,3,2,4]
--[[1,2,3,2,4]]
-}

-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems a = [gp a a]
-- gp :: Eq a => [a] [a] -> [a]
-- gp [] [_] = []
-- gp [a] [_] = [a]
-- gp (a : at) (b:bt)
--   | a == la = gp [a] a ++ gp (if length (head at) > 1 then at else [head at]) la
--   | otherwise = gp [a] la ++ gp at la .......

-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems [] = []
-- groupElems [a] = [[a]]
-- groupElems (a : at) =  
--   [f (groupElems [a] ++ groupElems at) [a]] where 
--    f:: [a] [a]->[[a]]
--    f [a][b]  = [b]
--    f (a:at) [b]
--     | a==head at = f at ([(head b) ++ (head at)++ (tail b)])
--     | otherwise =......

-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems [] = [[]]
-- groupElems [a] = [[a]]
-- groupElems (a : at) = ([a] ++ takeWhile (== a) at) : groupElems (dropWhile (== a) at)
-- groupElems [1,2,2,2,4]
-- <interactive>:16:1: error: Data constructor not in scope: GHCi :: [[a0]]


-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems [] = [[]]
-- groupElems [a] = [[a]]
-- groupElems (a : at) = ([a] ++ takeWhile (== a) at) : f at where
--   f b 
--    |length b ==2 = 
--    |otherwise = groupElems (dropWhile (== a) at) --f [] (a:at) where


--f c a = [(([a] ++ takeWhile (== a) at):), f dropWhile (== a) at]
--  where
--   f c a = [c++(takeWhile (== head a) a), f (dropWhile (/= head a) a)]


-- f :: [[a]] ->[[a]]
-- f a = helper a length a
--  where
--    helper _ 0 = _
--    helper a 1 =
--    helper a coun =  helper g (coun-1)
--     where
--       g .........

-----------------------------------------
-- works!!
-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems [] = []
-- groupElems [a] = [[a]]
-- groupElems (ah : at) = 
--   let d=dropWhile (==ah) at
--       t=takeWhile (==ah) at 
--   in 
--   if ah == head at  
--       then (
--             if not (null d)
--                     then (ah : t) : groupElems d
--                           else [ah : t])
--   else [ah]:groupElems at

-----------------------------------------
-- works!!
groupElems :: Eq a => [a] -> [[a]]
groupElems [] = []
groupElems [a] = [[a]]
groupElems (ah:at)
  | null (snd s) = [fst s]
  | length (snd s) == 1 = fst s:[snd s]
  | otherwise =fst s: groupElems (snd s)
 where s = span (==ah) (ah:at)

--https://stepik.org/lesson/8326/step/13?discussion=345722&thread=solutions&unit=1474
-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems [] = []
-- groupElems [x] = [[x]]
-- groupElems (x : xs)
--   | x == head xs =
--     let (r : rs) = groupElems xs
--      in (x : r) : rs
--   | otherwise = [x] : groupElems xs

-- +++
--https://stepik.org/lesson/8326/step/13?discussion=344676&thread=solutions&unit=1474
-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems [] = []
-- groupElems (x : xs) = case groupElems xs of
--   [] -> [[x]]
--   (g : gs)
--     | x == head g -> (x : g) : gs
--     | otherwise -> [x] : g : gs
--gs :: [[a]] (bound at E:\Docs\Haskell\stepik\3_1.hs:310:8)
--g :: [a] (bound at E:\Docs\Haskell\stepik\3_1.hs:310:4)
--xs :: [a] (bound at E:\Docs\Haskell\stepik\3_1.hs:308:18)
--x :: a (bound at E:\Docs\Haskell\stepik\3_1.hs:308:14)
--[2,2,3,3]
--2:[2,3,3]
--2:(2:[3,3])
--
--https://stepik.org/lesson/8326/step/13?discussion=336936&thread=solutions&unit=1474
-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems [] = []
-- groupElems xs = ys : groupElems zs
--   where
--     (ys, zs) = span (== head xs) xs
--
--https://stepik.org/lesson/8326/step/13?discussion=337721&thread=solutions&unit=1474
-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems xs = helper xs []
--   where
--     helper [] [] = []
--     helper [] ys = [ys]
--     helper (x : xs) [] = helper xs ([x])
--     helper (x : xs) ys = if x == head ys then helper xs (x : ys) else ys : (helper xs [x])
--
--https://stepik.org/lesson/8326/step/13?discussion=3199303&thread=solutions&unit=1474
-- groupElems :: Eq a => [a] -> [[a]]
-- groupElems [] = []
-- groupElems (x : xs) = takeWhile (== x) (x : xs) : groupElems (dropWhile (== x) (x : xs))