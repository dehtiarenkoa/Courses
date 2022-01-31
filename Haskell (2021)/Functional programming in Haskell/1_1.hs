{-# OPTIONS_GHC -Wno-deferred-type-errors #-}
module Hello where

--main = print $ (fibonacci1 3) --"Hello, world!"

{--
ghci
:load Hello
:reload Hello

main :: IO ()
main = putStrLn (2 ^ 3 ^ 2)
--}
--fu :: Floating a => a -> a -> a
--fu x y = sqrt (x + y)
--let d = fu 3 2
--putStrLn fu 3 2
--lenVec3 :: Floating a => a -> a -> a -> a
--lenVec3 x y z = sqrt (x * x + y * y + z * z)
--lenVec3 2 3 6

--1.3.8
{--
infixl 6 *+*
(*+*) a b = a ^ 2 + b ^ 2
--}

--1.3.9
{-- 
infixl 6 |-|
(|-|) a b = if a>b then a-b else b-a 
--}

--1.3.12
{--
logBase 4 (min 20 (9 + 7))=logBase 4 $ min 20 $ 9 + 7
--}

--1.4.7
{--
discount :: Double -> Double -> Double -> Double
discount limit proc sum = if sum >= limit then sum * (100 - proc) / 100 else sum

standardDiscount :: Double -> Double
standardDiscount = discount 1000 5

--}

--1.4.9
{--
import Data.Char
twoDigits2Int :: Char -> Char -> Int
twoDigits2Int x y = if isDigit x && isDigit y then digitToInt x * 10 + digitToInt y else 100
--}

--1.4.11
{--
dist :: (Double, Double) -> (Double, Double) -> Double
dist p1 p2 = sqrt ((fst p2- fst p1)^2 + (snd p2 - snd p1)^2)
--}

{--
>Prelude
:i (++)
(++) :: [a] -> [a] -> [a]       -- Defined in `GHC.Base'
infixr 5 ++
--}

--1.5.4
{--
doubleFact :: Integer -> Integer
doubleFact 1 = 1
doubleFact 2 = 2
doubleFact n = n * doubleFact (n -2)
--}

--1.5.8
{--
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = fibonacci (n - 1) + fibonacci (n - 2)

fibonacci1 0 = 0
fibonacci1 1 = 1
fibonacci1 (-1) = 1
fibonacci1 n | n > 0  = fibonacci1 (n - 1) + fibonacci1 (n - 2)
             | n < 0  = fibonacci1 (n + 2) - fibonacci1 (n + 1)
--}

-- GHCi> :set +s
-- GHCi > fibonacci 30 
-- 832040
-- (4.89 secs, 1, 223, 136, 048 bytes)!!!
-- (5.55 secs, 1, 384, 688, 288 bytes)

--1.5.10

-- below works!!!
fibonacci1 n 
  | n >= 0 = func2 0 1 n 1    
  | n < 0 = func3 0 1 (-n) 1

func3 fi se n count
  | n == (-1) = 1
  | count < n = func3 se (se + fi) n (count + 1)
  | otherwise = se * (-1) ^ (n -1)
func2 fi se n count
  | n == 0 = 0
  | n == 1 = 1  
  | count < n = func2 se (fi + se) n (count + 1)
  | otherwise = se
-- above works!!!

  --  0 1 5 0
  --p 1 1 5 1
  --p 1 2 5 2
  --p 2 3 5 3
  --p 3 5 5 4
-- stack overflow
func1 fi se n
  | se < n = func1 se (fi+se) n
  | otherwise = se
{--5
*Hello> fibonacci1 6
8
*Hello> fibonacci1 7
8
*Hello> fibonacci1 8
8
*Hello> fibonacci1 9
13
*Hello> fibonacci1 10
13
*Hello> fibonacci1 11
13
*Hello> fibonacci1 12
13
*Hello> fibonacci1 13
13
*Hello> fibonacci1 14
21
--}

prev accum 1 = 1
prev accum 0 = accum
prev accum xn = prev (accum + 2*xn -3) (xn - 2)
{--
prev accum 1 = 1
prev accum 0 = 0
prev accum xn = prev (accum + xn) (xn - 2)

0 3 = 
prev accum xn = prev (accum + xn) (xn - 1)
prev 0 3 = prev (0 + 3) (3 - 1)
prev 3 2 = prev (3 + 2) (2 - 1)
prev 5 1 = --}

-- 1.6.6
--a0​=1;a1​=2;a2​=3;ak+3​=ak+2​+ak+1​−2ak​.
seqA :: Integer -> Integer
seqA n 
  | n==0 = 1
  | n==1 = 2
  | n==2 = 3
  | n>2 = seqA (n-1) + seqA (n-2) - 2 * seqA (n-3)
-- tooooooooooo long 

--below works!!
seqf :: Integer -> Integer
seqf n
  | n == 0 = 1
  | n == 1 = 2
  | n == 2 = 3
  | n > 2 = helper n 0 3 2 1

helper n counter p pp ppp
  | (n > 2) && ((counter + 3) < n) = helper n (counter + 1) (p + pp -2 * ppp) p pp
  | (n > 2) && ((counter + 3) == n) = p + pp -2 * ppp
--above works!!

{--
working variant with 'let':
seqf :: Integer -> Integer
seqf n
  | n == 0 = 1
  | n == 1 = 2
  | n == 2 = 3
  | n > 2 =
    let helper n counter p pp ppp
          | (n > 2) && ((counter + 3) < n) = helper n (counter + 1) (p + pp -2 * ppp) p pp
          | (n > 2) && ((counter + 3) == n) = p + pp -2 * ppp
     in helper n 0 3 2 1
--}

--GHCi> seqA 301 = 1276538859311178639666612897162414
--[1,2,3,3,2,-1,-5,-10,-13,-13,-6,7,27,46,59,51,18,-49,-133,-218,-253]
-- n3 = n2+n1-2n0
-- n4 = n3+n2-2n1 = n2+n1-2n0+n2-2n1 = 2n2-n1-2n0
-- n5 = n4+n3-2n2 = n3+n2-2n1+n3-2n2 = 2*(n2+n1-2n0)-2n1-n2 = n2-4n0
-- n6 = n5+n4-2n3 = n4+n3-2n2+n4-2n3 = 2(n3+n2-2n1)+n3-2n2-2n3 = 2n3+2n2-4n1+n3-2n2-2n3=n3-4n1
--  = n2+n1-2n0-4n1 = n2-3n1-2n0  
-- n7 = n6+n5-2n4 = n5+n4-2n3+n5-2n4 = 2*(n2-4n0)-(2n2-n1-2n0)-2*(n2+n1-2n0) = n1-6n0-2n2-2n1+4n0 = -2n2-n1-2n0
-- n8 = n7+n6-2n5 = -2n2-n1-2n0+n2-3n1-2n0-2n2+8n0 = -3n2-4n1+4n0
-- n9 = n8+n7-2n6 = -3n2-4n1+4n0-2n2-n1-2n0-2n2+6n1+4n0 = -7n2+n1+6n0
--n3  1  1 -2 =3
--n4  2 -1 -2 =2
--n5  1  0 -4 =-1
--n6  1 -3 -2 =-5
--n7 -2 -1 -2 =-10
--n8 -3 -4  4 =-13
--n9 -7  1  6 =-13
--helper n nn3 nn2 nn1 = seqf nn1 + seqf nn2 - 2 * seqf nn3 

{--
working variant from the last comment https://stepik.org/lesson/8414/step/6?discussion=632527&unit=1553
helper :: Integer -> Integer -> Integer -> Integer -> Integer
helper n3 _ _ 2 = n3
helper n3 n2 n1 n = helper (n3 + n2 - 2 * n1) n3 n2 (n -1)
--}

--1.6.8
sum'n'count :: Integer -> (Integer, Integer)
sum'n'count x = helper1 0 0 (abs x)
  where 
    helper1 sumn count x
      | x<10 = (sumn+x, count+1)
      | otherwise = helper1 (sumn + (x `mod` 10)) (count+1) (x `div` 10)
--GHCi> sum'n'count (-39) =====(12, 2)

--1.6.9
integration :: (Double -> Double) -> Double -> Double -> Double
integration f a b = let 
  n=1000
  step =  (b-a)/n
  in helper2 step n f a b a (a+step) 0 0
    where     
      helper2 step n f a b aa bb summ count
        | count == n = summ
        | otherwise = 
          let                   
            su = 0.5*(f aa + f bb) * step
          in helper2 step n f a b (aa+step) (bb+step) (summ+su) (count+1)


--GHCi> integration sin pi 0 ==== -2.0
