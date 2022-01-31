{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RankNTypes #-}
import Control.Applicative (ZipList (..), (<**>), Alternative (empty, (<|>)))
import Text.Parsec hiding ((<|>))
import Data.Char
import Data.Maybe
import Control.Monad.RWS.Lazy (Functor)
import Data.Either
import System.Directory.Internal.Prelude (Applicative, killThread)
--import Control.Applicative as Ca

--1.1.4
{-
В модуле Data.Functor определен оператор <$>, являющийся инфиксным аналогом функции fmap:
GHCi> :info <$>
(<$>) :: Functor f => (a -> b) -> f a -> f b
        -- Defined in `Data.Functor'
infixl 4 <$>
В выражении succ <$> "abc" этот оператор имеет тип (Char -> Char) -> [Char] -> [Char]. 
Какой тип имеет первое (левое) вхождение этого оператора в выражении succ <$> succ <$> "abc"?
-}
--([Char] -> [Char]) -> [[Char]] -> [[Char]] --wrong
--(Char -> Char)-> Char-> Char --wrong
--(Char -> Char) -> [Char] -> [Char] --wrong
--(Char -> Char) ->(Identity x)->(Identity x) --wrong
--Prelude> :t (succ <$> "abc") 
--(succ <$> "abc") :: [Char]
--(() -> Char) -> [] -> [Char] ; ( -> Char) -> () -> [Char] ; ( -> Char) -> () -> [Char] --wrong
--(Char -> Char-> Char) -> [Char] -> [Char] --wrong

--https://stepik.org/lesson/28880/step/4?discussion=370245&unit=9912
--На вики хаскеля дан совет, как узнать тип выражения: через лямбды.    
-- :t \x y z -> x <$> y <$> z :: [Char]

--Works:
-- 1. 
-- (a -> b) -> f a -> f b
-- 2.
-- (a -> b) :: Char -> Char 
-- f a :: Char -> Char 
-- => 
-- f b :: Char -> Char 
-- 3.
-- (a -> b) -> f a -> f b == (Char -> Char )->(Char -> Char )->(Char -> Char )

--1.1.5
{-
Сделайте типы данных Arr2 e1 e2 и Arr3 e1 e2 e3 представителями класса типов Functor:
newtype Arr2 e1 e2 a = Arr2 { getArr2 :: e1 -> e2 -> a }
newtype Arr3 e1 e2 e3 a = Arr3 { getArr3 :: e1 -> e2 -> e3 -> a }
Эти типы инкапсулируют вычисление с двумя и тремя независимыми окружениями соответственно:
GHCi> getArr2 (fmap length (Arr2 take)) 10 "abc"
3
GHCi> getArr3 (tail <$> tail <$> Arr3 zipWith) (+) [1,2,3,4] [10,20,30,40,50]
[33,44]
--
newtype Arr2 e1 e2 a = Arr2 {getArr2 :: e1 -> e2 -> a}
newtype Arr3 e1 e2 e3 a = Arr3 {getArr3 :: e1 -> e2 -> e3 -> a}
instance Functor (Arr2 e1 e2) where
  fmap = undefined
instance Functor (Arr3 e1 e2 e3) where
  fmap = undefined
-}

newtype Arr2 e1 e2 a = Arr2 {getArr2 :: e1 -> e2 -> a}

newtype Arr3 e1 e2 e3 a = Arr3 {getArr3 :: e1 -> e2 -> e3 -> a}

instance Functor (Arr2 e1 e2) where
  --fmap :: (a -> b) -> Arr2 e1 e2 a -> Arr2 e1 e2 b
  fmap g (Arr2 f) = Arr2 (\x y -> g (getArr2 (Arr2 f ) x y))
instance Functor (Arr3 e1 e2 e3) where
  fmap g (Arr3 f) = Arr3 (\x y z -> g (getArr3 (Arr3 f) x y z))

--1.1.7
{-
Самостоятельно докажите выполнение первого (fmap id = id) и второго ( fmap f . fmap g = fmap (f . g)) законов функторов 
для функтора частично примененной функциональной стрелки (->) e. 
Отметьте те свойства оператора композиции функций, которыми вы воспользовались.
Выберите все подходящие ответы из списка
+Композиция ассоциативна.
Композиция не коммутативна.
+id является левым нейтральным элементом для композиции.
id является правым нейтральным элементом для композиции. 
-}
--(a -> b) -> f a -> f b
--1)
-- fmap id = id
-- fmap id cont= cont
--fmap (->)e  id  x  = fmap id (e -> x)  --id y where e -> x = y
--2)
--fmap f . fmap g = fmap (f . g)
--fmap f ( fmap g cont) = fmap (f . g) cont
--2.1
--fmap f (fmap g cont) ; cont= (->) e
---fmap f (fmap g (e->x) )=  fmap f (e->(g x))= (e->f(g x))
--2.2
--fmap (f . g) cont ; cont= (->) e
---fmap (f . g)(e->x)=(e->((f . g)x))

--https://stepik.org/lesson/28880/step/7?discussion=3704797&reply=3721472&unit=9912
--Поскольку id f = f, то id f = f можно переписать так fmap id f = id f


--1.1.9
{-
Докажите выполнение второго закона функторов для функтора списка: fmap f (fmap g xs) = fmap (f . g) xs. 
Предполагается, что все списки конечны и не содержат расходимостей.
-}
{-
My:

1)[]
fmap f (fmap g []) =? fmap (f . g) []
left part:
fmap f (fmap g []])= fmap f []=[]
Right part:
fmap (f . g) [] = []
[] == []
2)(x:xs)
fmap f (fmap g (x:xs)) =? fmap (f . g) (x:xs)
Induction assumption: 
let fmap f (fmap g xs) == fmap (f . g) xs
left part:
fmap f (fmap g (x : xs))=fmap f ((fmap g x):(fmap g xs))=(fmap f (fmap g x)):(fmap f(fmap g xs))=(fmap (f.g) x):(fmap (f.g) xs))
Right part:
fmap (f . g) (x : xs) = (fmap (f . g) x):(fmap (f . g) xs)
(fmap (f . g) x) : (fmap (f . g) xs) == (fmap (f . g) x) : (fmap (f . g) xs)
-}
{-
 Комментарий от преподавателя

Доказываем индукцией по списку xs⁡ 

База индукции: xs⁡=[] 

fmap f (fmap g []) ≡ fmap f []  -- def fmap
                   ≡ []         -- def fmap

fmap (f . g) [] ≡ []  -- def fmap

 Обе части равенства равны одному и тому же.

Индукционный переход: xs⁡=y⁡:ys⁡  

fmap f (fmap g (y : ys)) ≡ fmap f (g y : fmap g ys))     -- def fmap
                         ≡ f (g y) : fmap f (fmap g ys)  -- def fmap

fmap (f . g) (y : ys) ≡ (f . g) y : fmap (f . g) ys  -- def fmap
                      ≡ f (g y) : fmap (f . g) ys    -- def (.)

По предположению индукции, для ys⁡ утверждение верно, значит обе части равны.
-}

{-
alien variant

(это база)

Левая часть равенства:

fmap f (fmap g []) -- def fmap

== fmap f []            -- def fmap

== []

Правая часть равенства:

fmap (f . g) []   -- def fmap

== []

таким образом получаем [] == [], что очевидно верно(свойство рефлексивности равенства)

(IH: fmap f (fmap g xs) = fmap (f . g) xs  верно для xs)

докажем, что если IH верна, то утверждение также верно и для x:xs

Левая часть равенства:

fmap f (fmap g (x:xs))    -- def fmap

== fmap f ( g x : fmap g xs) -- def fmap

== fmap f (g x) : fmap f (fmap g xs) -- def (.)

== fmap (f . g) x : fmap f (fmap g xs) -- IH

== fmap (f . g) x : fmap (f .g) xs

Правая часть равенства:

fmap (f . g)  (x: xs) -- def fmap

== fmap (f . g) x : fmap (f . g) xs

отсюда получаем, что левая и правая части равны. Данное утверждение в купе с базовым случаем говорят о том, 
что утверждение  fmap f (fmap g xs) = fmap (f . g) xs верно для списков произвольной длины.
-}

{-
Сделаем структурную индукцию: два случая: xs = nil, xs = x :: tail.

1) fmap f (fmap g nil) = fmap (f . g) nil

Так как forall f: fmap f nil = nil, средуцируем обе части:

fmap f nil = nil.

nil = nil, refl.

2) fmap f (fmap g (x :: tail)) = fmap (f . g) (x :: tail)

Так как forall f, tail: fmap f (x :: tail) = (f x) :: (fmap f tail), средуцируем обе части:

fmap f ((g x) :: fmap g tail) = (f(g(x)) :: fmap (f . g) tail), и ещё ра левую:

f(g(x)) :: fmap f (fmap g tail) = (f(g(x)) :: fmap (f . g) tail). Списки равны, если равны их головы и хвосты:

головы: f(g(x)) = f(g(x)), refl.

хвосты: fmap f (fmap g tail) = fmap (f . g) tail, а здесь можно применить предположение индукции для xs = tail.

qed.
-}

{-
my assesment:
(fmap f (fmap g x)) : (fmap f (fmap g xs)) = (fmap (f . g) x) : (fmap (f . g) xs)) 
хитро, но так не пойдёт) мы же доказываем закон, поэтому не можем использовать его же в доказательстве 
-}

--1.1.15
{-
Следующий тип данных задает гомогенную тройку элементов, которую можно рассматривать как трехмерный вектор:
data Triple a = Tr a a a  deriving (Eq,Show)
Сделайте этот тип функтором и аппликативным функтором с естественной для векторов семантикой покоординатного применения:
GHCi> (^2) <$> Tr 1 (-2) 3
Tr 1 4 9
GHCi> Tr (^2) (+2) (*3) <*> Tr 2 3 4
Tr 4 5 12
--
data Triple a = Tr a a a deriving (Eq, Show)

-}
data Triple a = Tr a a a deriving (Eq, Show)

instance Functor Triple  where
  --fmap :: (t -> a) -> Triple t -> Triple a
  fmap g (Tr x y z) = Tr (g x) (g y) (g z)

instance Applicative Triple where
  pure  a = Tr  a a a
  -- <*> :: Triple (a -> b) -> Triple a -> Triple b
  --(<*>) :: f(a->b) -> f a-> f b 
  --Triple (a -> b)-> Triple a->Triple b

  --(<*>) f (Tr x y z) = f (Tr x y z)  --Couldn't match expected type ‘Triple a -> Triple b’ with actual type ‘Triple (a -> b)’
  -- (<*>) f (Tr x y z) = pure f (Tr x y z) --Expected type: Triple a -> Triple b    Actual type: Triple a -> Triple (a -> b)
  --f <*> (Tr x y z) = fmap f (Tr x y z) --Couldn't match expected type ‘a -> b’            with actual type ‘Triple (a -> b)’
  (Tr m n k)<*> (Tr x y z) = Tr  (m x) (n y) (k z)

--
{-
-----------------------------------
Laws for Applicative functor:
1) Identity:
pure id <*> v == v 
2) Homomorphism:
pure g <*> pure x == pure (g x)
3) Interchange:
cont <*> pure x == pure ($ x) <*> cont
4) Composition:
pure (.) <*> u <*> v <*> cont == u <*> (v <*> cont)

----------------------------------
hmmm...
>*Main> :t ($ [])
>($ []) :: ([a] -> b) -> b
--
>Prelude> :t ($)
>($) :: (a -> b) -> a -> b
>Prelude> x=5
>Prelude> :t ($ x)
>($ x) :: Num a => (a -> b) -> b
--
>Prelude> f = (+2)
>Prelude> :t ($ f)
>($ f) :: Num a => ((a -> a) -> b) -> b
--
>Prelude> :t ($)
>($) :: (a -> b) -> a -> b
--
>Prelude> :t ($ x)
>($ x) :: Num a => (a -> b) -> b
--
>Prelude> :t (x $)
>(x $) :: Num (a -> b) => a -> b
-}


--
{-
Предположим, что для стандартного функтора списка оператор (<*>) определен стандартным образом, а метод pure изменен на
pure x = [x,x]
К каким законам класса типов Applicative будут в этом случае существовать контрпримеры?
Выберите все подходящие ответы из списка 
--
+Homomorphism : pure g <*> pure x ≡ pure (g x)
+Interchange : fs <*> pure x ≡ pure ($ x) <*> fs
+Identity : pure id <*> xs ≡ xs
+Applicative - Functor : g <$> xs ≡ pure g <*> xs
-Composition : (.) <$> us <*> vs <*> xs ≡ us <*> (vs <*> xs)
-}

--https://stepik.org/lesson/30424/step/3?discussion=3034417&unit=11041
{-
module First where

import Prelude hiding (Applicative, pure, (<*>))

infixl 4 <*>

class Functor f => Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

instance Applicative [] where
  pure x = [x, x]
  fs <*> xs = [f x | f <- fs, x <- xs]

x = 3
g = (* 2)
us = [(+ 7)]
vs = [(+ 9)]
fs = [\x -> 3 * x, \x -> 4 * x]
xs = [3, 4]
comp1 = (.) <$> us <*> vs <*> xs
comp2 = us <*> (vs <*> xs)
iden1 = pure id <*> xs
iden2 = xs
appfun1 = g <$> xs
appfun2 = pure g <*> xs
homo1 = pure g <*> pure x :: [Integer]
homo2 = pure (g x) :: [Integer]
inter1 = fs <*> pure x
inter2 = pure ($ x) <*> fs

main = do
  putStrLn ("Composition:\t" ++ show comp1 ++ " ≡  " ++ show comp2)
  putStrLn ("Identity:\t" ++ show iden1 ++ " ≡  " ++ show iden2)
  putStrLn ("Appl-Functor:\t" ++ show appfun1 ++ " ≡  " ++ show appfun2)
  putStrLn ("Homomorphism:\t" ++ show homo1 ++ " ≡  " ++ show homo2)
  putStrLn ("Interchange:\t" ++ show inter1 ++ " ≡  " ++ show inter2)
-}


--1.2.5
{-
В модуле Data.List имеется семейство функций zipWith, zipWith3, zipWith4,..:
GHCi> let x1s = [1,2,3]
GHCi> let x2s = [4,5,6]
GHCi> let x3s = [7,8,9]
GHCi> let x4s = [10,11,12]
GHCi> zipWith (\a b -> 2*a+3*b) x1s x2s
[14,19,24]
GHCi> zipWith3 (\a b c -> 2*a+3*b+5*c) x1s x2s x3s
[49,59,69]
GHCi> zipWith4 (\a b c d -> 2*a+3*b+5*c-4*d) x1s x2s x3s x4s
[9,15,21]

Аппликативные функторы могут заменить всё это семейство
GHCi> getZipList $ (\a b -> 2*a+3*b) <$> ZipList x1s <*> ZipList x2s
[14,19,24]
GHCi> getZipList $ (\a b c -> 2*a+3*b+5*c) <$> ZipList x1s <*> ZipList x2s <*> ZipList x3s
[49,59,69]
GHCi> getZipList $ (\a b c d -> 2*a+3*b+5*c-4*d) <$> ZipList x1s <*> ZipList x2s <*>ZipList x3s <*> ZipList x4s
[9,15,21]

Реализуйте операторы (>*<) и (>$<), позволяющие спрятать упаковку ZipList и распаковку getZipList:
GHCi> (\a b -> 2*a+3*b) >$< x1s >*< x2s
[14,19,24]
GHCi> (\a b c -> 2*a+3*b+5*c) >$< x1s >*< x2s >*< x3s
[49,59,69]
GHCi> (\a b c d -> 2*a+3*b+5*c-4*d) >$< x1s >*< x2s >*< x3s >*< x4s
[9,15,21]
--
import Control.Applicative (ZipList (ZipList), getZipList)
-}

{-
>*Main> :t (<*>) --infixl 4 <*> --application operation
>(<*>) :: Applicative f => f (a -> b) -> f a -> f b
--
>*Main> :t (<$>) --infixl 4 <$> -==fmap
>(<$>) :: Functor f => (a -> b) -> f a -> f b
-}

newtype ZipList1 a = ZipList1 {getZipList1 :: [a]} deriving Show

instance Functor ZipList1 where
  fmap f (ZipList1 xs) = ZipList1 (map f xs)

instance Applicative ZipList1 where
  pure x = ZipList1 [x]
  ZipList1 gs <*> ZipList1 xs = ZipList1 (zipWith ($) gs xs)

infixl 4 >*<, >$<
(>$<) f a = getZipList1 (f <$> (ZipList1 a))
(>*<) f a = getZipList1 $ (ZipList1 f) <*> (ZipList1 a)
{-
infixl 4 >*<, >$<
(>$<) :: (a1 -> a2) -> [a1] -> [a2]
(>$<) f a = getZipList1 (f <$> (ZipList1 a))
(>*<) :: [a->b] -> [a] -> ZipList b
(>*<) a b = (ZipList1 a) <*> (ZipList1 b)
>*Main> (\a b -> 2*a+3*b) >$< x1s >*< x2s
>ZipList1 {getZipList1 = [14,19,24]}
-}
x1s = [1,2,3]
x2s = [4,5,6]
x3s = [7,8,9]
x4s = [10,11,12]
--


--1.2.8
{-
Функция
divideList :: Fractional a => [a] -> a
divideList []     = 1
divideList (x:xs) = (/) x (divideList xs)
сворачивает список посредством деления. 
Модифицируйте ее, реализовав divideList' :: (Show a, Fractional a) => [a] -> (String,a), 
такую что последовательность вычислений отражается в логе:

GHCi> divideList [3,4,5]
3.75
GHCi> divideList' [3,4,5]
("<-3.0/<-4.0/<-5.0/1.0",3.75)

Используйте аппликативный функтор пары, сохраняя близкую к исходной функции структуру реализации
--
divideList' :: (Show a, Fractional a) => [a] -> (String,a)
divideList' []     = _
divideList' (x:xs) = (/) <$> _ <*> _

-}
divideList' :: (Show a, Fractional a) => [a] -> (String, a)
divideList' [] = ("1.0", 1)
divideList' (x : xs) = (/) <$> ("<-"++ (show x) ++ "/", x) <*> (divideList' xs)


--1.2.10
{-
Сделайте типы данных Arr2 e1 e2 и Arr3 e1 e2 e3 представителями класса типов Applicative
newtype Arr2 e1 e2 a = Arr2 { getArr2 :: e1 -> e2 -> a }
newtype Arr3 e1 e2 e3 a = Arr3 { getArr3 :: e1 -> e2 -> e3 -> a }
с естественной семантикой двух и трех окружений:
GHCi> getArr2 (Arr2 (\x y z -> x+y-z) <*> Arr2 (*)) 2 3
-1
GHCi> getArr3 (Arr3 (\x y z w -> x+y+z-w) <*> Arr3 (\x y z -> x*y*z)) 2 3 4
-15
--
newtype Arr2 e1 e2 a = Arr2 {getArr2 :: e1 -> e2 -> a}
newtype Arr3 e1 e2 e3 a = Arr3 {getArr3 :: e1 -> e2 -> e3 -> a}
instance Functor (Arr2 e1 e2) where
  fmap = undefined
instance Functor (Arr3 e1 e2 e3) where
  fmap = undefined
instance Applicative (Arr2 e1 e2) where
  pure = undefined
  (<*>) = undefined
instance Applicative (Arr3 e1 e2 e3) where
  pure = undefined
  (<*>) = undefined
-}
newtype Arr20 e1 e2 a0 = Arr20 {getArr20 :: e1 -> e2 -> a0}
newtype Arr30 e1 e2 e3 a0 = Arr30 {getArr30 :: e1 -> e2 -> e3 -> a0}

instance Functor (Arr20 e1 e2) where
  --a0:: e1 -> e2 -> a
  --f :: a0 -> b
  --fmap f (Arr20 a0) = Arr20 (\x y -> f x y a0) --Couldn't match type ‘a’ with ‘e1’
  fmap f (Arr20 a0) = Arr20 (\x y -> f (a0 x y))
instance Functor (Arr30 e1 e2 e3) where
  fmap f (Arr30 a0) = Arr30 (\x y z -> f (a0 x y z))
instance Applicative (Arr20 e1 e2) where
  --pure :: a -> Arr20 e1 e2 a
  --pure x = Arr20 (\e1 e2 -> e1 e2 x) --Couldn't match type ‘e1’ with ‘e2 -> a -> a’
  pure x = Arr20 (\e1 e2 ->  x)
  -- <*> :: Arr20 e1 e2 (a -> b) -> Arr20 e1 e2 a -> Arr20 e1 e2 b
  -- (<*>) (Arr20 f) (Arr20 a0) = Arr20 (f a0) -- Couldn't match type ‘a’ with ‘e2’
  --f :: e1 -> e2 -> a -> b
  --a0 :: e1 -> e2 -> a
  --(<*>) (Arr20 f) (Arr20 a0) = Arr20 (\e1 e2 -> f e1 e2  ) 
  --Occurs check: cannot construct the infinite type: b ~ a -> b
  --Expected type: Arr20 e1 e2 b    Actual type: Arr20 e1 e2 (a -> b)
  --(<*>) (Arr20 f) (Arr20 a0) = Arr20 (\e1 e2 -> f e1 e2 a0) --cannot construct the infinite type: a ~ e1 -> e2 -> a
  (<*>) (Arr20 f) (Arr20 a0) = Arr20 (\e1 e2 -> f e1 e2 (a0 e1 e2))
instance Applicative (Arr30 e1 e2 e3) where
  pure x = Arr30 (\e1 e2 e3-> x)
  (<*>) (Arr30 f) (Arr30 a0) = Arr30 (\e1 e2 e3 -> f e1 e2 e3 (a0 e1 e2 e3))


--https://stepik.org/lesson/30424/step/9?discussion=370598&unit=11041
{-
zip <*> tail довольно регулярно используемая композиция для получения соседних пар.
zip <*> tail $ [1..] == [(1, 2), (2, 3), ... ]
Например, треугольник Паскаля построить:
pascalTriangle = iterate nextRow [1]
    where nextRow xs = (zipWith (+)) <*> tail ([0] ++ xs ++ [0])
-}
--https://stackoverflow.com/questions/19181917/how-does-the-expression-ap-zip-tail-work
{-
So the type signature of ap is Monad m => m (a -> b) -> m a -> m b. 
You've given it zip and tail as arguments, so let's look at their type signatures.
Starting with tail :: [a] -> [a] ~ (->) [a] [a] (here ~ is the equality operator for types), 
if we compare this type against the type of the second argument for ap,
 (->) [x]  [x] ~ m a
((->) [x]) [x] ~ m a
we get a ~ [x] and m ~ ((->) [x]) ~ ((->) a). 
Already we can see that the monad we're in is (->) [x], not []. 
If we substitute what we can into the type signature of ap we get:
(((->) [x]) ([x] -> b)) -> (((->) [x]) [x]) -> (((->) [x]) b)
Since this is not very readable, it can more normally be written as
  ([x] -> ([x] -> b)) -> ([x] -> [x]) -> ([x] -> b)
~ ([x] ->  [x] -> b ) -> ([x] -> [x]) -> ([x] -> b)
The type of zip is [x] -> [y] -> [(x, y)]. 
We can already see that this lines up with the first argument to ap where
[x]         ~    [x]   
[y]         ~    [x]   
[(x, y)]    ~    b
Here I've listed the types vertically so that you can easily see which types line up. 
So obviously x ~ x, y ~ x, and [(x, y)] ~ [(x, x)] ~ b, 
so we can finish substituting b ~ [(x, x)] into ap's type signature and get
([x] -> [x] -> [(x, x)]) -> ([x] -> [x]) -> ([x] -> [(x, x)])
--   zip                        tail        ( ap  zip  tail )
--                                            ap  zip  tail u = zip u (tail u)
-}

--https://coderoad.ru/29361326/%D0%9A%D0%B0%D0%BA-%D1%87%D0%B8%D1%82%D0%B0%D1%82%D1%8C-Haskell-%D1%85-%D0%90%D0%9F-zip-%D1%85%D0%B2%D0%BE%D1%81%D1%82-%D1%87%D1%82%D0%BE%D0%B1%D1%8B-%D0%BE%D0%B7%D0%BD%D0%B0%D1%87%D0%B0%D1%82%D1%8C-%D1%85-zip-%D1%85-%D1%85%D0%B2%D0%BE%D1%81%D1%82-%D1%85
{-
instance Applicative ((->) a) where
pure = const
(<*>) f g x = f x (g x)
-}


--1.2.11
{-
Сопоставьте вычислению, поднятому в аппликативный функтор, конкретного представителя класса типов Applicative, 
в котором это вычисление происходит.
\xs -> pure (++) <*> lookup 3 xs <*> lookup 5 xs ____ Maybe
zip <*> tail ____ (->) a
pure zip <*> (Sum 5, [1, 2, 3]) <*> (Sum 4, [5, 6]) ____ (,) a
(,) <$> "dog" <*> "cat" ____ []
-}

--1.2.14
{-
Двойственный оператор аппликации (<**>) из модуля Control.Applicative изменяет направление вычислений, не меняя порядок эффектов:
infixl 4 <**>
(<**>) :: Applicative f => f a -> f (a -> b) -> f b
(<**>) = liftA2 (flip ($))
Определим оператор (<*?>) с той же сигнатурой, что и у (<**>), но другой реализацией:
infixl 4 <*?>
(<*?>) :: Applicative f => f a -> f (a -> b) -> f b
(<*?>) = flip (<*>)
Для каких стандартных представителей класса типов Applicative можно привести цепочку аппликативных вычислений, 
дающую разный результат в зависимости от того, какой из этих операторов использовался?
В следующих шести примерах вашей задачей будет привести такие контрпримеры для стандартных типов данных, 
для которых они существуют. Следует заменить аппликативное выражение в предложении in на выражение того же типа, 
однако дающее разные результаты при вызовах с (<??>) = (<**>) и (<??>) = (<*?>). 
Проверки имеют вид exprXXX (<**>) == exprXXX (<*?>) для различных имеющихся XXX. 
Если вы считаете, что контрпримера не существует, то менять ничего не надо.
--
{-# LANGUAGE RankNTypes #-}

import Control.Applicative (ZipList (..), (<**>))

infixl 4 <*?>

(<*?>) :: Applicative f => f a -> f (a -> b) -> f b
(<*?>) = flip (<*>)

exprMaybe :: (forall a b. Maybe a -> Maybe (a -> b) -> Maybe b) -> Maybe Int
exprMaybe op =
  let (<??>) = op
      infixl 4 <??>
   in Just 5 <??> Just (+ 2) -- place for counterexample

exprList :: (forall a b. [a] -> [a -> b] -> [b]) -> [Int]
exprList op =
  let (<??>) = op
      infixl 4 <??>
   in [1, 2] <??> [(+ 3), (+ 4)] -- place for counterexample

exprZipList :: (forall a b. ZipList a -> ZipList (a -> b) -> ZipList b) -> ZipList Int
exprZipList op =
  let (<??>) = op
      infixl 4 <??>
   in ZipList [1, 2] <??> ZipList [(+ 3), (+ 4)] -- place for counterexample

exprEither :: (forall a b. Either String a -> Either String (a -> b) -> Either String b) -> Either String Int
exprEither op =
  let (<??>) = op
      infixl 4 <??>
   in Left "AA" <??> Right (+ 1) -- place for counterexample

exprPair :: (forall a b. (String, a) -> (String, a -> b) -> (String, b)) -> (String, Int)
exprPair op =
  let (<??>) = op
      infixl 4 <??>
   in ("AA", 3) <??> ("", (+ 1)) -- place for counterexample

exprEnv :: (forall a b. (String -> a) -> (String -> (a -> b)) -> (String -> b)) -> (String -> Int)
exprEnv op =
  let (<??>) = op
      infixl 4 <??>
   in length <??> (\_ -> (+ 5)) -- place for counterexample
-}
--{-# LANGUAGE RankNTypes #-}
--import Control.Applicative (ZipList (..), (<**>))

infixl 4 <*?>

(<*?>) :: Applicative f => f a -> f (a -> b) -> f b
(<*?>) = flip (<*>)

exprMaybe :: (forall a b. Maybe a -> Maybe (a -> b) -> Maybe b) -> Maybe Int
exprMaybe op =
  let (<??>) = op
      infixl 4 <??>
   in Just 5 <??> Just (+ 2) -- place for counterexample   

exprList :: (forall a b. [a] -> [a -> b] -> [b]) -> [Int]
exprList op =
  let (<??>) = op
      infixl 4 <??>
   in [1, 2, 3] <??> [(+ 3), (+ 4)] -- place for counterexample
  --  *Main> [1, 2,3] <*?> [(+ 3), (+ 4)]
  --  [4,5,6,5,6,7]
  --  *Main> [1, 2,3] <**> [(+ 3), (+ 4)]
  --  [4,5,5,6,6,7]

exprZipList :: (forall a b. ZipList a -> ZipList (a -> b) -> ZipList b) -> ZipList Int
exprZipList op =
  let (<??>) = op
      infixl 4 <??>
   in ZipList [1, 2] <??> ZipList [(+ 3), (+ 4)] -- place for counterexample   

exprEither :: (forall a b. Either String a -> Either String (a -> b) -> Either String b) -> Either String Int
exprEither op =
  let (<??>) = op
      infixl 4 <??>
   in Left "AA" <??> Right (+ 1) -- place for counterexample   
  --  *Main> Left "AN" <**> Left "Jk"
  --  Left "AN"
  --  *Main> Left "AN" <*?> Left "Jk"
  --  Left "Jk"

exprPair :: (forall a b. (String, a) -> (String, a -> b) -> (String, b)) -> (String, Int)
exprPair op =
  let (<??>) = op
      infixl 4 <??>
   in ("AAi", 3) <??> ("", (+ 1)) -- place for counterexample
  --  *Main> ("AA", 3) <**> ("i", (+ 1))
  --  ("AAi",4)
  --  *Main> ("AA", 3) <*?> ("i", (+ 1))
  --  ("iAA",4)

exprEnv :: (forall a b. (String -> a) -> (String -> (a -> b)) -> (String -> b)) -> (String -> Int)
exprEnv op =
  let (<??>) = op
      infixl 4 <??>
   in length <??> (\_ -> (+ 5)) -- place for counterexample
   --reverse <*?>(\_ -> (++ "5")) $ "d6" --Nope
   --(!! 8) <*?> (\_ -> (+ 5))
   -- (head.lines) <*?> (\_ -> (++ "5i"))  $ "d6"

--1.3.3
{-
Какие из следующих примитивных парсеров имеются в библиотеке Text.Parsec.Char ? 
+Парсер, разбирающий в точности последовательность символов возврата каретки ('\r') и новой строки ('\n')
+Парсер, разбирающий в точности символ табуляции ('\t')
+Парсер, разбирающий в точности символ новой строки ('\n')
+Парсер, разбирающий произвольный символ
-Парсер, разбирающий в точности символ возврата каретки ('\r')
-Парсер, разбирающий в точности символ пробела (' ') 
-}

--1.3.5
{-


Реализуйте парсер getList, который разбирает строки из чисел, разделенных точкой с запятой, и возвращает список строк, 
представляющих собой эти числа:
GHCi> parseTest getList "1;234;56"
["1","234","56"]
GHCi> parseTest getList "1;234;56;"
parse error at (line 1, column 10):
unexpected end of input
expecting digit
GHCi> parseTest getList "1;;234;56"
parse error at (line 1, column 3):
unexpected ";"
expecting digit
Совет: изучите парсер-комбинаторы, доступные в модуле Text.Parsec, и постарайтесь найти наиболее компактное решение.
--
import Text.Parsec
getList :: Parsec String u [String]
getList = undefined
-}

--import Text.Parsec
getList :: Parsec String u [String]
getList = many1 digit `sepBy1` (char ';')

--1.3.7
{-
Используя аппликативный интерфейс Parsec, реализуйте функцию ignoreBraces, которая принимает три аргумента-парсера. 
Первый парсер разбирает текст, интерпретируемый как открывающая скобка, второй — как закрывающая, 
а третий разбирает весь входной поток, расположенный между этими скобками. 
Возвращаемый парсер возвращает результат работы третьего парсера, скобки игнорируются.
GHCi> test = ignoreBraces (string "[[") (string "]]") (many1 letter)
GHCi> parseTest test "[[ABC]]DEF"
"ABC"
--
import Text.Parsec
ignoreBraces :: Parsec [Char] u a -> Parsec [Char] u b -> Parsec [Char] u c -> Parsec [Char] u c
ignoreBraces = undefined
-}
--import Text.Parsec
ignoreBraces :: Parsec [Char] u a -> Parsec [Char] u b -> Parsec [Char] u c -> Parsec [Char] u c
--ignoreBraces =  many1 (char '[') <* many1 anyChar *> many1 (char ']')
--Couldn't match expected type ‘Parsec [Char] u a -> Parsec [Char] u b -> Parsec [Char] u c -> Parsec [Char] u c’
--with actual type ‘ParsecT s0 u0 m0 [Char]’
ignoreBraces a b c =  a *> (c <* b) --where
 --a = string "[[" --many1 (char '[')
 --b = string "]]" --many1 (char ']')
 --c = many1 letter --many1 (noneOf "]")

--1.4.4
{-
Предположим, тип парсера определен следующим образом:
newtype Prs a = Prs { runPrs :: String -> Maybe (a, String) }
Сделайте этот парсер представителем класса типов Functor. Реализуйте также парсер anyChr :: Prs Char, 
удачно разбирающий и возвращающий любой первый символ любой непустой входной строки.
GHCi> runPrs anyChr "ABC"
Just ('A',"BC")
GHCi> runPrs anyChr ""
Nothing
GHCi> runPrs (digitToInt <$> anyChr) "BCD"
Just (11,"CD")
--
instance Functor Prs where
  fmap = undefined

anyChr :: Prs Char
anyChr = undefined
-}
newtype Prs a = Prs {runPrs :: String -> Maybe (a, String)}

instance Functor Prs where
  --fmap f (Prs "") = Nothing
  --s:: String
  --fmap f (Prs s) = Prs (f s, s)
  --fmap f p = Prs( \s->Just (f n, m)) where
    --Just (n, m) = runPrs p s --Variable not in scope: s :: String
--
  --fmap f p = Prs k  where
      --k s = Just (f n, m)  where
          --Just (n, m) = runPrs p s
--          
    -- fmap f p = Prs k      where
    --     k s = Just (f n, m)          where
    --         Just (n, m) = runPrs p s
--Failed. "Haskell: test #5 failed (Functor test)"
--          
  -- fmap f p = Prs k where
  --   k s  |Just (n, m) == runPrs p s =Just (f n, m)
  --        |Nothing==runPrs p s  = Nothing
--  
  -- fmap f p = Prs k where  
  --   k s = if x == Nothing then Nothing else Just (f n, m)   where      
  --     x = runPrs  p  s
  --     Just (n, m) = x

 fmap f p = Prs k      where
    k s = if isNothing (runPrs p s) then Nothing else Just (f n, m)          where
        Just (n, m) = runPrs p s

anyChr :: Prs Char
anyChr = Prs f
  where
    f "" = Nothing
    f (c : cs) = Just (c, cs)

--https://stepik.org/lesson/30425/step/4?discussion=1887129&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs fun
    where
      fun "" = Nothing
      fun s = do
        (x, xs) <- runPrs p s
        Just (f x, xs)
-}

--https://stepik.org/lesson/30425/step/4?discussion=1450558&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs parse
    where
      parse s = fn (runPrs p s)
        where
          fn (Just (x, xs)) = Just (f x, xs)
          fn Nothing = Nothing
-}

--https://stepik.org/lesson/30425/step/4?discussion=367511&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs fun
    where
      fun s = (\(a, s') -> (f a, s')) <$> runPrs p s
-}

--https://stepik.org/lesson/30425/step/4?discussion=599555&thread=solutions&unit=11042
{-
import Control.Arrow (first)
instance Functor Prs where
  fmap f (Prs p) = Prs $ (fmap . fmap . first) f p ---???
-}

--https://stepik.org/lesson/30425/step/4?discussion=370809&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f (Prs p) = Prs fun
    where
      fun = fmap (fmap pf) p
        where
          pf (a, b) = (f a, b)
-}

--https://stepik.org/lesson/30425/step/4?discussion=603449&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs fun where 
      fun x = do (a, as) <- (runPrs p x)
                 Just (f a, as)
-}

--https://stepik.org/lesson/30425/step/4?discussion=1043401&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs fun
    where
      fun s = do
        (c, cs) <- runPrs p s
        return (f c, cs)
-}

--https://stepik.org/lesson/30425/step/4?discussion=372169&thread=solutions&unit=11042
{-
import Control.Arrow
instance Functor Prs where
  fmap f prs = Prs f'
    where
      f' s = first f <$> runPrs prs s ---???
-}

--https://stepik.org/lesson/30425/step/4?discussion=662972&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs $ \s -> (\(a, s') -> (f a, s')) <$> runPrs p s
-}

--https://stepik.org/lesson/30425/step/4?discussion=1450558&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs parse
    where
      parse s = fn (runPrs p s)
        where
          fn (Just (x, xs)) = Just (f x, xs)
          fn Nothing = Nothing
-}

--https://stepik.org/lesson/30425/step/4?discussion=520799&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs fp
    where
      fp = ((swap . (f <$>) . swap) <$>) <$> runPrs p
      swap = uncurry $ flip (,)      
-}

--https://stepik.org/lesson/30425/step/4?discussion=373997&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f (Prs g) = Prs $ \str -> case g str of
    Nothing -> Nothing
    Just (result, str') -> Just (f result, str')
-}

--https://stepik.org/lesson/30425/step/4?discussion=2807754&thread=solutions&unit=11042
{-
import           Data.Tuple (swap)

instance Functor Prs where
  fmap f parser = Prs $ fmap (swap . fmap f . swap) . runPrs parser

anyChr :: Prs Char
anyChr = Prs $ \s -> case s of
    []     -> Nothing
    (x:xs) -> Just $ (x, xs)
-}

--https://stepik.org/lesson/30425/step/4?discussion=1431801&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f p = Prs $ \s -> case (runPrs p) s of
    Nothing -> Nothing
    otherwise -> let Just (val, l) = (runPrs p) s in Just (f val, l)
-}

--https://stepik.org/lesson/30425/step/4?discussion=1412810&thread=solutions&unit=11042
{-
instance Functor Prs where
  fmap f prs = Prs (\s -> (\(a, str) -> (f a, str)) <$> runPrs prs s)
-}

--https://stepik.org/lesson/30425/step/4?discussion=1101779&thread=solutions&unit=11042
{-
{-# LANGUAGE MonadComprehensions #-}
{-# LANGUAGE TypeOperators #-}

instance Functor Prs where
  fmap f (Prs {runPrs = a}) = Prs {runPrs = \x -> [(f v, s) | (v, s) <- a x]}
-}

--https://stepik.org/lesson/30425/step/4?discussion=561226&thread=solutions&unit=11042
{-
import Control.Arrow (first)
instance Functor Prs where
  fmap = (Prs .) . flip ((.) . (.)) runPrs . fmap . first
-}

--https://stepik.org/lesson/30425/step/4?discussion=1578342&thread=solutions&unit=11042
{-
import Control.Arrow (first)

instance Functor Prs where
  fmap f p = Prs $ (first f <$>) <$> runPrs p
-}

--1.4.6
{-
Сделайте парсер
newtype Prs a = Prs { runPrs :: String -> Maybe (a, String) }
из предыдущей задачи аппликативным функтором с естественной для парсера семантикой:
GHCi> runPrs ((,,) <$> anyChr <*> anyChr <*> anyChr) "ABCDE"
Just (('A','B','C'),"DE")
GHCi> runPrs (anyChr *> anyChr) "ABCDE"
Just ('B',"CDE")
Представитель для класса типов Functor уже реализован.
--
instance Applicative Prs where
  pure = undefined
  (<*>) = undefined
-}

--newtype Prs a = Prs {runPrs :: String -> Maybe (a, String)}

instance Applicative Prs where
  pure x=  Prs (\s->Just (x, s))
  --(<*>) f a = Prs (runPrs f <*>runPrs a)
  --  Expected type: String -> Maybe (a, String) -> Maybe (b, String)
  --  Actual type: String -> Maybe (a -> b, String)

  --(<*>) f a = Prs (runPrs f) <*>  Prs(runPrs a)
  --GHCi> runPrs ((,,) <$> anyChr <*> anyChr <*> anyChr) "ABCDE" 
  --STACKOVERFLOW!!!!!!!!!

  --(<*>) f a = Prs (\s -> (Just (fst (runPrs f), s)) (Just (fst (runPrs a), s)))   


  -- (<*>) f a = Prs t where
  --   t s= Just(fx ax, s2) where
  --    Just (fx, s1) = runPrs f s
  --    Just (ax, s2) = runPrs a s1
  --Failed. "Haskell: test #5 failed"

  (<*>) f a = Prs t where
    t s = case f1 of
      Nothing -> Nothing
      Just (fx, s1) ->case f2 of
        Nothing -> Nothing
        Just (ax, s2) ->Just (fx ax, s2)
      where
        f1 = runPrs f s
        f2 = runPrs a s1
        Just (fx, s1) = f1
        Just (ax, s2) = f2

--https://stepik.org/lesson/30425/step/6?discussion=367516&thread=solutions&unit=11042
{-
Работает для любого контейнера результатов разбора, являющегося монадой. А есть ли элегантный способ сделать это, используя только Applicative? Для pure можно сделать

instance Applicative Prs where
  pure a = Prs func where
    func s = pure (a, s)

Но как при этом сделать <*>?

instance Applicative Prs where
  pure a = Prs func
    where
      func s = return (a, s)
  pf <*> pv = Prs func
    where
      func s = do
        (g, s') <- runPrs pf s
        (a, s'') <- runPrs pv s'
        return (g a, s'')
  -- https://stepik.org/lesson/30425/step/6?discussion=367516&reply=367646&thread=solutions&unit=11042
   u <*> v = Prs f
     where
       f s = case runPrs u s of
         Nothing -> Nothing
         Just (g, s') -> runPrs (g <$> v) s'
-}

--https://stepik.org/lesson/30425/step/6?discussion=374101&thread=solutions&unit=11042
{-
Когда мы узнали, что слева получилась не ошибка, а функция, дальше применить её уже очень просто.

instance Applicative Prs where
  pure x = Prs $ \s -> Just (x, s)
  pf <*> px = Prs $ \s -> case runPrs pf s of
                            Nothing -> Nothing
                            Just (f, s') -> runPrs (f <$> px) s'
-}

--https://stepik.org/lesson/30425/step/6?discussion=641491&thread=solutions&unit=11042
{-
instance Applicative Prs where
  pure a = Prs fn
    where
      fn s = Just (a, s)
  pf <*> pv = Prs fn
    where
      fn s = do
        (g, s') <- runPrs pf s
        runPrs (g <$> pv) s'
-}

--https://stepik.org/lesson/30425/step/6?discussion=534296&thread=solutions&unit=11042
{-
instance Applicative Prs where
  pure x = Prs $ \str -> Just (x, str)
  af <*> av = Prs $ \str -> do
    (f, str') <- runPrs af str
    (v, str'') <- runPrs av str'
    return (f v, str'')
-}

--https://stepik.org/lesson/30425/step/6?discussion=947412&thread=solutions&unit=11042
{-
instance Applicative Prs where
  pure a = Prs $ \s -> Just (a, s)
  pf <*> px = Prs p
    where
      p s = runPrs pf s >>= \(f, s') -> runPrs px s' >>= \(x, s'') -> return (f x, s'')
-}

--1.4.8
{-
Рассмотрим более продвинутый парсер, позволяющий возвращать пользователю причину неудачи при синтаксическом разборе:
newtype PrsE a = PrsE { runPrsE :: String -> Either String (a, String) }
Реализуйте функцию satisfyE :: (Char -> Bool) -> PrsE Char таким образом, чтобы функция
charE :: Char -> PrsE Char
charE c = satisfyE (== c)
обладала бы следующим поведением:
GHCi> runPrsE (charE 'A') "ABC"
Right ('A',"BC")
GHCi> runPrsE (charE 'A') "BCD"
Left "unexpected B"
GHCi> runPrsE (charE 'A') ""
Left "unexpected end of input"
-}
newtype PrsE a = PrsE {runPrsE :: String -> Either String (a, String)}
satisfyE :: (Char -> Bool) -> PrsE Char
satisfyE f = PrsE t where
  t (c : cs)
   | f c = Right (c, cs)
   | otherwise = Left ("unexpected "++[c])
  t ""  = Left "unexpected end of input"

--  f (c : cs)
--    | (f c) = Right (c, cs)
--    | c: cs == "" = Left "unexpected end of input"
--    | otherwise = Left "unexpected "++c
charE :: Char -> PrsE Char
charE c = satisfyE (== c)

--1.4.8
{-
Сделайте парсер
newtype PrsE a = PrsE { runPrsE :: String -> Either String (a, String) }
из предыдущей задачи функтором и аппликативным функтором:
GHCi> let anyE = satisfyE (const True)
GHCi> runPrsE ((,) <$> anyE <* charE 'B' <*> anyE) "ABCDE"
Right (('A','C'),"DE")
GHCi> runPrsE ((,) <$> anyE <* charE 'C' <*> anyE) "ABCDE"
Left "unexpected B"
GHCi> runPrsE ((,) <$> anyE <* charE 'B' <*> anyE) "AB"
Left "unexpected end of input"
-}

--runPrsE :: PrsE a -> String -> Either String (a, String)
instance Functor PrsE where
  fmap f a = PrsE k where
    k "" = Left "unexpected end of input"
    k s = do
      (a1,s1) <- runPrsE a s
      return (f a1,s1)

instance Applicative PrsE where
 pure a = PrsE (\s ->  Right (a, s))
 --(<*>) f a = f -- cannot construct the infinite type: b ~ a -> b  ___ Expected type: PrsE b   ___ Actual type: PrsE (a -> b)
 --(<*>) f a = PrsE (\s -> Right ( (runPrsE f s) a,s) )

 {-- right for Rigt: --
 (<*>) f a = PrsE t   where
     t s = Right ((fst k) (fst m), snd m)       where
         Right k = runPrsE f s
         Right m = runPrsE a (snd k)
 -}
{-
(<*>) f a = PrsE t
  where
    t s
      | isLeft ff && isLeft aa = Left n
      | isLeft ff || isLeft aa = Left nn
      | otherwise = Right ((fst k) (fst m), snd m)
      where
        ff = runPrsE f s
        aa = runPrsE a s --(snd k)
        Left n = ff
        Left nn = aa
        Right k = ff
        Right m = aa
-}
{-
*Main> runPrsE ((,) <$> anyE <* charE 'B' <*> anyE) "ABCDE"
Right (('A','A'),"ABCDE")
-}

{-+++++++++
fs <*> as ' can be understood as the do expression
 do f <- fs
    a <- as
    pure (f a)
-}

 (<*>) f a = PrsE t where
    t s
      | isLeft ff = Left n-- && isLeft aa = n 
      | isLeft aa = Left nn
      | otherwise = Right ((fst k) (fst m), snd m)       where
          ff = runPrsE f s
          aa = runPrsE a (snd k)
          Left n = ff
          Left nn = aa
          Right k = ff
          Right m = aa

anyE :: PrsE Char
anyE = satisfyE (const True)

--https://stepik.org/lesson/30425/step/9?discussion=375100&thread=solutions&unit=11042
{-
  pf <*> pv = PrsE $ \s -> do
    (f, s') <- runPrsE pf s
    (v, s'') <- runPrsE pv s'
    return (f v, s'')
-}

--https://stepik.org/lesson/30425/step/9?discussion=364465&thread=solutions&unit=11042
{-
newtype PrsE a = PrsE {runPrsE :: String -> Either String (a, String)}

instance Functor PrsE where
  fmap f = PrsE . (fmap . fmap $ \(a, s) -> (f a, s)) . runPrsE

instance Applicative PrsE where
  pure x = PrsE $ \s -> Right (x, s)
  pf <*> px = PrsE $ \s -> case runPrsE pf s of
    Left e -> Left e
    Right (f, s') -> runPrsE (f <$> px) s'
-}

--https://stepik.org/lesson/30425/step/9?discussion=373317&thread=solutions&unit=11042
{-
(<*>) pf pv = PrsE fun where
      fun s = either Left fwd $ runPrsE pf s where
              fwd (f,s2) = runPrsE (f <$> pv) s2
-}

--1.4.12
{-
Сделайте парсер
newtype Prs a = Prs { runPrs :: String -> Maybe (a, String) }
представителем класса типов Alternative с естественной для парсера семантикой:
GHCi> runPrs (char 'A' <|> char 'B') "ABC"
Just ('A',"BC")
GHCi> runPrs (char 'A' <|> char 'B') "BCD"
Just ('B',"CD")
GHCi> runPrs (char 'A' <|> char 'B') "CDE"
Nothing
Представители для классов типов Functor и Applicative уже реализованы. 
Функцию char :: Char -> Prs Char включать в решение не нужно, но полезно реализовать для локального тестирования.
--
instance Alternative Prs where
  empty = undefined
  (<|>) = undefined
-}

--newtype Prs a = Prs {runPrs :: String -> Maybe (a, String)}
instance Alternative Prs where
  empty = Prs (\s-> Nothing)
  (<|>) f g = Prs  k where --do
    k s
     |isJust (runPrs f s) = runPrs f s--fst x
     |isJust (runPrs g s) = runPrs g s--fst y
     |otherwise  = Nothing where
       Just x = runPrs f s
       Just y = runPrs g s

      --x<- runPrs f s
      --y<- runPrs g s
      --return (if isNothing x then y else x)

--- ????????
-- instance Show (Prs a) where
--   show (Prs a) =  case (runPrs a s) of --let k = (\s -> a) in case runPrs k of
--     Just (n,s) -> "Just"-- ++  show (n,s) 
--     Nothing -> "Nothing"
--- ???????

--https://stepik.org/lesson/30425/step/12?discussion=1452062&thread=solutions&unit=11042
{-
char через уже известный нам satisfy

satisfy :: (Char -> Bool) -> Prs Char
satisfy pr = Prs f
  where
    f "" = Nothing
    f (c : cs)
      | pr c = Just (c, cs)
      | otherwise = Nothing

char :: Char -> Prs Char
char c = satisfy (== c)

instance Alternative Prs where
  empty = Prs f
    where
      f _ = Nothing

  (<|>) p q = Prs f
    where
      f s = fn (runPrs p s)
        where
          fn Nothing = runPrs q s
          fn a = a
-}

--https://stepik.org/lesson/30425/step/12?discussion=370976&thread=solutions&unit=11042
{-
Раз Maybe сам является представителем класса типов Alternative, то эти можно воспользоваться.

instance Alternative Prs where
  empty = Prs $ const Nothing
  l <|> r = Prs f where
    f s = runPrs l s <|> runPrs r s

-}

--https://stepik.org/lesson/30425/step/12?discussion=373377&thread=solutions&unit=11042
{-
Пользуясь подсказкой из комментов, используем Applicativeность стрелки и Alternativeность Maybe.

instance Alternative Prs where
  empty = Prs $ const Nothing
  (<|>) pa pb = Prs $ (<|>) <$> (runPrs pa) <*> (runPrs pb)

-}

--https://stepik.org/lesson/30425/step/12?discussion=624618&thread=solutions&unit=11042
{-
Реализуем вручную, пользуясь функцией maybe

instance Alternative Prs where
    empty = Prs $ const Nothing
    Prs px <|> Prs py = Prs $ \s -> maybe (py s) Just (px s)

-}

--https://stepik.org/lesson/30425/step/12?discussion=3336285&thread=solutions&unit=11042
{-
Не увидел варианта с liftM2. Оставлю свой.

import Control.Monad

instance Alternative Prs where
  empty = Prs $ const Nothing
  (Prs f1) <|> (Prs f2) = Prs $ liftM2 (<|>) f1 f2

-}

--1.4.14
{-
Реализуйте для парсера
newtype Prs a = Prs { runPrs :: String -> Maybe (a, String) }
парсер-комбинатор many1 :: Prs a -> Prs [a], который отличается от many только тем, что он терпит неудачу в случае, 
когда парсер-аргумент неудачен на начале входной строки.
> runPrs (many1 $ char 'A') "AAABCDE"
Just ("AAA","BCDE")
> runPrs (many1 $ char 'A') "BCDE"
Nothing
Функцию char :: Char -> Prs Char включать в решение не нужно, но полезно реализовать для локального тестирования.
--
many1 :: Prs a -> Prs [a]
many1 = undefined
-}
many11 :: Prs a -> Prs [a]
--many11 p = Prs (\s -> (Just (pure (:) <*> p <*> many11 p, s)) <|> Nothing) --empty
-- +++ many11 p =  pure (:) <*> p <*> many11 p <|> pure [] --<|> (Prs (\s -> Nothing))
many11 p = pure (:) <*> p <*> m p where
  m p= pure (:) <*> p <*> m p <|> pure []
 --k = pure (:) <*> (p <|> Prs (\[]-> Nothing)) <*> many11 p  <|> pure [] -- <|> Prs (\s -> Nothing))
 -- *Main> runPrs (many11 $ char1 'A') "AAABCDE"      *** Exception: s1_1.hs:1350:31-43: Non-exhaustive patterns in lambda
 
 --k = (pure (:) <*> p <*> (Prs ( \_ -> Nothing))) <|> (pure (:) <*> p <*> many11 p <|> pure []) --(Prs (\_ -> Nothing)) --(pure (:) <*> p <*> many11 p)
 --k = pure (:) <*> ( Prs (\_ -> Nothing)) <*> many11 p <|> m  where m =pure [] --(pure (\"" -> []) <*> Prs (\_ -> Nothing))
 --where --Prs (k ) where --Prs (\s ->  (runPrs p s) <*> many11 p <|> pure Nothing)
  --k s = Just ((pure (:) <*> r), s1)
 --k s = Just  ((pure (:) <*>r),s1) <|> Nothing where
  --Just (r,s1) = runPrs p s
 --m= k<|> Nothing-- pure (:) <*> k

char1 :: Char -> Prs Char
char1 c = Prs k where
  k "" = Nothing
  k (a:as)
   |c==a = Just (c, as)
   |otherwise = Nothing
{-
*Main> runPrs (char1 'A') "ABC"
Just ('A',"BC")
*Main> runPrs (char1 'A') "BC"
Nothing
*Main> runPrs (char1 'A') ""
Nothing
-}

--https://stepik.org/lesson/30425/step/14?discussion=369866&thread=solutions&unit=11042
{-
many реализован для Alternative, а не для Parser, а поэтому доступен и тут.
Собственно, many1 можно тоже сразу для Alternative реализовать.
many1 :: Alternative f => f a -> f [a]
many1 p = (:) <$> p <*> many p
-}

--https://stepik.org/lesson/30425/step/14?discussion=369866&reply=369908&thread=solutions&unit=11042
{-
Можно еще короче написать
many1 :: Prs a -> Prs [a]
many1 = some
-}

--https://stepik.org/lesson/30425/step/14?discussion=369500&thread=solutions&unit=11042
{-
many1 :: Prs a -> Prs [a]
many1 p = (:) <$> p <*> (many1 p <|> pure [])
-}

--https://stepik.org/lesson/30425/step/14?discussion=371494&thread=solutions&unit=11042
{-
many1 :: Prs a -> Prs [a]
many1 p = (:) <$> p <*> many p <|> empty
-}

--https://stepik.org/lesson/30425/step/14?discussion=368733&thread=solutions&unit=11042
{-
many1 :: Prs a -> Prs [a]
many1 p = ((:) <$> p) <*> (many p)
-}

--https://stepik.org/lesson/30425/step/14?discussion=374533&thread=solutions&unit=11042
{-
many1 :: Prs a -> Prs [a]
many1 p = liftA2 (:) p (many p)
-}

--https://stepik.org/lesson/30425/step/14?discussion=587957&thread=solutions&unit=11042
{-
many1 :: Prs a -> Prs [a]
many1 a =
  let x = (:) <$> a <*> many
      many = x <|> pure []
   in x
-}

--https://stepik.org/lesson/30425/step/14?discussion=371906&thread=solutions&unit=11042
{-
many1 :: Prs a -> Prs [a]
many1 p =
  Prs
    ( \s -> do
        res <- fn s
        if snd res == s
          then Nothing
          else return res
    )
  where
    fn "" = pure (empty, "")
    fn s = (\prs (acc, rem) -> ((fst prs) : acc, rem)) <$> runPrs p s <*> (fn $ tail s) <|> pure (empty, s)
-}


--1.4.15
{-
Реализуйте парсер nat :: Prs Int для натуральных чисел, так чтобы парсер
mult :: Prs Int
mult = (*) <$> nat <* char '*' <*> nat
обладал таким поведением
GHCi> runPrs mult "14*3"
Just (42,"")
GHCi> runPrs mult "64*32"
Just (2048,"")
GHCi> runPrs mult "77*0"
Just (0,"")
GHCi> runPrs mult "2*77AAA"
Just (154,"AAA")
Реализацию функции char :: Char -> Prs Char следует включить в присылаемое решение, только если она нужна для реализации парсера nat.
--
nat :: Prs Int
nat = undefined
-}

--newtype Prs a = Prs {runPrs :: String -> Maybe (a, String)}

nat1 :: Prs Char
nat1 = Prs k where
 k "" = Nothing
 k (c : cs) 
   | isDigit c = Just (c, cs) -- \n -> if isDigit n then Prs (digitToInt n) else Prs Nothing
   | otherwise = Nothing

nat :: Prs Int
nat = Prs  r  where
   --Prs  n  where --k <*> nat where
  r s
   |isJust p= Just (read (fst b):: Int, snd b)
   |otherwise = Nothing where
     p = runPrs (many11 nat1) s
     Just b= p
  --p = (:) <$> p <*> many p

mult :: Prs Int
mult = (*) <$> nat <* char1 '*' <*> nat

--https://stepik.org/lesson/30425/step/15?discussion=371001&thread=solutions&unit=11042
{-
import Data.Char (isDigit)

many1 :: Prs a -> Prs [a]
many1 p = (:) <$> p <*> many p

satisfy :: (Char -> Bool) -> Prs Char
satisfy p = Prs f
  where
    f "" = Nothing
    f (c : cs)
      | p c = Just (c, cs)
      | otherwise = Nothing

digit :: Prs Char
digit = satisfy isDigit

digits :: Prs String
digits = many1 digit

nat :: Prs Int
nat = fmap read digits
-}

--https://stepik.org/lesson/30425/step/15?discussion=470558&thread=solutions&unit=11042
{-
import Data.Char

nat :: Prs Int
nat = Prs $ \s ->
  let (ns, rest) = span isDigit s
   in case ns of
        [] -> Nothing
        _ -> Just (read ns, rest)
-}

--1.5.3 
{-
Населите допустимыми нерасходящимися выражениями следующие типы
type A = ((,) Integer |.| (,) Char) Bool
type B t = ((,,) Bool (t -> t) |.| Either String) Int
type C = (|.|) ((->) Bool) ((->) Integer) Integer
--
type A = ((,) Integer |.| (,) Char) Bool
type B t = ((,,) Bool (t -> t) |.| Either String) Int
type C = (|.|) ((->) Bool) ((->) Integer) Integer
a :: A
a = undefined
b :: B t
b = undefined
c :: C
c = undefined
-}
infixr 9 |.|
newtype (|.|) f g a = Cmps {getCmps :: f (g a)}   deriving (Eq, Show)

type A   = ((,) Integer |.| (,) Char) Bool
type B t = ((,,) Bool (t -> t) |.| Either String) Int
type C   = (|.|) ((->) Bool) ((->) Integer) Integer

--i = 9 :: Integer

a :: A
a = Cmps (6, ('h', True))

b :: B t
b = Cmps (True,\t->t ,Left "gg")

c :: C
boo::Bool 
boo=True 
c = Cmps (\boo -> (\i -> i*9))

--https://stepik.org/lesson/30426/step/3?discussion=374605&thread=solutions&unit=11043
--c  = Cmps $ flip const

--https://stepik.org/lesson/30426/step/3?discussion=371024&thread=solutions&unit=11043
--Cmps seq

--https://stepik.org/lesson/30426/step/3?discussion=2149525&thread=solutions&unit=11043
--c  = Cmps $ \b -> \x -> x
-- +++++ -- Используя type holes вида "a = Cmps $ _", компилятор выводит читаемые типы

--1.5.5
{-
Сделайте тип
newtype Cmps3 f g h a = Cmps3 { getCmps3 :: f (g (h a)) } 
  deriving (Eq,Show) 
представителем класса типов Functor при условии, что первые его три параметра являются функторами:
GHCi> fmap (^2) $ Cmps3 [[[1],[2,3,4],[5,6]],[],[[7,8],[9,10,11]]]
Cmps3 {getCmps3 = [[[1],[4,9,16],[25,36]],[],[[49,64],[81,100,121]]]}
--
newtype Cmps3 f g h a = Cmps3 {getCmps3 :: f (g (h a))}
  deriving (Eq, Show)
instance Functor (Cmps3 f g h) where
  fmap = undefined
-}
newtype Cmps3 f g h a = Cmps3 {getCmps3 :: f (g (h a))}
  deriving (Eq, Show)

instance (Functor f, Functor g, Functor h) => Functor (Cmps3 f g h) where
  --fmap :: (a -> b) -> Cmps3 f g h a -> Cmps3 f g h b
  -- x:: a->b
  -- i ::  (g (h a)) -> (g (h b))
  -- j :: h a -> h b
  -- y::f (g (h a))
  --
  fmap x (Cmps3 y) = Cmps3 $ fmap (fmap (fmap x)) y
  

--https://stepik.org/lesson/30426/step/4?discussion=371338&unit=11043
{-
Интересно, что эту штуку можно записать просто как fmap . fmap, если у нас обычная вложенность контейнеров,
 а не через специальное произведение. Аналогично можно сделать и для тройной вложенности.
Но при этом компилятор не справляется с динамическим выражением
fmapDeep depth = foldl1 (.) $ replicate depth fmap
-}

--https://stepik.org/lesson/30426/step/5?discussion=549748&thread=solutions&unit=11043
{-
newtype Cmps3 f g h a = Cmps3 {getCmps3 :: f (g (h a))}
  deriving (Eq, Show)

instance (Functor f, Functor g, Functor h) => Functor (Cmps3 f g h) where
  fmap = (Cmps3 .) . (. getCmps3) . fmap . fmap . fmap
-}

--https://stepik.org/lesson/30426/step/5?discussion=557555&thread=solutions&unit=11043
{-
newtype Cmps3 f g h a = Cmps3 {getCmps3 :: f (g (h a))}
  deriving (Eq, Show)

instance (Functor f, Functor g, Functor h) => Functor (Cmps3 f g h) where
  fmap p (Cmps3 x) = Cmps3 $ (fmap . fmap . fmap) p x
-}

--https://stepik.org/lesson/30426/step/5?discussion=504615&thread=solutions&unit=11043
{-
newtype Cmps3 f g h a = Cmps3 {getCmps3 :: f (g (h a))}
  deriving (Eq, Show)

instance (Functor f, Functor g, Functor h) => Functor (Cmps3 f g h) where
  {-
  fmap :: (a -> b) -> f a -> f b = (a -> b) -> Cmps3 f g h a -> Cmps3 f g h b
  fmap m x = Cmps3 $ omega m x = Cmps3 $ fmap (fmap (fmap m)) x

  m :: (a -> b)
  x :: f (g (h a)) -> f (g (h b))

  ksi :: (a -> b) -> h a -> h b
  ksi = fmap

  phi :: (a -> b) -> g (h a) -> g (h b)
  phi m = fmap (ksi m)

  omega :: (g (h a) -> g (h b)) -> f (g (h a)) -> f (g (h b))
  omega m x = fmap (phi m) x = fmap (fmap (ksi m)) x = fmap (fmap (fmap m)) x
  -}
  fmap m (Cmps3 x) = Cmps3 $ fmap (fmap (fmap m)) x
-}

--https://stepik.org/lesson/30426/step/5?discussion=668976&thread=solutions&unit=11043
{-
newtype Cmps3 f g h a = Cmps3 {getCmps3 :: f (g (h a))} deriving (Eq, Show)

instance (Functor f, Functor g, Functor h) => Functor (Cmps3 f g h) where
  fmap fn = Cmps3 . (fmap . fmap . fmap $ fn) . getCmps3
-}

--https://stepik.org/lesson/30426/step/5?discussion=1364465&thread=solutions&unit=11043
{-
newtype Cmps3 f g h a = Cmps3 {getCmps3 :: f (g (h a))}
  deriving (Eq, Show)

instance (Functor f, Functor g, Functor h) => Functor (Cmps3 f g h) where
  fmap c (Cmps3 x) = Cmps3 $ (fmap . fmap . fmap) c x
-}



--1.5 
{-
Докажите выполнение второго закона функторов для композиции двух функторов:
fmap h2 (fmap h1 (Cmps x)) = fmap (h2 . h1) (Cmps x).
-}
--fmap h (Cmps x) = Cmps $ fmap (fmap h) x

{-
(1) fmap id cont = id cont -- => fmap id == id
fmap id (Cmps x)           -- def fmap (Cmps)
== Cmps $ fmap (fmap id) x -- (1) fmap (g)
== Cmps $ fmap id x        -- (1) fmap (f)
== Cmps x

-}


--fmap h (Cmps x) = Cmps $ fmap (fmap h) x   --(1)

--fmap h2 (fmap h1 (Cmps x)) = fmap (h2 . h1) (Cmps x) 

--left part
--fmap h2 (fmap h1 (Cmps x)) == fmap h2 (Cmps $ fmap (fmap h1) x) --according to (1)
-- == Cmps $ (h2 ( fmap (fmap h1) x)) == Cmps $ h2((fmap h1) x))
-- == Cmps $ (h2 (h1 x)) == Cmps $ (h2.h1) x

--right part
--fmap (h2 . h1) (Cmps x) == Cmps $ fmap (fmap (h2.h1)) x) -- according to (1)
-- == Cmps $ fmap (fmap (h2.h1)) x) == Cmps $ (fmap (h2.h1)) x 
-- == Cmps $ (h2.h1) x 

{-
 Комментарий от преподавателя
Назовём наши функторы F⁡ \operatorname{F} F и G⁡ \operatorname{G} G, то есть
Cmps x :: (F |.| G) a
Для удобства перепишем левую часть закона через композицию. Теперь требуется проверить, что:
(fmap h2 . fmap h1) (Cmps x) = fmap (h2 . h1) (Cmps x)
для произвольных
x  :: F (G a)
h1 :: a -> b
h2 :: b -> c
Проверяем:
fmap h2 (fmap h1 (Cmps x)) ≡ fmap h2 (Cmps $ fmap (fmap h1) x)  -- def fmap
                           ≡ Cmps $ fmap (fmap h2) (fmap (fmap h1) x)  -- def fmap
                           ≡ Cmps $ (fmap (fmap h2) . fmap (fmap h1)) x  -- def (.)
                           = Cmps $ fmap (fmap h2 . fmap h1) x  -- Functor F
                           = Cmps $ fmap (fmap (h2 . h1)) x  -- Functor G
fmap (h2 . h1) (Cmps x) ≡ Cmps $ fmap (fmap (h2 . h1)) x  -- def fmap
-}

{-
Для доказательства воспользуемся следующими определениями: 

def fmap для f |.| g

instance (Functor f, Functor g) => Functor (f |.| g) where
  fmap f (Cmps x) = Cmps $ fmap (fmap f) x

def (.)

f . g = \y -> f (g y)

Требуется доказать, что 

fmap h2 (fmap h1 (Cmps x)) = fmap (h2 . h1) (Cmps x)

Доказательство: 

fmap h2 (fmap h1 (Cmps x))                               -- (1)
  ≡ fmap h2 (Cmps $ fmap (fmap h1) x)                    -- def fmap для f |.| g
  ≡ Cmps $ fmap (fmap h2) (fmap (fmap h1) x)             -- def fmap для f |.| g 

fmap (h2 . h1) (Cmps x)                                   -- (2)
  ≡ Cmps $ fmap (fmap (h2 . h1)) x                        -- def fmap для f |.| g
  ≡ Cmps $ fmap (fmap h2 . fmap h1) x                     -- Второй закон функторов для функтора g (второй операнд композиции типов) 
  ≡ Cmps $ (fmap (fmap h2) . fmap (fmap h1)) x            -- Второй закон функторов для функтора f (первый операнд композиции типов)
  ≡ Cmps $ (\y -> fmap (fmap h2) (fmap (fmap h1) y)) x    -- def (.)
  ≡ Cmps $ fmap (fmap h2) (fmap (fmap h1) x)              -- β-редукция

(1) ≡ (2), что требовалось доказать 
-}

{-
-- fmap law
fmap f (fmap g x) == fmap (f . g) x
fmap f $ fmap g x == fmap (f . g) x
fmap f $ fmap g $ x == fmap (f . g) $ x
fmap f . fmap g $ x == fmap (f . g) $ x
fmap f . fmap g == fmap (f . g)

-- def fmap
fmap h (Cmps x) = Cmps $ fmap (fmap h) x

-- left part
fmap h2 (fmap h1 (Cmps x))
== fmap h2 (Cmps $ fmap (fmap h1) x)  -- def fmap
== Cmps $ fmap (fmap h2) (fmap (fmap h1) x)  -- def fmap
== Cmps $ fmap (fmap h2) $ fmap (fmap h1) x
== Cmps $ fmap (fmap h2) $ fmap (fmap h1) x
== Cmps $ fmap f $ fmap g x -- let f = fmap h2, let g = fmap h1
== Cmps (fmap f $ fmap g $ x)
== Cmps (fmap f . fmap g $ x)
== Cmps (fmap f . fmap g $ x)
== Cmps (fmap (f . g) x) -- fmap law
== Cmps (fmap (fmap h2 . fmap h1) x) -- reduct f and g
== Cmps $ fmap (fmap h2 . fmap h1) x
== Cmps $ fmap (fmap (h2 . h1)) x -- fmap law

-- right part
fmap (h2 . h1) (Cmps x)
== Cmps $ fmap (fmap (h2 . h1)) x

Cmps $ fmap (fmap (h2 . h1)) x == Cmps $ fmap (fmap (h2 . h1)) x
-}

{-


(1) fmap f (Cmps x) = fmap (fmap f) x
(2) fmap f (fmap g x) = fmap (f . g) x
(3) fmap f . fmap g = fmap (f . g) т.к.
  fmap f . fmap g = \x -> (fmap f $ fmap g x) = (2) = \x -> fmap (f . g) x = fmap (f . g)

fmap h2 (fmap h1 (Cmps x)) =                             -- (1)
fmap h2 (Cmps $ fmap (fmap h1) x) =               -- (1)
Cmps $ fmap (fmap h2) (fmap (fmap h1) x) =  -- (2), f = (fmap h2), g = (fmap h1)
Cmps $ fmap (fmap h2 . fmap h1) x =                -- (3)
Cmps $ fmap (fmap (h2 . h1)) x =                        -- (1)
fmap (h1 . h2) (Cmps x)

-}

--1.5.9
{-
Напишите универсальные функции
unCmps3 :: Functor f => (f |.| g |.| h) a -> f (g (h a))
unCmps4 :: (Functor f2, Functor f1) => (f2 |.| f1 |.| g |.| h) a -> f2 (f1 (g (h a)))
позволяющие избавляться от синтаксического шума для композиции нескольких функторов:
GHCi> pure 42 :: ([] |.| [] |.| []) Int
Cmps {getCmps = [Cmps {getCmps = [[42]]}]}
GHCi> unCmps3 (pure 42 :: ([] |.| [] |.| []) Int)
[[[42]]]
GHCi> unCmps3 (pure 42 :: ([] |.| Maybe |.| []) Int)
[Just [42]]
GHCi> unCmps4 (pure 42 :: ([] |.| [] |.| [] |.| []) Int)
[[[[42]]]]
-}

-- :set -XTypeOperators

--https://stepik.org/lesson/30426/step/9?discussion=1854804&unit=11043
{-
Вот код, чтобы можно было компилировать локально.

infixr 9 |.|
newtype (|.|) f g a = Cmps {getCmps :: f (g a)} deriving (Eq, Show)

instance (Functor f, Functor g) => Functor (f |.| g) where
  fmap h (Cmps x) = Cmps $ fmap (fmap h) x

instance (Applicative f, Applicative g) => Applicative (f |.| g) where
  pure = Cmps . pure . pure
  (<*>) = undefined
-}

{-
https://stepik.org/lesson/30426/step/9?discussion=373406&reply=1512264&unit=11043
*Main> pure 42 :: ([] |.| [] |.| []) Int
Cmps {getCmps = [Cmps {getCmps = [[42]]}]}
*Main> getCmps (pure 42 :: ([] |.| [] |.| []) Int)
[Cmps {getCmps = [[42]]}]
-}

instance (Functor f, Functor g) => Functor (f |.| g) where
  fmap h (Cmps x) = Cmps $ fmap (fmap h) x

instance (Applicative f, Applicative g) => Applicative (f |.| g) where
  pure = Cmps . pure . pure
  (<*>) = undefined

--newtype (|.|) f g a = Cmps {getCmps :: f (g a)} deriving (Eq, Show)

unCmps3 :: Functor f => (f |.| g |.| h) a -> f (g (h a))
--unCmps3 :: (|.|) f (g |.| h) a -> f (g (h a))
--unCmps3 f a= getCmps f g h a --  Expected type: f (g (h a))    Actual type: t1 -> g (h a)
--unCmps3 f = getCmps (fmap f (Cmps g h) ) a
--x :: f ((|.|) g h a)
--y :: f (g a)

-- unCmps3 (Cmps x) =  x where 
--   f (Cmps y) = x where
--    g (Cmps h) =y
-- cannot construct the infinite type: g ~ g |.| h
--   Expected type: f (g (h a))
--     Actual type: f ((|.|) g h a)

unCmps3 (Cmps   y) = fmap getCmps y

unCmps4 :: (Functor f2, Functor f1) => (f2 |.| f1 |.| g |.| h) a -> f2 (f1 (g (h a)))
unCmps4 (Cmps y) = fmap unCmps3 y


--https://stepik.org/lesson/30426/step/9?discussion=369407&thread=solutions&unit=11043
{-
unCmps3 :: Functor f => (f |.| g |.| h) a -> f (g (h a))
unCmps3 = fmap getCmps . getCmps

unCmps4 :: (Functor f2, Functor f1) => (f2 |.| f1 |.| g |.| h) a -> f2 (f1 (g (h a)))
unCmps4 = fmap unCmps3 . getCmps
-}

--https://stepik.org/lesson/30426/step/9?discussion=374153&thread=solutions&unit=11043
{-
unCmps3 :: Functor f => (f |.| g |.| h) a -> f (g (h a))
unCmps3 (Cmps x) = getCmps <$> x

unCmps4 :: (Functor f2, Functor f1) => (f2 |.| f1 |.| g |.| h) a -> f2 (f1 (g (h a)))
unCmps4 (Cmps x) = unCmps3 <$> x
-}

--https://stepik.org/lesson/30426/step/9?discussion=962364&thread=solutions&unit=11043
{-
unCmps3 :: Functor f => (f |.| g |.| h) a -> f (g (h a))
unCmps3 = fmap getCmps . getCmps

unCmps4 :: (Functor f2, Functor f1) => (f2 |.| f1 |.| g |.| h) a -> f2 (f1 (g (h a)))
unCmps4 = (fmap . fmap) getCmps . fmap getCmps . getCmps
-}

--https://stepik.org/lesson/30426/step/9?discussion=373000&thread=solutions&unit=11043
{-
unCmps3 :: Functor f => (f |.| g |.| h) a -> f (g (h a))
unCmps3 = fmap getCmps . getCmps

unCmps4 :: (Functor f2, Functor f1) => (f2 |.| f1 |.| g |.| h) a -> f2 (f1 (g (h a)))
unCmps4 = fmap (fmap getCmps) . fmap getCmps . getCmps
--https://stepik.org/lesson/30426/step/9?discussion=373000&reply=373006&thread=solutions&unit=11043
--По-моему самое понятное решение: снимаем внешнюю обертку (getCmps) и протаскиваем через получившийся контейнер снималку для внутренней обертки (fmap getCmps). 
-}