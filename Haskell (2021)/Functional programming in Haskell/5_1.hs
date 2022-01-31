import Control.Monad (ap, liftM)
--5.1.3
{-
Определите представителя класса Functor для следующего типа данных, представляющего точку в трёхмерном пространстве:
data Point3D a = Point3D a a a deriving Show
GHCi> fmap (+ 1) (Point3D 5 6 7)
Point3D 6 7 8
-}
data Point3D a = Point3D a a a deriving (Show)
instance Functor Point3D where
  fmap f (Point3D a b c) = Point3D (f a) (f b) (f c)

--5.1.4
{-
Определите представителя класса Functor для типа данных GeomPrimitive, который определён следующим образом:
data GeomPrimitive a = Point (Point3D a) | LineSegment (Point3D a) (Point3D a)
При определении, воспользуйтесь тем, что Point3D уже является представителем класса Functor.
GHCi> fmap (+ 1) $ Point (Point3D 0 0 0)
Point (Point3D 1 1 1)
GHCi> fmap (+ 1) $ LineSegment (Point3D 0 0 0) (Point3D 1 1 1)
LineSegment (Point3D 1 1 1) (Point3D 2 2 2)
-}

data GeomPrimitive a = Point (Point3D a) | LineSegment (Point3D a) (Point3D a) deriving Show
instance Functor GeomPrimitive where
  fmap f (Point (Point3D a b c)) = Point (Point3D (f a) (f b) (f c))
  fmap f (LineSegment (Point3D a b c) (Point3D a1 b1 c1)) = LineSegment (Point3D (f a) (f b) (f c)) (Point3D (f a1) (f b1) (f c1))

--https://stepik.org/lesson/8432/step/4?discussion=625604&thread=solutions&unit=2743
-- on :: (b -> b -> c) -> (a -> b) -> a -> a -> c
-- on f g x y = f (g x) (g y)
-- instance Functor GeomPrimitive where
--   fmap f (Point p) = Point $ fmap f p
--   fmap f (LineSegment p1 p2) = on (LineSegment) (fmap f) p1 p2

--5.1.6
{-
Определите представителя класса Functor для бинарного дерева, в каждом узле которого хранятся элементы типа Maybe:
data Tree a = Leaf (Maybe a) | Branch (Tree a) (Maybe a) (Tree a) deriving Show
GHCi> words <$> Leaf Nothing
Leaf Nothing
GHCi> words <$> Leaf (Just "a b")
Leaf (Just ["a","b"])
-}

data Tree a = Leaf (Maybe a) | Branch (Tree a) (Maybe a) (Tree a) deriving (Show)
instance Functor Tree where
  --fmap f (Leaf (Maybe a)) = Leaf (Maybe (f a))  
  fmap f (Leaf a) = Leaf (fmap f a)
  --fmap f (Branch (Tree a) x (Tree b)) = Branch (Tree (fmap f a)) (fmap f x) (Tree (fmap f b))
  fmap f (Branch (Leaf a) x (Leaf b)) = Branch (fmap f (Leaf a)) (fmap f x) (fmap f (Leaf b))
  fmap f (Branch (Branch a b c) x (Branch a1 b1 c1)) = Branch (fmap f (Branch a b c)) (fmap f x) (fmap f (Branch a1 b1 c1))
  fmap f (Branch (Branch a b c) x (Leaf b1)) = Branch (fmap f (Branch a b c)) (fmap f x) (fmap f (Leaf b1))
  fmap f (Branch (Leaf b) x (Branch a1 b1 c1)) = Branch (fmap f (Leaf b)) (fmap f x) (fmap f (Branch a1 b1 c1))

-- ++https://stepik.org/lesson/8432/step/6?discussion=339644&thread=solutions&unit=2743
-- instance Functor Tree where
--   fmap f (Leaf a) = Leaf (fmap f a)
--   fmap f (Branch a b c) = Branch (fmap f a) (fmap f b) (fmap f c)

--5.1.8
{-
 Определите представителя класса Functor для типов данных Entry и Map. Тип Map представляет словарь, ключами которого являются пары:
data Entry k1 k2 v = Entry (k1, k2) v  deriving Show
data Map k1 k2 v = Map [Entry k1 k2 v]  deriving Show
В результате должно обеспечиваться следующее поведение: fmap применяет функцию к значениям в словаре, не изменяя при этом ключи.
GHCi> fmap (map toUpper) $ Map []
Map []
GHCi> fmap (map toUpper) $ Map [Entry (0, 0) "origin", Entry (800, 0) "right corner"]
Map [Entry (0,0) "ORIGIN",Entry (800,0) "RIGHT CORNER"]
-}
data Entry k1 k2 v = Entry (k1, k2) v deriving (Show)
data Map k1 k2 v = Map [Entry k1 k2 v] deriving (Show)

instance Functor (Entry k1 k2) where
  fmap f (Entry (k1, k2) v) = Entry (k1, k2) (f v)
instance Functor (Map k1 k2) where
  fmap f (Map []) = Map []
  -- fmap f (Map [(Entry (k1, k2) v)]) = Map [Entry (k1, k2) (f v)]
  -- fmap f ((Entry (k1, k2) v) : (Map [(Entry (k3, k4) v2)])) = Map [Entry (k1, k2) (f v)]
  fmap f (Map [e]) = Map [fmap f e]
  fmap f (Map (e : ex)) = Map ((fmap f e) : (map (fmap f) ex)) --Map ((fmap f e) : (fmap f  (Map ex)))

-- (Map(fmap f  ex))
-- • Couldn't match expected type ‘[Entry k1 k2 b]’
--               with actual type ‘Map k10 k20 v0’
-- (fmap f  (Map ex))
-- • Couldn't match type ‘Map k1 k2’ with ‘[]’
--   Expected type: [Entry k1 k2 b]
--     Actual type: Map k1 k2 b

--https://stepik.org/lesson/8432/step/8?discussion=1169556&thread=solutions&unit=2743
-- instance Functor (Entry k1 k2) where
--   fmap f (Entry k v) = Entry k $ f v
-- instance Functor (Map k1 k2) where
--   fmap f (Map es) = Map [fmap f e | e <- es]

--  https://stepik.org/lesson/8432/step/8?discussion=573881&thread=solutions&unit=2743
-- import Data.Functor
-- instance Functor (Entry k1 k2) where
--   fmap f (Entry k v) = Entry k $ f v
-- instance Functor (Map k1 k2) where
--   fmap f (Map arr) = Map $ (f <$>) <$> arr

-- https://stepik.org/lesson/8432/step/8?discussion=346661&thread=solutions&unit=2743
-- instance Functor (Entry k1 k2) where
--   fmap f (Entry (k1, k2) v) = Entry (k1, k2) (f v)
-- instance Functor (Map k1 k2) where
--   fmap f (Map m) = Map (map (fmap f) m)

-- Законы для функторов:
-- fmap id = id
-- fmap (f.g) = fmap f . fmap g
--
-- fmap id xs = id xs
-- fmap (f.g) xs = (fmap f . fmap g) xs

--5.2.3
{-
Введём следующий тип:
data Log a = Log [String] a
Реализуйте вычисление с логированием, используя Log. Для начала определите функцию toLogger
toLogger :: (a -> b) -> String -> (a -> Log b)
которая превращает обычную функцию, в функцию с логированием:
GHCi> let add1Log = toLogger (+1) "added one"
GHCi> add1Log 3
Log ["added one"] 4
GHCi> let mult2Log = toLogger (* 2) "multiplied by 2"
GHCi> mult2Log 3
Log ["multiplied by 2"] 6
Далее, определите функцию execLoggers
execLoggers :: a -> (a -> Log b) -> (b -> Log c) -> Log c
Которая принимает некоторый элемент и две функции с логированием. execLoggers возвращает результат последовательного 
применения функций к элементу и список сообщений, которые были выданы при применении каждой из функций:
GHCi> execLoggers 3 add1Log mult2Log
Log ["added one","multiplied by 2"] 8
toLogger :: (a -> b) -> String -> (a -> Log b)
toLogger f msg = undefined
execLoggers :: a -> (a -> Log b) -> (b -> Log c) -> Log c
execLoggers x f g = undefined
-}
data Log a = Log [String] a deriving Show
toLogger :: (a -> b) -> String -> (a -> Log b)
--toLogger f msg a= Log [msg] (f  a)
toLogger f msg = Log [msg] . f
execLoggers :: a -> (a -> Log b) -> (b -> Log c) -> Log c
--execLoggers a (toLogger f msg1) (toLogger g [msg2]) = Log [msg1++msg2] (g.f)
execLoggers x f g = Log [msg1 , msg2] z where --(g (f x)) where
    Log [msg1] y =f x
    Log [msg2] z = g y
add1Log = toLogger (+ 1) "added one"
mult2Log = toLogger (* 2) "multiplied by 2"
    --y = f x
  --toLogger f1 msg1 =f 
  -- Log [msg1] x1
  -- • Couldn't match expected type ‘a -> Log b’
  --             with actual type ‘Log a0’

--Log [msg1 ++ msg2] .g.f :: a0 -> Log (Log c) instead of Log c
-- execLoggers a n (Log [msg2] c) = Log [msg1++msg2] (c.b) where
--   n = Log [msg1] b
--execLoggers a n m = Log [msg1 ++ msg2] c where
---execLoggers a ( Log [msg1] f  ) (Log [msg2] g) = Log [msg1 ++ msg2] (g (f a)) where
-- Couldn't match expected type ‘a -> Log b’
--               with actual type ‘Log (a -> t0)’
-- • In the pattern: Log [msg1] f
-- execLoggers a n m =( Log [msg1 ++ msg2] (g.f) )  where
--    n :: a -> Log b
--    n  = toLogger f msg1
--   n = toLogger f msg1-- a --Log a
--   m :: b -> Log c
--   m = toLogger g msg2
  -- m = toLogger g msg2 b --Log b
-- execLoggers x f g = helper f  where --g.f $ x
--  helper f g = 
  --   b=f  a
  -- c = g b

--5.2.5
{-
Функции с логированием из предыдущего задания возвращают в качестве результата значение с некоторой 
дополнительной информацией в виде списка сообщений. Этот список является контекстом. Реализуйте функцию returnLog
returnLog :: a -> Log a
которая является аналогом функции return для контекста Log. 
Данная функция должна возвращать переданное ей значение с пустым контекстом.
-}
returnLog :: a -> Log a
returnLog = \a->Log [] a
--returnLog a = Log [] a
--returnLog =Log [] 

--5.2.7
{-
Реализуйте фукцию bindLog
bindLog :: Log a -> (a -> Log b) -> Log b
которая работает подобно оператору >>= для контекста Log.
GHCi> Log ["nothing done yet"] 0 `bindLog` add1Log
Log ["nothing done yet","added one"] 1
GHCi> Log ["nothing done yet"] 3 `bindLog` add1Log `bindLog` mult2Log
Log ["nothing done yet","added one","multiplied by 2"] 8
-}
bindLog :: Log a -> (a -> Log b) -> Log b
bindLog (Log msg1 x) fu = Log (msg1 ++ msg2) y
  where
    Log msg2 y = fu x


--5.2.8
{-
Реализованные ранее returnLog и bindLog позволяют объявить тип Log представителем класса Monad:
instance Monad Log where
    return = returnLog
    (>>=) = bindLog
Используя return и >>=, определите функцию execLoggersList
execLoggersList :: a -> [a -> Log a] -> Log a
которая принимает некоторый элемент, список функций с логированием и возвращает результат последовательного применения 
всех функций в списке к переданному элементу вместе со списком сообщений, которые возвращались данными функциями:
GHCi> execLoggersList 3 [add1Log, mult2Log, \x -> Log ["multiplied by 100"] (x * 100)]
Log ["added one","multiplied by 2","multiplied by 100"] 800
-}

instance Functor Log where
  fmap = liftM
instance Applicative Log where
  pure = return
  (<*>) = ap
{-
instance Monad Log where
  return = returnLog
  (>>=) = bindLog
-----
No instance for (Applicative Log)
      arising from the superclasses of an instance declaration
    In the instance declaration for ‘Monad Log’
Failed, modules loaded: none.
Prelude> :i Monad
class Applicative m => Monad (m :: * -> *) where
  (>>=) :: m a -> (a -> m b) -> m b
  (>>) :: m a -> m b -> m b
  return :: a -> m a
  fail :: String -> m a
--
https://stepik.org/lesson/8437/step/8?discussion=351716&reply=351729&unit=1572
 Класс типов Monad теперь расширяет классы типов Functor и Applicative, поэтому нужно добавить представителей для них. Например, так
instance Functor Log where
  fmap = liftM
instance Applicative Log where
  pure = return
  (<*>) = ap
(Это подойдет для любого однапараметрического типа, не только Log). Нужно еще не забыть импортировать Control.Monad (ap, liftM).
-}

instance Monad Log where
  return = returnLog
  (>>=) = bindLog

execLoggersList :: a -> [a -> Log a] -> Log a
execLoggersList x [] = Log [] x
execLoggersList x (s : sx) = Log (msg0 ++ msg) z where
 Log msg0 y = s x
 Log msg z = execLoggersList y sx

 --5.3.3
 {-
 Если некоторый тип является представителем класса Monad, то его можно сделать представителем 
 класса Functor, используя функцию return и оператор >>=. Причём, это можно сделать даже не зная, 
 как данный тип устроен.
Пусть вам дан тип 
data SomeType a = ...
и он является представителем класса Monad. Сделайте его представителем класса Functor.
instance Functor SomeType where
  fmap f x = undefined
 -}


instance Applicative SomeType where
  pure = return
  (<*>) = ap


data SomeType a = Mop a
instance Functor SomeType where  
  --x::SomeType a  -- = f a   -- = m a -- = return a
  --y::SomeType b  -- = f b   -- = m b -- = return b
  --  
  --  (>>=) :: m a -> (a -> m b) -> m b
  --  return :: a -> m a
  --fmap :: (a -> b) -> SomeType a -> SomeType b
  --fmap f x =
  fmap f x = x >>= (\a -> return (f a)) where --- == return . f
   --where--y
    --z a = return (f a)
    --y = x >>= z
    
    
    --(\s -> f s) ==f  --!!!


    --

    -- d = return y -- :: SomeType (SomeType b) -- loop
    -- y :: SomeType b--a -> SomeType b -- loop
    -- return b = y-- :: f b -- loop
    -- z b= return f -- loop
    -- y = x >>= z -- :: f b -- loop
    -- loop


  --fmap f x = 
  --y@(fmap f x@(return a)) = x >>=(\a ->y)
  --fmap f x@(return a) = y@(x >>=(\a ->y)) 
  -- fmap f x = y @ (x >>=(\a ->y))  where
  --   f a = x
  --
  -- fmap f x = y where --loop
  --   f b = y --loop
  --   y = (x >>= f) --loop
  --
  -- fmap f x = y :: SomeType b where --loop
  --   f :: (a -> SomeType b) --loop
  --   f a = y --loop
  --   y = (x >>= f) :: SomeType b --loop
  --

    -- ::SomeType  a ->x => SomeType b -> (SomeType a -> (a -> SomeType b) -> SomeType b) 
    --return a = f :: Monad f => f a
    --return a0 = x :: Monad f => SomeType a
    --return a0 = x :: Monad f =>  SomeType a->f x
    -- return (\a0->x >>= f) --return  (x >>= f)
  --  fmap f x = y where
    --y = x >>= const y -- loop
    --fmap f x = x >>= const (fmap f x) -- loop
    --return a = x    
    --return b = y
    --d = x >>= f
     
    

--https://stepik.org/lesson/8438/step/3?discussion=347092&reply=347155&unit=1573
-- x :: SomeType a
-- f :: a -> b
-- (>>=) :: SomeType a -> (a -> SomeType c) -> SomeType c
-- x >>= f :: ???
--  fmap f (f a) = f a >>= f (f a)
--  return a = f a
-- x >>= f = y t  where -- :: SomeType a -> SomeType (a -> b)->

    --b = f x
  --fmap f x = return x >>= f x -- :: SomeType
  --
instance Monad SomeType where
  --x >>= f = y
  --return a0 = x:: SomeType a
  
  --x >>= f= x
  --x >>= f = return a
  --f a >>= (\a -> f b) = f b :: SomeType f
  --return a = f a  :: SomeType f
  --return b = return a >>= (\a -> return b) -- :: SomeType f
  -- b = f a >>= (\a -> f b) :: SomeType f

  --Mop b = Mop a >>= (\a -> Mop b) 
  --return a = Mop a
  --fmap f x = f (return a) = 
  --fmap :: (a -> m a) -> SomeType a -> SomeType (m a)
  --fmap :: return -> x -> SomeType (m a)
  --fmap :: return -> x -> SomeType (m a)
  
  --fmap f x = x >>= const (f x) :: SomeType b2 where
    --f x = x >>= const (f x)
    --f x = x >>= \x -> y
   --return y = return x >>= \x -> return y -- :: SomeType y
--Found:  return x >>= (\ x -> return y)
--Why not:  (\ x -> return y) x    
-- y :: m b2
-- y = error "not implemented"
--  fmap f x = y --where
--  --return x >>= \ x -> return y = return y
--  return y = return x >>= (\x -> return y)
--  return y = return y
    
 
 --https://stepik.org/lesson/8438/step/3?discussion=1194498&unit=1573
-- Вы должны реализовать тело функции fmap с типом
-- (a -> b) -> f a -> f b
-- используя её параметры
-- x :: f a
-- f :: (a -> b)
-- и 2 функции из класса типов Monad:
-- (>>=) :: m a -> (a -> m b) -> m b
-- return :: a -> m a


--https://stepik.org/lesson/8438/step/2?discussion=340661&reply=340674&unit=1573
{-
Класс типов Monad теперь расширяет классы типов Functor и Applicative, поэтому нужно добавить представителей для них:
instance Functor Identity where
  fmap  f (Identity x) = Identity (f x)
instance Applicative Identity where
  pure x = Identity x
  Identity f <*> Identity v = Identity (f v) 

But better:
instance Applicative SomeType where
  pure = return
  (<*>) = ap
-}

--5.3.6
{-
Вспомним тип Log
data Log a = Log [String] a
который мы сделали монадой в предыдущем модуле. Функция return для Log оборачивает переданное значение в лог с пустым списком сообщений. 
Оператор >>= возвращает лог с модифицированным значением и новым списком сообщений, который состоит из прежнего списка и добавленного 
в конец списка сообщений, полученных при модификации значения.
Пусть теперь функция return будет оборачивать переданное значение в список, содержащий одно стандартное сообщение "Log start".
Выберите верные утверждения относительно выполнения законов для монады с новым поведением функции return.
+Не выполняется первый закон
+Не выполняется второй закон
Не выполняется третий закон
Все законы выполняются
-}
{-
1st law of monads
return a >>= k == k a
2nd law of monads
m>>=return == m
3rd law of monads
m>>=k>>=k` == m>>= (\x -> k x >>=k`)
-}

--(<=<):: Monad m => (b-> m c) -> (a -> m b) -> a-> m c
--(<=<) f g x =  g x >>= f 

--https://habr.com/ru/post/128070/
-- mapply :: m b -> (b -> m c) -> m c   
-- mcompose :: (a -> m b) -> (b -> m c) -> (a -> m c) ----==mcompose :: (a -> m b) -> (b -> m c) -> a -> m c
-- mcompose f g x = (f x) `mapply` g -- или: mapply (f x) g
--
-- (>>=) :: m a -> (a -> m b) -> m b
-- (=<<) :: (a -> m b) -> m a -> m b
-- f =<< x = x >>= f
--
-- flip :: (a -> b -> c) -> (b -> a -> c)
-- flip f = \x y -> f y x
-- (=<<) = flip (>>=)
--
-- f :: a -> m b
-- g :: b -> m c
-- f >=> g = \x -> (f x >>= g)
--
-- (>=>) :: (a -> m b) -> (b -> m c) -> (a -> m c)
-- (<=<) :: (b -> m c) -> (a -> m b) -> (a -> m c)
-- (<=<) = flip (>=>)
--
-- functionToMonadicFunction :: (a -> b) -> (a -> m b)
-- functionToMonadicFunction f = \x -> return (f x) -- or
-- functionToMonadicFunction f = return . f -- or
-- functionToMonadicFunction = (return .)
--
    -- getLine :: IO String
    -- putStrLn :: String -> IO ()
    -- readAndPrintLine = getLine >>= putStrLn

--https://habr.com/ru/post/128538/
{-
class Monad m where
  (>>=) :: m a -> (a -> m b) -> m b
  return :: a -> m a
  (>>) :: m a -> m b -> m b
  fail :: String -> m a

(>>) :: m a -> m b -> m b
mv1 >> mv2 = mv1 >>= (\_ -> mv2)

оператор (>=>) — это монадический оператор композиции функций
f >=> g = \x -> (f x >>= g)

1 . return >=> f == f
2 . f >=> return == f
3 . (f >=> g) >=> h == f >=> (g >=> h)

1.
    return >=> f == f
    \x -> (return x >>= f) == \x -> f x
    return x >>= f == f x -- Q.E.D. ("Что и требовалось доказать")
2.
    f >=> return == f
    \x -> (f x >>= return) == \x -> f x
    f x >>= return == f x
    let mv == f x
    mv >>= return == mv — Q.E.D.
3.
    (f >=> g) >=> h                 ==    f >=> (g >=> h)
    \x -> ((f >=> g) x >>= h)       ==    \x -> (f x >>= (g >=> h))
    (f >=> g) x >>= h               ==    f x >>= (g >=> h)
    (\y -> (f y >>= g)) x >>= h     ==    f x >>= (\y -> (g y >>= h))
    -- Вычисляем (\y -> (f y >>= g)) x получаем: (f x >>= g)
    (f x >>= g) >>= h               ==    f x >>= (\y -> (g y >>= h))
    -- Пусть mv = f x, тогда:
    (mv >>= g) >>= h                ==    mv >>= (\y -> (g y >>= h))
    -- Заменяем g на f, h на g:
    (mv >>= f) >>= g                ==    mv >>= (\y -> (f y >>= g)) 
    -- Заменяем y на x в правом выражении и получаем:
    (mv >>= f) >>= g                ==    mv >>= (\x -> (f x >>= g))   -- Q.E.D.
-}


data Log1 a = Log1 [String] a deriving (Show)
toLogger1 :: (a -> b) -> String -> (a -> Log1 b)
toLogger1 f msg = Log1 [msg] . f
execLoggers1 :: a -> (a -> Log1 b) -> (b -> Log1 c) -> Log1 c
execLoggers1 x f g = Log1 [msg1, msg2] z
  where
    Log1 [msg1] y = f x
    Log1 [msg2] z = g y
add1Log1 = toLogger1 (+ 1) "added one"
mult2Log1 = toLogger1 (* 2) "multiplied by 2"

instance Functor Log1 where
  fmap = liftM

instance Applicative Log1 where
  pure = return
  (<*>) = ap

returnLog1 :: a -> Log1 a
returnLog1 = \a -> Log1 ["Log start"] a

bindLog1 :: Log1 a -> (a -> Log1 b) -> Log1 b
bindLog1 (Log1 msg1 x) fu = Log1 (msg1 ++ msg2) y
  where
    Log1 msg2 y = fu x

instance Monad Log1 where
  return = returnLog1
  (>>=) = bindLog1

execLoggersList1 :: a -> [a -> Log1 a] -> Log1 a
execLoggersList1 x [] = Log1 [] x
execLoggersList1 x (s : sx) = Log1 (msg0 ++ msg) z
  where
    Log1 msg0 y = s x
    Log1 msg z = execLoggersList1 y sx

{-
1st law of monads
return a >>= k == k a
2nd law of monads
m>>=return == m
3rd law of monads
m>>=k>>=k` == m>>= (\x -> k x >>=k`)
-}
cc=22::Int
p1 = returnLog1 cc `bindLog1` add1Log1 --Log1 ["Log start","added one"] 23
p2 = add1Log1 cc --Log1 ["added one"] 23
--law1 = p1 == p2 --False
p3 = Log1 ["8908"] cc >>= return --Log1 ["8908","Log start"] 22
p30 = add1Log1 cc >>= return -- Log1 ["added one","Log start"] 23
p4 = Log1 ["8908"] cc --Log1 ["8908"] 22
p40 = add1Log1 cc -- Log1 ["added one"] 23
--law2 = p3 == p4 --False
p5 = Log1 ["8908"] cc >>= add1Log1 >>= mult2Log1 --Log1 ["8908","added one","multiplied by 2"] 46
p6 = Log1 ["8908"] cc >>= (\x -> add1Log1 x >>= mult2Log1) --Log1 ["8908","added one","multiplied by 2"] 46
--law3 = p5 == p6 --True
 ---   

-- data Log1 a = Log1 [String] a deriving Show 
-- instance Functor Log1 where
--   fmap f (Log1 s x) = Log1 s (f x) -- ??
-- instance Applicative Log1 where
--   pure = return
--   (<*>) = ap
-- instance Monad Log1 where
--   return  x = Log1 ["Log start"] x
--   Log1 s1 a >>= fuu= Log1 (s1++s2) y where
--     Log1 s2  y =fuu a 
-- returnm  x = Log1 ["Log start"] x
-- Log1 s1 a `mu` fuu= Log1 (s1++s2) y where
--     Log1 s2  y =fuu a 
-- r1 :: Num a => Log1 a
-- r1 = returnm 7

-- (>>=) :: m a -> (a -> m b) -> m b
-- return :: a -> m a
--1st law of monads: return a >>= k == k a
--refactored: (k == k a) a
--a=3
--r1 = returnm 
-------r2 :: Log1 a -> (a -> Log1 b) -> (Log1 b)
-- --r2 = returnm a `mu` k
-------r2 = (returnm 7) `mu` (\x->(Log1 ["df"] (x+1)))
-- r3 = k a
-- r = (r2 == k a)
-- 2nd law of monads
-- m>>=return == m
-- 3rd law of monads
-- m>>=k>>=k` == m>>= (\x -> k x >>=k`)



--  +++++++ https://bartoszmilewski.com/2011/03/14/monads-for-the-curious-programmer-part-2/
-- Nothing >>= cont = Nothing
-- (Just x) >>= cont = cont x
-- (>>=) :: Maybe a -> (a -> Maybe b) -> Maybe b

-- compose1 n =
--   f n >>= \n1 ->
--     g n1 >>= \n2 ->
--       h n2 >>= \n3 ->
--         return n3

---
-- Using both bind and return we can lift any function f:
-- f :: a -> b
-- to a function g:
-- g :: M a -> M b
-- Here’s the magic formula that defines g in terms of f, bind, and return (the dot denotes regular function composition):
-- g ma = bind ma (return . f)
-- Or, using infix notation:
-- g ma = ma >>= (return . f)
---
-- fmap :: (a -> b) -> (M a -> M b)
-- bind :: M a -> (a -> M b) -> M b
-- (a -> M b) -> (M a -> M b)
-- return :: a -> M a
---
-- join :: M (M a) -> M a
-- join mmx = mmx >>= id
-- bind :: M a' -> (a' -> M b') -> M b'
-- M (M a) -> (M a -> M a) -> M a
---
---
-- bind :: [a] -> (a -> [b]) -> [b]
-- xs >>= cont = concat (map cont xs)
---
--general formula for converting the functor definition of a monad to a Kleisli triple is:
-- 1. Take the object-mapping part of the functor (the type constructor)
-- 2. Define bind as
-- bind x f = join ((fmap f) x))
--  where fmap is the part of the functor that maps morphisms.
-- 3. Define return as unit
---
---
-- toss2Dice = do 
    -- n <- tossDie 
    -- m <- tossDie 
    -- return (n + m)
---
-- toss2Dice = [n + m | n <- tossDie, m <- tossDie]
---
--https://bartoszmilewski.com/2011/03/17/monads-for-the-curious-programmer-part-3/
----The Monadic Calculator
-- newtype Calc = Calc [Int]
-- popCalc =  \(Calc lst) -> (Calc (tail lst), head lst)
-- pushCalc n =  \(Calc lst) -> (Calc (n : lst), ())
-- addCalc =  \(Calc lst) ->     let (a : b : rest) = lst      in (Calc ((a + b) : rest), ())
-- add x y =
--   let pux = pushCalc x -- promise to push x
--       puy = pushCalc y -- promise to push y
--       axy = addCalc -- promise to add top numbers
--       pp = popCalc -- promise to pop the result
--       calc = Calc [] -- we need a calculator
--       (calc1, _) = pux calc -- actually push x
--       (calc2, _) = puy calc1 -- actually push y
--       (calc3, _) = axy calc2 -- actually add top numbers
--       (_, z) = pp calc3 -- actually pop the result
--    in z -- return the result
-----
-- bind :: (Calc -> (Calc, a)) ->        -- action
--         (a -> (Calc -> (Calc, b)) ->  -- continuation
--         (Calc -> (Calc, b))           -- new action
----or 
-- type Action a = Calc -> (Calc, a)
-- bind :: (Action a) -> (a -> (Action b)) -> (Action b)
-- bind act cont =    \calc -> ... produce (Calc, b) tuple ...
-- let (calc', v) = act calc
-- act' = cont v
-- bind act cont =  \calc ->
--     let (calc', v) = act calc
--         act' = cont v
--      in act' calc'
-- return :: a -> Action a
-- return v = \calc -> (calc, v)
---
-- add x y = do
--   pushCalc x
--   pushCalc y
--   addCalc
--   r <- popCalc
--   return r
---
-- add x y =
--   bind (pushCalc x)
--     ( \() -> bind (pushCalc y)
--           ( \() -> bind addCalc
--                 ( \() -> bind  popCalc
--                       (\z -> return z)
--                 )
--           )
--     )
---- 

-- 5.3.7
{-
Продолжим обсуждать монаду для Log. Пусть теперь у нас будет новая версия оператора >>=, которая будет добавлять сообщения не в конец 
результирующего списка, а в начало (при этом функция return предполагается возвращенной к исходной реализации).
Выберите верные утверждения относительно выполнения законов для монады с новым поведением оператора >>=.
Не выполняется первый закон
Не выполняется второй закон
Не выполняется третий закон
+Все законы выполняются 
-}

{-
1st law of monads
return a >>= k == k a
2nd law of monads
m>>=return == m
3rd law of monads
m>>=k>>=k` == m>>= (\x -> k x >>=k`)
-}

--https://stepik.org/lesson/8438/step/7?discussion=351232&reply=351896&unit=1573
{-
return 0 >>= \x -> add1Log x >>= \y -> mult2Log y
Log ["multiplied by 2", "added one"] 2
return 0 >>= (\x -> add1Log x) >>= (\y -> mult2Log y)
Log ["multiplied by 2", "added one"] 2
return 0 >>= (\x -> add1Log x) >>= (\y -> add1Log y)
Log ["added one", "added one"] 2
return 0 >>= \x -> add1Log x >>= \y -> add1Log y
Log ["added one", "added one"] 2
-}

--
--copypasting own previous functions
data Log2 a = Log2 [String] a deriving (Show)

toLogger2 :: (a -> b) -> String -> (a -> Log2 b)
toLogger2 f msg = Log2 [msg] . f

execLoggers2 :: a -> (a -> Log2 b) -> (b -> Log2 c) -> Log2 c
execLoggers2 x f g = Log2 [msg1, msg2] z 
  where
    Log2 [msg1] y = f x
    Log2 [msg2] z = g y

add1Log2 = toLogger2 (+ 1) "added one"
mult2Log2 = toLogger2 (* 2) "multiplied by 2"

instance Functor Log2 where
  fmap = liftM
instance Applicative Log2 where
  pure = return
  (<*>) = ap

returnLog2 :: a -> Log2 a
returnLog2 = \a -> Log2 [] a

bindLog2 :: Log2 a -> (a -> Log2 b) -> Log2 b
bindLog2 (Log2 msg1 x) fu = Log2 (msg2 ++ msg1) y
  where
    Log2 msg2 y = fu x

instance Monad Log2 where
  return = returnLog2
  (>>=) = bindLog2

execLoggersList2 :: a -> [a -> Log2 a] -> Log2 a
execLoggersList2 x [] = Log2 [] x
execLoggersList2 x (s : sx) = Log2 (msg0 ++ msg) z
  where
    Log2 msg0 y = s x
    Log2 msg z = execLoggersList2 y sx

p21 = returnLog2 cc `bindLog2` add1Log2 --Log2 ["added one"] 23

p22 = add1Log2 cc --Log2 ["added one"] 23
--law1 = p21 == p22 --True

p23 = Log2 ["8908"] cc >>= return --Log2 ["8908"] 22

p230 = add1Log2 cc >>= return --  Log2 ["added one"] 23

p24 = Log2 ["8908"] cc --Log2 ["8908"] 22

p240 = add1Log2 cc -- Log2 ["added one"] 23
--law2 = p23 == p24 --True

p25 = Log2 ["8908"] cc >>= add1Log2 >>= mult2Log2 --["multiplied by 2","added one","8908"] 46

p26 = Log2 ["8908"] cc >>= (\x -> add1Log2 x >>= mult2Log2) -- ["multiplied by 2","added one","8908"] 46
--law3 = p25 == p26 --True

--5.3.8
{-
И снова монада Log. Пусть теперь оператор >>= будет добавлять сообщения как в начало списка, так и в конец. 
Выберите верные утверждения относительно выполнения законов для монады с новым поведением оператора >>=.
-}

data Log3 a = Log3 [String] a deriving (Show)

toLogger3 :: (a -> b) -> String -> (a -> Log3 b)
toLogger3 f msg = Log3 [msg] . f

execLoggers3 :: a -> (a -> Log3 b) -> (b -> Log3 c) -> Log3 c
execLoggers3 x f g = Log3 [msg1, msg2] z
  where
    Log3 [msg1] y = f x
    Log3 [msg2] z = g y

add1Log3 = toLogger3 (+ 1) "added one"

mult2Log3 = toLogger3 (* 2) "multiplied by 2"

instance Functor Log3 where
  fmap = liftM

instance Applicative Log3 where
  pure = return
  (<*>) = ap

returnLog3 :: a -> Log3 a
returnLog3 = \a -> Log3 [] a

bindLog3 :: Log3 a -> (a -> Log3 b) -> Log3 b
bindLog3 (Log3 msg1 x) fu = Log3 (msg2 ++ msg1 ++ msg2) y
  where
    Log3 msg2 y = fu x

instance Monad Log3 where
  return = returnLog3
  (>>=) = bindLog3

execLoggersList3 :: a -> [a -> Log3 a] -> Log3 a
execLoggersList3 x [] = Log3 [] x
execLoggersList3 x (s : sx) = Log3 (msg0 ++ msg) z
  where
    Log3 msg0 y = s x
    Log3 msg z = execLoggersList3 y sx

p31 = returnLog3 cc `bindLog3` add1Log3 -- Log3 ["added one","added one"] 23

p32 = add1Log3 cc --Log3 ["added one"] 23
--law1 = p21 == p22 --False

p33 = Log3 ["8908"] cc >>= return --Log3 ["8908"] 22

p330 = add1Log3 cc >>= return --  Log3 ["added one"] 23

p34 = Log3 ["8908"] cc --Log3 ["8908"] 22

p340 = add1Log3 cc -- Log3 ["added one"] 23
--law2 = p23 == p24 --True

p35 = Log3 ["8908"] cc >>= add1Log3 >>= mult2Log3 --["multiplied by 2","added one","8908","added one","multiplied by 2"] 46

p36 = Log3 ["8908"] cc >>= (\x -> add1Log3 x >>= mult2Log3) -- ["multiplied by 2","added one","multiplied by 2","8908","multiplied by 2","added one","multiplied by 2"] 46
--law3 = p25 == p26 --False
