--4.1.5
{-
Тип данных Color определен следующим образом
data Color = Red | Green | Blue
Определите экземпляр класса Show для типа Color, сопоставляющий каждому из трех цветов его текстовое представление.
GHCi> show Red
"Red"
-}
data Color = Red | Green | Blue
instance Show Color where
  --show:: Color ->[Char]
  show  Red ="Red"
  show  Green = "Green"
  show  Blue = "Blue"

-- https://stepik.org/lesson/4916/step/5?discussion=1804126&thread=solutions&unit=1082
-- instance Show Color where
--   show x = case x of
--     Red -> "Red"
--     Green -> "Green"
--     Blue -> "Blue"

--4.1.7
{-
Определите частичную (определенную на значениях от '0' до '9') функцию charToInt.
GHCi> charToInt '0'
0GHCi> charToInt '9'
9
-}
charToInt :: Char -> Int
charToInt '0' = 0
charToInt '1' = 1
charToInt '2' = 2
charToInt '3' = 3
charToInt '4' = 4
charToInt '5' = 5
charToInt '6' = 6
charToInt '7' = 7
charToInt '8' = 8
charToInt '9' = 9

--4.1.8
{-
Определите (частичную) функцию stringToColor, которая по строковому представлению цвета как в прошлой задаче возвращает исходный цвет.
GHCi> stringToColor "Red"
Red
-}

--data Color = Red | Green | Blue
stringToColor :: String -> Color
stringToColor "Red" = Red
stringToColor "Green" = Green
stringToColor "Blue" = Blue

--4.1.10
{-
Пусть определены следующие функции:
emptyOrSingleton :: Bool -> a -> [a]
emptyOrSingleton False _ = []
emptyOrSingleton True x = [x]
isEqual :: (Eq a, Eq b) => (a, b) -> (a, b) -> Bool
isEqual (a, b) (a', b') = a == a' && b == b'
Выберите варианты вызовов этих функций, при которых сопоставление с образцом будет осуществлено успешно.
-}
-- emptyOrSingleton undefined 5
-- isEqual undefined (undefined, undefined)
-- isEqual undefined undefined
-- + isEqual (undefined, undefined) (undefined, undefined)
-- isEqual (undefined, undefined) undefined
-- + emptyOrSingleton True undefined
-- + emptyOrSingleton False undefined 

-- 4.1.11
{-Тип LogLevel описывает различные уровни логирования.
data LogLevel = Error | Warning | Info
Определите функцию cmp, сравнивающую элементы типа LogLevel так, чтобы было верно, что Error > Warning > Info.
GHCi> cmp Error Warning
GT
GHCi> cmp Info Warning
LT
GHCi> cmp Warning Warning
EQ
-}
data LogLevel = Error | Warning | Info
cmp :: LogLevel -> LogLevel -> Ordering
cmp Error Warning = GT
cmp Warning Info = GT
cmp Error Info = GT
cmp Info Warning = LT
cmp Info Error = LT
cmp Warning Error = LT
cmp Info Info = EQ
cmp Error Error = EQ
cmp Warning Warning = EQ

-- 4.1.13
{-
Пусть объявлен следующий тип данных:
data Result = Fail | Success
И допустим определен некоторый тип данных SomeData и некоторая функция
doSomeWork :: SomeData -> (Result,Int)
возвращающая результат своей работы и либо код ошибки в случае неудачи, либо 0 в случае успеха.
Определите функцию processData, которая вызывает doSomeWork и возвращает строку "Success" в случае 
ее успешного завершения, либо строку "Fail: N" в случае неудачи, где N — код ошибки.
-}

-- processData :: SomeData -> String
-- processData z = case doSomeWork z of
--     (_,0) ->"Success"
--     (_,n) ->"Fail: "++ show n
--Works!

--4.2.3
{-
Реализуйте функцию distance, возвращающую расстояние между двумя точками.
-}
data Point = Point Double Double
origin :: Point
origin = Point 0.0 0.0

distanceToOrigin :: Point -> Double
distanceToOrigin (Point x y) = sqrt (x ^ 2 + y ^ 2)

distance :: Point -> Point -> Double
distance (Point x1 y1) (Point x2 y2) = sqrt ((x2-x1) ^ 2 + (y2-y1) ^ 2)

--4.2.5
{-
Определим тип фигур Shape:
data Shape = Circle Double | Rectangle Double Double
У него два конструктора: Circle r — окружность радиуса r, и Rectangle a b — прямоугольник с размерами сторон a и b. 
Реализуйте функцию area, возвращающую площадь фигуры. Константа pi уже определена в стандартной библиотеке.
-}

-- data Shape = Circle Double | Rectangle Double Double
-- area :: Shape -> Double
-- area  Circle r = pi*r*r
-- area Rectangle a b = a * b
--main.hs:4:1:    Equations for ‘area’ have different numbers of arguments

data Shape = Circle Double | Rectangle Double Double deriving Show
area :: Shape -> Double
area (Circle r) = pi * r * r
area (Rectangle a b) = a * b

--4.2.6
{-
В одном из прошлых заданий мы встречали тип Result и функцию doSomeWork:
data Result = Fail | Success
doSomeWork :: SomeData -> (Result,Int)
Функция doSomeWork возвращала результат своей работы и либо код ошибки в случае неудачи, либо 0 в случае успеха. 
Такое определение функции не является наилучшим, так как в случае успеха мы вынуждены возвращать некоторое значение, 
которое не несет никакой смысловой нагрузки.
Используя функцию doSomeWork, определите функцию doSomeWork' так, чтобы она возвращала код ошибки только в случае неудачи. 
Для этого необходимо определить тип Result'. Кроме того, определите instance Show для Result' так, 
чтобы show возвращал "Success" в случае успеха и "Fail: N" в случае неудачи, где N — код ошибки.
-}


-- data Result' = Fail1 Int | Success1
-- instance Show Result' where
--   show :: Result' -> [Char]
--   show Fail1 n = "Fail: " ++ show n
--   show Success1 = "Success"
-- doSomeWork' :: SomeData -> [Char]
-- doSomeWork' z = case doSomeWork z of
--   (_, 0) -> (show Success1)
--   (_, n) -> (show Fail1)
--doesnt work...

-- Compilation error
-- main.hs:10:11:
--     Illegal type signature in instance declaration:
--       show :: Result' -> [Char]
--     (Use InstanceSigs to allow this)
--     In the instance declaration for ‘Show Result'’
-- main.hs:11:3:
--     Equations for ‘show’ have different numbers of arguments
--       main.hs:11:3-35
--       main.hs:12:3-27
--     In the instance declaration for ‘Show Result'’
-- main.hs:17:14:
--     No instance for (Show (Int -> Result'))
--       arising from a use of ‘show’
--     In the expression: (show Fail1)
--     In a case alternative: (_, n) -> (show Fail1)
--     In the expression:
--       case doSomeWork z of {
--         (_, 0) -> (show Success1)
--         (_, n) -> (show Fail1) }


-- data Result = Fail | Success
-- doSomeWork :: SomeData -> (Result, Int)
-- processData z = case doSomeWork z of
--     (_,0) ->"Success"
--     (_,n) ->"Fail: "++ show n

-- data Result' = Fail1 Int | Success1
-- instance Show Result' where
--   --show :: Result' -> [Char] -- Illegal type signature in instance declaration
--   show (Fail1 n) = "Fail: " ++ show n
--   show Success1 = "Success"
-- doSomeWork' :: SomeData -> Result'
-- doSomeWork' z = case doSomeWork z of
--   (_, 0) -> Success1
--   (_, n) -> (Fail1 n)
-- works!

--4.2.8
{-
Реализуйте функцию isSquare, проверяющую является ли фигура квадратом.
-}

--data Shape = Circle Double | Rectangle Double Double

square :: Double -> Shape
square a = Rectangle a a

isSquare :: Shape -> Bool
-- isSquare z = case z of
--   (Rectangle a a) -> True
isSquare (Rectangle a b) |a==b =True
                         |otherwise = False
isSquare _ = False
-- works!    

--https://stepik.org/lesson/4985/step/8?discussion=342908&thread=solutions&unit=1083
-- isSquare (Rectangle a b) = a == b
-- isSquare _ = False

--4.2.9
{-
Целое число можно представить как список битов со знаком.
Реализуйте функции сложения и умножения для таких целых чисел, считая, что младшие биты идут в начале списка, 
а старшие — в конце. Можно считать, что на вход не будут подаваться числа с ведущими нулями. 
-}

{-
data Bit = Zero | One deriving Show
data Sign = Minus | Plus deriving Show
data Z = Z Sign [Bit] deriving Show

bitgt :: Bit -> Bit -> Ordering
bitgt Zero Zero = EQ
bitgt One One = EQ
bitgt One Zero = GT
bitgt Zero One = LT

bitlistgt :: [Bit] -> [Bit] -> Ordering
bitlistgt [] [] = EQ
bitlistgt [x1] [x2] = bitgt x1 x2
bitlistgt a@(x1 : xs1) b@(x2 : xs2) = case bitgt (last a) (last b) of
    EQ -> bitlistgt (init a) (init b)
    GT ->GT
    LT ->LT

gtb :: Z -> Z -> Ordering
gtb (Z s1 b1) (Z s2 b2) = boo where
    boo
     |length b1> length b2 = GT
     |length b1< length b2 = LT
     |length b1 == length b2 = bitlistgt b1 b2

bitlistadd :: [Bit] -> [Bit] -> [Bit]
bitlistadd [] [] =[]
bitlistadd x [] = x
bitlistadd [] x= x
bitlistadd [Zero] [Zero] = [Zero]
bitlistadd [One] [Zero] = [One]
bitlistadd [Zero] [One] = [One]
bitlistadd [One] [One] = [Zero, One]
bitlistadd (x1 : xs1) (x2 : xs2) = bitlistadd [x1] [x2] ++ bitlistadd xs1 xs2

bsum :: [Bit] -> [Bit] -> Sign -> [Bit]
bsum b1 b2 sign = bitlistadd b1 b2

add :: Z -> Z -> Z
add (Z Minus b1) (Z Minus b2) = Z Minus (bsum b1 b2 Minus)
add (Z Plus b1) (Z Plus b2) = Z Plus (bsum b1 b2 Plus)
add (Z s1 b1) (Z s2 b2) = Z s3 (bsum b1 b2 s3) where
    s3 = case gtb (Z s1 b1) (Z s2 b2) of
          GT -> s1
          LT -> s2
          EQ -> Plus -- ==0

bitlistmul :: [Bit] -> [Bit] -> [Bit]
bitlistmul = undefined

mul :: Z -> Z -> Z
mul (Z Minus b1) (Z Minus b2) = Z Plus (bitlistmul b1 b2)
mul (Z Plus b1) (Z Plus b2) = Z Plus (bitlistmul b1 b2)
mul (Z _ b1) (Z _ b2) = Z Minus (bitlistmul b1 b2)


--https://stepik.org/lesson/4985/step/9?discussion=348505&unit=1083

test001 = (add (Z Plus []) (Z Plus [])) == Z Plus []
test002 = (add (Z Plus []) (Z Plus [One])) == Z Plus [One]
test003 = (add (Z Plus []) (Z Minus [One])) == Z Minus [One]
test011 = (add (Z Plus [Zero, One, One]) (Z Plus [One])) == Z Plus [One, One, One]
test012 = (add (Z Plus [Zero, One, One]) (Z Plus [Zero, One])) == Z Plus [Zero, Zero, Zero, One] 
---my Z Plus [Zero,Zero,One,One]
test013 = (add (Z Plus [Zero, One, One]) (Z Plus [Zero, One, One])) == Z Plus [Zero, Zero, One, One] 
---my Z Plus [Zero,Zero,One,Zero,One]
test021 = (add (Z Minus [Zero, One, One]) (Z Minus [One])) == Z Minus [One, One, One]
test022 = (add (Z Minus [Zero, One, One]) (Z Minus [Zero, One])) == Z Minus [Zero, Zero, Zero, One] 
---my Z Minus [Zero,Zero,One,One]
test023 = (add (Z Minus [Zero, One, One]) (Z Minus [Zero, One, One])) == Z Minus [Zero, Zero, One, One]
test031 = (add (Z Minus [Zero, One, One]) (Z Plus [One])) == Z Minus [One, Zero, One]
test032 = (add (Z Minus [Zero, One, One]) (Z Plus [Zero, One])) == Z Minus [Zero, Zero, One]
test033 = (add (Z Minus [Zero, One, One]) (Z Plus [Zero, One, One])) == Z Plus []
test041 = (add (Z Plus [Zero, One, One]) (Z Minus [One])) == Z Plus [One, Zero, One]
test042 = (add (Z Plus [Zero, One, One]) (Z Minus [Zero, One])) == Z Plus [Zero, Zero, One]
test043 = (add (Z Plus [Zero, One, One]) (Z Minus [Zero, One, One])) == Z Plus []
test051 = (add (Z Plus [One]) (Z Minus [One])) == Z Plus []
test052 = (add (Z Plus [One]) (Z Minus [One, One])) == Z Minus [Zero, One]
test053 = (add (Z Plus [One]) (Z Minus [Zero, One])) == Z Minus [One]
test054 = (add (Z Plus [One]) (Z Minus [Zero, Zero, Zero, One])) == Z Minus [One, One, One]
test055 = (add (Z Plus [One]) (Z Minus [Zero, One, Zero, One])) == Z Minus [One, Zero, Zero, One]
test056 = (add (Z Plus [Zero, One]) (Z Minus [Zero, One, One])) == Z Minus [Zero, Zero, One]
test057 = (add (Z Plus [Zero, One]) (Z Minus [Zero, Zero, One])) == Z Minus [Zero, One]
test058 = (add (Z Plus [One, Zero, One]) (Z Minus [Zero, One, Zero, One])) == Z Minus [One, Zero, One]
test101 = (mul (Z Plus []) (Z Plus [])) == emptyZ
test102 = (mul (Z Plus []) (Z Plus [One])) == emptyZ
test103 = (mul (Z Plus []) (Z Minus [One])) == emptyZ
test104 = (mul (Z Plus [One]) (Z Plus [])) == emptyZ
test105 = (mul (Z Minus [One]) (Z Plus [])) == emptyZ
test111 = (mul (Z Plus [One]) (Z Plus [One])) == Z Plus [One]
test112 = (mul (Z Minus [One]) (Z Plus [One])) == Z Minus [One]
test113 = (mul (Z Plus [One]) (Z Minus [One])) == Z Minus [One]
test114 = (mul (Z Minus [One]) (Z Minus [One])) == Z Plus [One]
test121 = (mul (Z Plus [One]) (Z Plus [Zero, One])) == Z Plus [Zero, One]
test122 = (mul (Z Plus [Zero, Zero, One]) (Z Plus [Zero, Zero, One])) == Z Plus [Zero, Zero, Zero, Zero, One]
test131 = (mul (Z Plus [One, Zero, One, Zero, One]) (Z Plus [One, One, One])) == Z Plus [One, One, Zero, Zero, One, Zero, Zero, One]
testAdd = test001 && test002 && test003 && test011 && test012 && test013 && test021 && test022 && test023 && test031 && test032 && test033 && test041 && test042 && test043 && test051 && test052 && test053 && test054 && test055 && test056 && test057 && test058
testMul = test101 && test102 && test103 && test104 && test105 && test111 && test112 && test113 && test114 && test121 && test122 && test131
testAll = testAdd && testMul
-}

{-
data Bit = Zero | One deriving (Show, Eq, Ord)
data Sign = Minus | Plus deriving (Show, Eq, Ord)
data Z = Z Sign [Bit] deriving (Show, Eq, Ord)

compare :: Bit -> Bit -> Ordering
compare Zero Zero = EQ
compare One One = EQ
compare One Zero = GT
compare Zero One = LT

compareli :: [Bit] -> [Bit] -> Ordering
compareli [] [] = EQ
compareli [x1] [x2] = Main.compare x1 x2
compareli a@(x1 : xs1) b@(x2 : xs2) = case Main.compare (last a) (last b) of
  EQ -> compareli (init a) (init b)
  GT -> GT
  LT -> LT

gtb :: Z -> Z -> Ordering
gtb (Z s1 b1) (Z s2 b2) = boo
  where
    boo
      | length b1 > length b2 = GT
      | length b1 < length b2 = LT
      | length b1 == length b2 = compareli b1 b2

bitlistadd :: [Bit] -> [Bit] -> [Bit]
bitlistadd [] [] = []
bitlistadd x [] = x
bitlistadd [] x = x
bitlistadd [Zero] [Zero] = [] --[Zero]
bitlistadd [One] [Zero] = [One]
bitlistadd [Zero] [One] = [One]
bitlistadd [One] [One] = [Zero, One]
--bitlistadd [One] (One : xs2) = [Zero, One] ++ bitlistadd xs1 xs2
bitlistadd (x1 : xs1) (x2: xs2) | length (bitlistadd [x1] [x2]) < 2 = bitlistadd [x1] [x2] ++ bitlistadd xs1 xs2
                                 | otherwise = [Zero, One] ++ bitlistadd (bitlistadd xs1 [One]) xs2--[Zero]++ bitlistadd (shiftplus xs1) xs2
--bitlistadd (x1 : xs1) (x2 : xs2) = bitlistadd [x1] [x2] ++ bitlistadd xs1 xs2
  

shiftplus :: [Bit] -> [Bit]
shiftplus [] = []
shiftplus [One] = [Zero, One] --[Zero]
shiftplus [Zero] = [One]
shiftplus (One : bs) = Zero : shiftplus bs
shiftplus (Zero : bs) = One : bs

shift ::[Bit] -> [Bit]
shift [] = []
shift [One] = [Zero]
shift [Zero] = [One]
shift (One : bs) = Zero : bs
shift (Zero : bs) = One : shift bs

bdiff :: [Bit] -> [Bit] -> [Bit]
bdiff [] [] = []
bdiff x [] = x
--bdiff [] x = x
bdiff [] [Zero] = [Zero]
bdiff [] [One] = [One,Zero,Zero]
bdiff [Zero] [Zero] = [Zero] 
bdiff [One] [Zero] = [One]
bdiff [Zero] [One] = [One, Zero]
bdiff [One] [One] = [Zero]
bdiff (x1 : xs1) (x2 : xs2) 
  | length (bdiff [x1] [x2]) == 3 =bdiff xs1 (shift  xs2)
  | length (bdiff [x1] [x2]) == 2 =[One]++bdiff (shift xs1) xs2
  | otherwise  = bdiff [x1] [x2] ++ bdiff xs1 xs2
  
  --if null xs1 && null xs2 then [One]  else [One] ++ bdiff [One] (bdiff xs1 xs2)

add :: Z -> Z -> Z
add (Z Minus b1) (Z Minus b2) = Z Minus (bitlistadd b1 b2)
add (Z Plus b1) (Z Plus b2) = Z Plus (bitlistadd b1 b2)
add (Z s1 b1) (Z s2 b2) = temp--Z s3 (bdiff b1 b2)
  where
    temp = case gtb (Z s1 b1) (Z s2 b2) of
      GT -> Z s1 (bdiff b1 b2)
      LT ->  Z s2 (bdiff b2 b1)
      EQ ->  Z Plus (bdiff b1 b2) -- ==0

bitlistmul :: [Bit] -> [Bit] -> [Bit]
bitlistmul = undefined

mul :: Z -> Z -> Z
mul (Z Minus b1) (Z Minus b2) = Z Plus (bitlistmul b1 b2)
mul (Z Plus b1) (Z Plus b2) = Z Plus (bitlistmul b1 b2)
mul (Z _ b1) (Z _ b2) = Z Minus (bitlistmul b1 b2)


--https://stepik.org/lesson/4985/step/9?discussion=348505&unit=1083

test001 = (add (Z Plus []) (Z Plus [])) == Z Plus []
test002 = (add (Z Plus []) (Z Plus [One])) == Z Plus [One]
test003 = (add (Z Plus []) (Z Minus [One])) == Z Minus [One]
test011 = (add (Z Plus [Zero, One, One]) (Z Plus [One])) == Z Plus [One, One, One]
test012 = (add (Z Plus [Zero, One, One]) (Z Plus [Zero, One])) == Z Plus [Zero, Zero, Zero, One] 
test013 = (add (Z Plus [Zero, One, One]) (Z Plus [Zero, One, One])) == Z Plus [Zero, Zero, One, One] 
test021 = (add (Z Minus [Zero, One, One]) (Z Minus [One])) == Z Minus [One, One, One]
test022 = (add (Z Minus [Zero, One, One]) (Z Minus [Zero, One])) == Z Minus [Zero, Zero, Zero, One] 
test023 = (add (Z Minus [Zero, One, One]) (Z Minus [Zero, One, One])) == Z Minus [Zero, Zero, One, One]
--Z Minus [Zero,One,One]
test031 = (add (Z Minus [Zero, One, One]) (Z Plus [One])) == Z Minus [One, Zero, One]
test032 = (add (Z Minus [Zero, One, One]) (Z Plus [Zero, One])) == Z Minus [Zero, Zero, One]
test033 = (add (Z Minus [Zero, One, One]) (Z Plus [Zero, One, One])) == Z Plus []
test041 = (add (Z Plus [Zero, One, One]) (Z Minus [One])) == Z Plus [One, Zero, One]
-- +my Z Plus [One,One,One]
test042 = (add (Z Plus [Zero, One, One]) (Z Minus [Zero, One])) == Z Plus [Zero, Zero, One]
test043 = (add (Z Plus [Zero, One, One]) (Z Minus [Zero, One, One])) == Z Plus []
test051 = (add (Z Plus [One]) (Z Minus [One])) == Z Plus []
test052 = (add (Z Plus [One]) (Z Minus [One, One])) == Z Minus [Zero, One]
test053 = (add (Z Plus [One]) (Z Minus [Zero, One])) == Z Minus [One]
-- +Z Minus [One,Zero]
test054 = (add (Z Plus [One]) (Z Minus [Zero, Zero, Zero, One])) == Z Minus [One, One, One]
test055 = (add (Z Plus [One]) (Z Minus [Zero, One, Zero, One])) == Z Minus [One, Zero, Zero, One]
test056 = (add (Z Plus [Zero, One]) (Z Minus [Zero, One, One])) == Z Minus [Zero, Zero, One]
test057 = (add (Z Plus [Zero, One]) (Z Minus [Zero, Zero, One])) == Z Minus [Zero, One]
test058 = (add (Z Plus [One, Zero, One]) (Z Minus [Zero, One, Zero, One])) == Z Minus [One, Zero, One]
test101 = (mul (Z Plus []) (Z Plus [])) == emptyZ
test102 = (mul (Z Plus []) (Z Plus [One])) == emptyZ
test103 = (mul (Z Plus []) (Z Minus [One])) == emptyZ
test104 = (mul (Z Plus [One]) (Z Plus [])) == emptyZ
test105 = (mul (Z Minus [One]) (Z Plus [])) == emptyZ
test111 = (mul (Z Plus [One]) (Z Plus [One])) == Z Plus [One]
test112 = (mul (Z Minus [One]) (Z Plus [One])) == Z Minus [One]
test113 = (mul (Z Plus [One]) (Z Minus [One])) == Z Minus [One]
test114 = (mul (Z Minus [One]) (Z Minus [One])) == Z Plus [One]
test121 = (mul (Z Plus [One]) (Z Plus [Zero, One])) == Z Plus [Zero, One]
test122 = (mul (Z Plus [Zero, Zero, One]) (Z Plus [Zero, Zero, One])) == Z Plus [Zero, Zero, Zero, Zero, One]
test131 = (mul (Z Plus [One, Zero, One, Zero, One]) (Z Plus [One, One, One])) == Z Plus [One, One, Zero, Zero, One, Zero, Zero, One]
testAdd = test001 && test002 && test003 && test011 && test012 && test013 && test021 && test022 && test023 && test031 && test032 && test033 && test041 && test042 && test043 && test051 && test052 && test053 && test054 && test055 && test056 && test057 && test058
testMul = test101 && test102 && test103 && test104 && test105 && test111 && test112 && test113 && test114 && test121 && test122 && test131
testAll = testAdd && testMul
-}


data Bit = Zero | One deriving (Show, Eq, Ord)
data Sign = Minus | Plus deriving (Show, Eq, Ord)
data Z = Z Sign [Bit] deriving (Show, Eq, Ord)

foo ::Bit ->Int
foo Zero = 0
foo One = 1

translate ::Z->Int
translate (Z Minus bits) = - (btr (map foo bits) 0)
translate (Z Plus bits) = btr (map foo bits) 0

btr :: [Int] -> Int -> Int
btr [] _= 0
btr (x : xs) n = (x * 2 ^ n) + btr xs (n + 1)

fooz :: Int -> Bit
fooz 0= Zero
fooz 1= One

translatez :: Int -> Z
translatez n |n==0 =Z Plus []
             |n<0 = Z Minus (map fooz (btrz (abs n)) )
             |n>0= Z Plus (map fooz ( norm (btrz n)) )

norm :: [Int] -> [Int]
norm v = reverse (norm1 (reverse v)) where
    norm1 [] = []
    norm1 [c] = [c]
    norm1 (b : bs) = if b == 1 then b : bs else (norm1 bs)

btrz :: Int -> [Int]
btrz 0 = []
btrz n =snd m : btrz (fst m) where m = n `divMod` 2

add :: Z -> Z -> Z
add z1 z2 = translatez( (translate z1) + (translate z2))

mul :: Z -> Z -> Z
mul z1 z2 = translatez (translate z1 * translate z2)

--https://stepik.org/lesson/4985/step/9?discussion=461116&thread=solutions&unit=1083
{-
data Bit = Zero | One deriving (Show, Eq)
data Sign = Minus | Plus deriving Show
data Z = Z Sign [Bit] deriving Show

add' :: [Bit] -> [Bit] -> [Bit]
add' []     []      = []
add' []     y       = y 
add' x      []      = x
add' (Zero:xs) (Zero:ys)  = Zero : add' xs ys
add' (One:xs) (One:ys)  = Zero :  add' [One] (add' xs ys)
add' (_:xs) (_:ys)  = One : add' xs ys

del :: [Bit] -> [Bit] -> [Bit]
del x y = reverse (dropWhile (/=One) (reverse (del' x y))) -- Удаление нулей на выходе
  where
    del' :: [Bit] -> [Bit] -> [Bit]
    del' []     []     = []
    del' x      []     = x
    del' []     y      = y
    del' (One:[]) (One:[]) = []
    del' (One:xs) (One:ys) = Zero : del' xs ys
    del' (Zero:xs) (Zero:ys) = Zero : del' xs ys
    del' (One:xs) (Zero:ys) = One : del' xs  ys
    del' (Zero:xs) (One:ys) = One : del' (del' xs  ys) [One]

maxOrEq :: [Bit] -> [Bit] -> Bool
maxOrEq x y = (add' (del x y) y == x)

maxIsFirst :: [Bit] -> [Bit] -> Bool
maxIsFirst x y = maxOrEq x y && x /= y

add :: Z -> Z -> Z
add x (Z _ [])              = x
add (Z _ []) y              = y
add (Z Plus x) (Z Plus y)   = (Z Plus (add' x y))
add (Z Minus x) (Z Minus y) = (Z Minus (add' x y))
add (Z Plus x) (Z Minus y)  | maxOrEq x y = Z Plus (del x y)
                            | otherwise = Z Minus (del y x)
add (Z _ x) (Z _ y) | maxIsFirst x y = Z Minus (del x y)
                    | otherwise = Z Plus (del y x)

mul :: Z -> Z -> Z
mul _ (Z _ []) = Z Plus []
mul (Z _ []) _ = Z Plus []
mul (Z Plus x) (Z Plus y) = Z Plus (mul' x (reverse y))
mul (Z Minus x) (Z Minus y) = Z Plus (mul' x (reverse y))
mul (Z _ x) (Z _ y) = Z Minus (mul' x (reverse y))

mul' :: [Bit] -> [Bit] -> [Bit]
mul' x [] = []
mul' x (Zero:xs) = mul' x xs
mul' x (One:xs) = add' ((map (\ x -> if x == One then Zero else x) xs) ++ x) (mul' x xs)
-}

-- ++https://stepik.org/lesson/4985/step/9?discussion=3057108&thread=solutions&unit=1083
{-
data Bit = Zero | One

data Sign = Minus | Plus

data Z = Z Sign [Bit]

-- порядок на битах
cB One Zero = GT
cB Zero One = LT
cB _ _ = EQ

-- порядок на знаках
cS Plus Minus = GT
cS Minus Plus = LT
cS _ _ = EQ

-- наследие порядка на мл.разряде ст.разряде
lCB x EQ = x
lCB _ y = y

-- порядок на списке битов
cBs [] [] = EQ
cBs x [] = GT
cBs [] y = LT
cBs (x : xs) (y : ys) = lCB (cB x y) (cBs xs ys)

-- сложение одиночных битов (мл.разряд, ст.разряд)
plB Zero Zero = (Zero, Zero)
plB One One = (Zero, One)
plB _ _ = (One, Zero)

-- сложение списков битов
pls x [] = x
pls [] y = y
pls (x : xs) (y : ys) = a : ([b] `pls` xs `pls` ys) where (a, b) = plB x y

-- вычитание одиночных битов (мл.разряд, ст.разряд)
mnB One Zero = (One, Zero)
mnB Zero One = (One, One)
mnB _ _ = (Zero, Zero)

-- вычитание списков битов
mns [] [Zero] = []
mns x [] = x
mns (x : xs) (y : ys) = a : (xs `mns` [b] `mns` ys) where (a, b) = mnB x y

-- убрать лишние нули в конце
kZer [Zero] = [Zero]
kZer x = reverse $ dropWhile (\x -> cB x Zero == EQ) $ reverse x

-- сложение Z-чисел
add :: Z -> Z -> Z
add (Z sx bsx) (Z sy bsy)
  | cS sx sy == EQ = Z sx (kZer $ pls bsx bsy)
  | cBs bsx bsy == GT = Z sx (kZer $ mns bsx bsy)
  | cBs bsx bsy == LT = Z sy (kZer $ mns bsy bsx)
  | otherwise = Z Plus [Zero]

-- список частных сумм произведения
ppl x y = zipWith f (iterate (Zero :) x) y
  where
    f t Zero = [Zero]
    f t One = t

-- произведение списка битов
prd x [Zero] = [Zero]
prd [Zero] y = [Zero]
prd x y = foldl pls [Zero] (ppl x y)

-- произведение Z-чисел
mul :: Z -> Z -> Z
mul (Z sx bsx) (Z sy bsy)
  | cS sx sy == EQ = Z Plus (prd bsx bsy)
  | otherwise = Z Minus (prd bsx bsy)
-}

--https://stepik.org/lesson/4985/step/9?discussion=3034123&thread=solutions&unit=1083
{-
data Bit = Zero | One deriving (Eq, Show)

data Sign = Minus | Plus deriving (Show)

data Z = Z Sign [Bit] deriving (Show)

add z1@(Z s1 b1) z2@(Z s2 b2) = toZ $ summator nb1 nb2
  where
    nb1 = normalize n z1
    nb2 = normalize n z2
    n = bitness [b1, b2]

mul z1@(Z s1 _) z2@(Z s2 _) = natToZ s $ natsMultiplier nb1 nb2
  where
    s = mulSigns s1 s2
    nb1 = normalizeNats z1
    nb2 = normalizeNats z2
    mulSigns Plus Plus = Plus
    mulSigns Minus Minus = Plus
    mulSigns _ _ = Minus

-- суммирует обычные двоичные числа равной разрядности
summator a b = fst $ foldr reducer ([], Zero) $ zip a b
  where
    onesCount = length . filter (== One)
    reducer (a, b) (xs, c) = case onesCount [a, b, c] of
      0 -> (Zero : xs, Zero)
      1 -> (One : xs, Zero)
      2 -> (Zero : xs, One)
      3 -> (One : xs, One)

-- преобразовние между положительными и отрицательными числами
negateBits xs = summator one . map mapper $ xs
  where
    one = padL Zero (length xs) [One]
    mapper One = Zero
    mapper Zero = One

-- умножение натуральных двоичных чисел столбиком
natsMultiplier xs ys = foldl sumNats [] products
  where
    products = map mapper factors
    factors = map fst . filter ((== One) . snd) . zip [0 ..] $ xs
    mapper i = ys ++ replicate (length xs - i - 1) Zero
    sumNats b1 b2 = summator (pad b1) (pad b2)
      where
        pad = padL Zero $ bitness [b1, b2]

-- необходимая разрядность для сложения чисел
bitness = (+ 2) . maximum . map length

-- конвертация между Z и обычными двоичными числами

changeBySign s = case s of Plus -> id; Minus -> negateBits

normalize n (Z s bs) = changeBySign s . padL Zero n . reverse $ bs

normalizeNats (Z s bs) = (Zero :) . reverse $ bs

toZ bss@(b : _) =
  let s = case b of Zero -> Plus; One -> Minus
   in natToZ s . changeBySign s $ bss

natToZ s = Z s . reverse . dropWhile (== Zero)

-- ...
padL s n l
  | length l >= n = l
  | otherwise = replicate (n - length l) s ++ l
-}

--https://stepik.org/lesson/4985/step/9?discussion=1523208&thread=solutions&unit=1083

{-
Попытался решить задачу без преобразований в Integer. Для этого было решено реализовать логические операции для класса типов BinaryType, 
представителями которого являются Bit (используемый в проверке решения), ShortBit (для более быстрой работы в консоли) и тип знака Sign 
(для него тоже актуальны логические операции). Также в данном классе типов реализован двоичный сумматор по модели Full Adder.
Для представления отрицательный чисел и работы с ними было решено использовать дополнительный код, что позволяет ограничиться 
реализацией только операции сложения

data Sign = Minus | Plus deriving (Show, Eq, Ord)

data Bit = Zero | One deriving (Show, Eq, Ord)

data ShortBit = O | I deriving (Show, Eq, Ord)

data Z = Z Sign [Bit] deriving (Show, Eq)

class (Show a, Eq a, Ord a) => BinaryType a where
  bitOne :: a
  bitZero :: a

  notBit :: a -> a
  notBit x
    | x == bitOne = bitZero
    | otherwise = bitOne

  andBit :: a -> a -> a
  andBit x y
    | x == bitOne && y == bitOne = bitOne
    | otherwise = bitZero

  orBit :: a -> a -> a
  orBit x y
    | x == bitZero && y == bitZero = bitZero
    | otherwise = bitOne

  xorBit :: a -> a -> a
  xorBit x y = (x `andBit` (notBit y)) `orBit` ((notBit x) `andBit` y)

  -- x - бит первого числа
  -- y - бит второго числа
  -- z - бит переноса
  -- sOut - сумма
  -- cOut - бит переноса (curry)
  fullAdder :: (a, a, a) -> (a, a)
  fullAdder (x, y, z) = (sOut, cOut)
    where
      sOut = xorBit z (xorBit x y)
      cOut = orBit (andBit x y) (andBit z (xorBit x y))

instance BinaryType Bit where
  bitOne = One
  bitZero = Zero

instance BinaryType ShortBit where
  bitOne = I
  bitZero = O

instance BinaryType Sign where
  bitZero = Minus
  bitOne = Plus

isEmptyZ :: Z -> Bool
isEmptyZ (Z s1 bs) = all (\x -> x == bitZero) bs

getFirstComplement :: Z -> Z
getFirstComplement (Z s xs') = (Z s (helper xs'))
  where
    helper :: BinaryType a => [a] -> [a]
    helper [] = []
    helper (x : xs) = notBit x : helper xs

getSecondComplement :: Z -> Z
getSecondComplement (Z Plus xs') = (Z Plus xs')
getSecondComplement (Z Minus xs') = (Z Minus (init $ addPositives bits [bitOne] bitZero))
  where
    (Z _ bits) = getFirstComplement (Z Plus xs')

alignBitArrays :: BinaryType a => [a] -> [a] -> ([a], [a])
alignBitArrays [] [] = (bitZero : [], bitZero : [])
alignBitArrays (x : xs) [] = ((x : xss), (bitZero : yss))
  where
    (xss, yss) = alignBitArrays xs []
alignBitArrays [] (y : ys) = ((bitZero : xss), (y : yss))
  where
    (xss, yss) = alignBitArrays [] ys
alignBitArrays (x : xs) (y : ys) = ((x : xss), (y : yss))
  where
    (xss, yss) = alignBitArrays xs ys

addPositives :: BinaryType a => [a] -> [a] -> a -> [a]
addPositives [] [] currBit
  | currBit == bitOne = bitOne : []
  | otherwise = bitZero : []
addPositives (x : xs) [] currBit = sBit : addPositives xs [] cBit
  where
    (sBit, cBit) = fullAdder (x, bitZero, currBit)
addPositives [] (y : ys) currBit = sBit : addPositives [] ys cBit
  where
    (sBit, cBit) = fullAdder (bitZero, y, currBit)
addPositives (x : xs) (y : ys) currBit = sBit : (addPositives xs ys cBit)
  where
    (sBit, cBit) = fullAdder (x, y, currBit)

binaryTypeToSign :: BinaryType a => a -> Sign
binaryTypeToSign x
  | x == bitZero = Plus
  | otherwise = Minus

add :: Z -> Z -> Z
add (Z s1 xss) (Z s2 yss) = getSecondComplement (Z resSign ans')
  where
    (ans', signBit) = (init bits, binaryTypeToSign (last bits))
      where
        bits = addPositives xs' ys' bitZero
          where
            ((Z _ xs'), (Z _ ys')) =
              (getSecondComplement (Z s1 xss_aligned), getSecondComplement (Z s2 yss_aligned))
              where
                (xss_aligned, yss_aligned) = alignBitArrays xss yss

    resSign = xorBit (xorBit s1 s2) signBit

mul :: Z -> Z -> Z
mul (Z s1 bs1) (Z s2 bs2) = (Z resSign resBits)
  where
    (Z _ resBits) = mulHelper (Z Plus bs1) (Z Plus bs2)
      where
        mulHelper z1 z2
          | not (isEmptyZ z2) = z1 `add` (mulHelper z1 (add z2 (Z Minus [bitOne])))
          | otherwise = (Z Plus [])
    resSign = notBit (xorBit s1 s2)
-}

--https://stepik.org/lesson/4985/step/9?discussion=3587667&thread=solutions&unit=1083

{-
data Bit = Zero | One

data Sign = Minus | Plus

data Z = Z Sign [Bit]

incb :: [Bit] -> [Bit]
incb [] = [One]
incb (b : bs) =
  case b of
    Zero -> (One : bs)
    _ -> Zero : (incb bs)

decb :: [Bit] -> [Bit]
decb [One] = []
decb (b : bs) =
  case b of
    One -> (Zero : bs)
    _ -> One : (decb bs)

inc :: Z -> Z
inc (Z Minus [One]) = Z Plus []
inc (Z s bs) =
  case s of
    Plus -> (Z s (incb bs))
    _ -> (Z s (decb bs))

dec :: Z -> Z
dec (Z Plus []) = Z Minus [One]
dec (Z s bs) =
  case s of
    Minus -> (Z s (incb bs))
    _ -> (Z s (decb bs))

add :: Z -> Z -> Z
add a (Z Plus []) = a
add (Z Plus []) b = b
add (Z as abs) (Z bs bbs) =
  case bs of
    Plus -> add (inc (Z as abs)) (dec (Z bs bbs))
    _ -> add (dec (Z as abs)) (inc (Z bs bbs))

mul :: Z -> Z -> Z
mul (Z Plus []) _ = (Z Plus [])
mul _ (Z Plus []) = (Z Plus [])
mul (Z as abs) (Z bs bbs) = (Z rs rbs)
  where
    rs = case as of
      Plus -> bs
      _ -> case bs of
        Minus -> Plus
        _ -> as
    (Z _ rbs) = addn (Z Plus abs) (Z Plus abs) (Z Plus bbs)
    addn a _ (Z Plus [One]) = a
    addn a b n = addn (add a b) b (dec n)
-}

--4.2.11
{-
Пусть определена следующая функция:
foo :: Bool -> Int
foo ~True = 1
foo False = 0
Что произойдет при вызове foo False?
-}
-- Функция вернет 1