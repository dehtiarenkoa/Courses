import GHC.Float
import Data.Fixed

import Data.Char --(isDigit)
--import qualified Data.List as L
--import Prelude hiding (lookup)

--4.3.3
{-
Определите тип записи, который хранит элементы лога. Имя конструктора должно совпадать с именем типа, 
и запись должна содержать три поля:
    timestamp — время, когда произошло событие (типа UTCTime);
    logLevel — уровень события (типа LogLevel);
    message — сообщение об ошибке (типа String).
Определите функцию logLevelToString, возвращающую текстуальное представление типа LogLevel, и функцию logEntryToString, 
возвращающую текстуальное представление записи в виде:
<время>: <уровень>: <сообщение>
Для преобразование типа UTCTime в строку используйте функцию timeToString.
-}

import Data.Time.Clock
import Data.Time.Format
--import System.Locale

timeToString :: UTCTime -> String
timeToString = formatTime defaultTimeLocale "%a %d %T"

data LogLevel = Error | Warning | Info

data LogEntry = LogEntry {timestamp :: UTCTime, logLevel :: LogLevel, message :: String}

logLevelToString :: LogLevel -> String
logLevelToString Error = "Error"
logLevelToString Info = "Info"
logLevelToString Warning = "Warning"

logEntryToString :: LogEntry -> String
logEntryToString j = (timeToString $ timestamp j) ++ ": " ++ (logLevelToString $ logLevel j) ++ ": " ++ (message j)

--logEntryToString j = (timeToString $ j & timestamp)  ++ ": " ++ (logLevelToString $ j & logLevel) ++ ": " ++ (j & message)

-- infixl 1 &
-- x & f = f x
-- import Data.Function
-- (&) x f = f x

--4.3.5
{-
Определите функцию updateLastName person1 person2, которая меняет фамилию person2 на фамилию person1.
-}
data Person = Person {firstName :: String, lastName :: String, age :: Int}

updateLastName :: Person -> Person -> Person
updateLastName p1 p2 = p2 {lastName = lastName p1}

--4.3.7
{-
Допустим мы объявили тип
data Shape = Circle Double | Rectangle Double Double
Что произойдет при объявлении такой функции:
isRectangle :: Shape -> Bool
isRectangle Rectangle{} = True
isRectangle _ = False
-}

--4.3.8
{-
Определить функцию abbrFirstName, которая сокращает имя до первой буквы с точкой, то есть, если имя было "Ivan", 
то после применения этой функции оно превратится в "I.". Однако, если имя было короче двух символов, то оно не меняется.
-}

data Person1 = Person1 {firstName1 :: String, lastName1 :: String, age1 :: Int}

abbrFirstName :: Person1-> Person1
abbrFirstName p = Person1 {firstName1= foo (firstName1 p), lastName1=lastName1 p, age1=age1 p}
foo :: [Char] -> [Char]
foo str =  if length str <2 then str else [head str]++"."

--4.4.3
{-
Реализуйте функции distance, считающую расстояние между двумя точками с вещественными координатами, 
и manhDistance, считающую манхэттенское расстояние между двумя точками с целочисленными координатами.
-}

data Coord a = Coord a a deriving Show

distance :: Coord Double -> Coord Double -> Double
distance (Coord x1 y1) (Coord x2 y2) = sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

manhDistance :: Coord Int -> Coord Int -> Int
manhDistance (Coord x1 y1) (Coord x2 y2) = abs(x2 - x1) + abs(y2 - y1)

--4.4.4
{-
Плоскость разбита на квадратные ячейки. Стороны ячеек параллельны осям координат. 
Координаты углов ячейки с координатой (0,0) имеют неотрицательные координаты. 
Один из углов этой ячейки имеет координату (0,0). С ростом координат ячеек увеличиваются координаты точек внутри этих ячеек.

Реализуйте функции getCenter, которая принимает координату ячейки и возвращает координату ее центра, 
и функцию getCell, которая принимает координату точки и возвращает номер ячейки в которой находится данная точка. 
В качестве первого аргумента обе эти функции принимают ширину ячейки.
-}

--data Coord a = Coord a a

getCenter :: Double -> Coord Int -> Coord Double
getCenter w (Coord x y) = Coord (x1 :: Double) (y1 :: Double) where
    --x1 ::Double
    xd = int2Double x
    yd = int2Double y
    --x1 = (int2Double (floor (xd / w))) * w + w / 2
    x1 = xd * w + w / 2
    --x11 = int2Double x1
    --x1 = (x `Data.Fixed.div'` w) * w + w / 2
    y1 = yd * w + w / 2
    -- zerox ::Double= (x `div` w)*w
    -- zeroy :: Double = (y `div` w) * w
-- getCenter1 w (Coord x y) = Coord (zero + w / 2) (zero + w / 2)  where
--     zero = (x `div` w) * w

getCell :: Double -> Coord Double -> Coord Int
getCell w (Coord x y) = Coord (floor (x / w) :: Int) (floor (y / w) :: Int)
--getCell w (Coord x y) = Coord ((x `div` w) :: Int) ((y `div` w) :: Int) 

--https://stepik.org/lesson/5746/step/4?discussion=182827&unit=1256
-- *Main> getCell 1 (Coord (-1) (-1))
-- Coord (-1) (-1)
-- *Main> getCell 1 (Coord 10 10)
-- Coord 10 10
-- *Main> getCenter 8 (Coord (-1) (-1))
-- Coord (-4.0) (-4.0)
-- *Main> getCenter 1 (Coord (-1) (-1))
-- Coord (-0.5) (-0.5)
-- *Main> getCell 1 (Coord 1 1)
-- Coord 1 1
-- *Main> getCell 10 (Coord 23 47)
-- Coord 2 4
-- *Main> getCenter 5.0 (Coord 2 3)
-- Coord 12.5 17.5
-- *Main> getCell 1 (Coord 0.5 0)
-- Coord 0 0
-- *Main> getCell 1 (Coord 1 1)
-- Coord 1 1

--4.4.6
{-
Реализуйте функцию, которая ищет в строке первое вхождение символа, 
который является цифрой, и возвращает Nothing, если в строке нет цифр.
-}
--import Data.Char (isDigit)
findDigit :: [Char] -> Maybe Char
findDigit str = foo str 0 where
    foo ::[Char]-> Int-> Maybe Char
    foo [] _ = Nothing
    foo (x:xs) n = if isDigit x then Just x else foo xs (n+1)
  --foo (x:xs) n = if isDigit x then Just (head.show $ n) else foo xs (n+1)

--4.4.7
{- 
Реализуйте функцию findDigitOrX, использующую функцию findDigit (последнюю реализовывать не нужно). 
findDigitOrX должна находить цифру в строке, а если в строке цифр нет, то она должна возвращать символ 'X'. 
Используйте конструкцию case.(https://stepik.org/lesson/4916/step/12?course=%D0%A4%D1%83%D0%BD%D0%BA%D1%86%D0%B8%D0%BE%D0%BD%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B5-%D0%BF%D1%80%D0%BE%D0%B3%D1%80%D0%B0%D0%BC%D0%BC%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5-%D0%BD%D0%B0-%D1%8F%D0%B7%D1%8B%D0%BA%D0%B5-Haskell&unit=1082)
-}
--import Data.Char (isDigit)
--findDigit :: [Char] -> Maybe Char
findDigitOrX :: [Char] -> Char
findDigitOrX str = case findDigit str of
     Just i -> i
     Nothing ->'X'

--4.4.8
{-
Maybe можно рассматривать как простой контейнер, например, как список длины 0 или 1. 
Реализовать функции maybeToList и listToMaybe, преобразующие Maybe a в [a] и наоборот 
(вторая функция отбрасывает все элементы списка, кроме первого).
-}
maybeToList :: Maybe a -> [a]
maybeToList (Just a) = [a]
maybeToList Nothing  = []

listToMaybe :: [a] -> Maybe a
listToMaybe (s:sx) = Just s
listToMaybe [] = Nothing

--4.4.9
{-


Реализуйте функцию parsePerson, которая разбирает строки вида 
firstName = John\nlastName = Connor\nage = 30 и возвращает либо результат типа Person, либо ошибку типа Error.
    Строка, которая подается на вход, должна разбивать по символу '\n' на список строк, 
    каждая из которых имеет вид X = Y. Если входная строка не имеет указанный вид, то функция должна возвращать ParsingError.
    Если указаны не все поля, то возвращается IncompleteDataError.
    Если в поле age указано не число, то возвращается IncorrectDataError str, где str — содержимое поля age.
    Если в строке присутствуют лишние поля, то они игнорируются.
-}
data Error = ParsingError | IncompleteDataError | IncorrectDataError String deriving (Show)

data Person2 = Person2 {firstName2 :: String, lastName2 :: String, age2 :: Int} deriving Show
-- parsePerson :: String -> Either Error Person2
-- parsePerson str = if (   (take f fi) == "firstName = " 
--                         && not (null a)
--                         && (take s se) == "lastName = " 
--                         && not (null b)
--                         && (take t th) == "age = " 
--                         && not (null c) 
--                         && (foldr (\x b1-> (isDigit x) && b1 ) True c)
--                         )
--                     then Right Person2 {firstName2 = a, lastName2 = b, age2 = c1}
--                     else 
--                         if (   (take f fi) == "firstName = " 
--                         && not (null a)
--                         && (take s se) == "lastName = " 
--                         && not (null b)
--                         && (take t th) == "age = "
--                         ) then Left (IncorrectDataError c)
--                         else 
--                             if (   (take f fi) == "firstName = " && not (null a)
--                             || (take s se) == "lastName = " && not (null b)
--                             || (take t th) == "age = "   && not (null c)
--                                ) then Left IncompleteDataError
--                             else Left ParsingError  where
--         li = lines str
--         foo:: [a] -> [a]->[(a,a)]
--         foo [] _= [(,)]
--         foo (x:xs) y= (if "firstName = " == take f x then ("firstName = ",drop f x) else 
--                             if "lastName = "== take s x then ("lastName = ",drop s x) else
--                                 if "age = "

--                               ):
--         fi = li !! 0
--         se = li !! 1
--         th = li !! 2
--         f = length "firstName = "
--         s = length "lastName = "
--         t = length "age = "
--         a = drop f fi
--         b = drop s se
--         c = drop t th        
--         c1:: Int
--         -- function to make an Int from [Char] 
--         -- "423" -> 423
--         --c1 = foldl (\su x -> (digitToInt x) * 10 ^ (if su == 0 then (if c == '0' then 1 else 0) else length (show su)) + su) 0 (reverse c)
--         --c1 = foldr (\x su -> (digitToInt x) * 10 ^ (if su==0 then (if [last c]  == "0" then 1 else 0) else length (show su) ) + su) 0 c
--         c1 = foldr (\(x, y) su -> (digitToInt x) * 10 ^ y + su) 0 (zip c (reverse [0 .. (length c-1)]))
--         -- Works!!

--         -- but !!!! -- [length c.. 0] ~> [] :(...
--         -- !!!! -- [5.. 0] ~> [] :(...
--         -- * Main Data.Char> c = "4520204" ~>45224

--         -- *Main Data.Char> parsePerson "firs'tName = John\niage = 3k"
--         -- *** Exception: Prelude.!!: index too large

-- v = parsePerson "firstNamie = John\niage = 3k"

--https://hackage.haskell.org/package/base-4.15.0.0/docs/Prelude.html#g:13
-- +++++ Read an integer from a string using readMaybe. If we succeed, return twice the integer; that is, apply (*2) to it. If instead we fail to parse an integer, return 0 by default:
-- >>> import Text.Read ( readMaybe )
-- >>> maybe 0 (*2) (readMaybe "5")
-- 10
-- >>> maybe 0 (*2) (readMaybe "")
-- 0

--------------- aaaaaaaaaaaaaaaaaaaa
-- read :: Read a => String -> a
-- The read function reads input from a string, which must be completely consumed by the input process. 
-- read fails with an error if the parse is unsuccessful, and it is therefore discouraged from being used 
--    in real applications. Use readMaybe or readEither for safe alternatives.
-- >>> read "123" :: Int
-- 123
-- >>> read "hello" :: Int
-- *** Exception: Prelude.read: no parse


parsePerson :: String -> Either Error Person2
parsePerson str = if (   a /= [] && b /= [] && c /= []
                        && (foldr (\x b1-> (isDigit x) && b1 ) True c)
                        )
                    then Right Person2 {firstName2 = a, lastName2 = b, age2 = c1}
                    else
                        if ( a /= [] && b /= [] && c /= []
                        ) then Left (IncorrectDataError c)
                        else
                            if (   a /= [] || b /= [] || c /= []
                               ) then Left IncompleteDataError
                            else Left ParsingError  where
        li = lines str
        foo :: [[Char]] -> [([Char], [Char])]
        foo [] = []
        foo (x:xs) = (if  fu "firstName = " x /=  ([],[]) then (fu "firstName = " x) else
                            if fu "lastName = " x /= ([],[]) then (fu "lastName = " x) else
                                if fu "age = " x /= ([], []) then (fu "age = " x) else
                                    ([], [])
                      ): foo xs
        r = foo li
        fu :: [Char] -> [Char] -> ([Char], [Char])
        fu str x = if str == take (length str) x then (str,drop (length str) x) else ([],[])
        f = length "firstName = "
        s = length "lastName = "
        t = length "age = "
        a, b, c :: [Char]
        a = maybe [] id (lookup "firstName = " r) -- was show intead of id and gave Left (IncorrectDataError "\"30\"") on "30"
        b = maybe [] id (lookup "lastName = " r) -- was show intead of id and gave Left (IncorrectDataError "\"30\"") on "30"
        c = maybe [] id (lookup "age = " r) -- was show intead of id and gave Left (IncorrectDataError "\"30\"") on "30"
        c1:: Int
        c1 = foldr (\(x, y) su -> (digitToInt x) * 10 ^ y + su) 0 (zip c (reverse [0 .. (length c-1)]))
--- works!!!
v = parsePerson "firstNamie = John\niage = 3k"
v1 = parsePerson "firstName = John\nlastName = Connor\nage = 30"
v2 = parsePerson "firstName = John\nage = 30"
v3 = parsePerson "first;Name = John\nlastName = Connor\nage = 30"
v4 = parsePerson "firstName = John\nlastName = Connor\nage = 30"


--https://stepik.org/lesson/5746/step/9?discussion=346398&thread=solutions&unit=125
{-
data Error = ParsingError | IncompleteDataError | IncorrectDataError String

data Person = Person {firstName :: String, lastName :: String, age :: Int}

parsePerson :: String -> Either Error Person
parsePerson s = makePerson (lineWith "firstName ") (lineWith "lastName ") (lineWith "age ")
  where
    info :: [(String, String)]
    info = map (break (== '=')) . lines $ s

    lineWith :: String -> Maybe String
    lineWith = flip lookup info

    makePerson :: Maybe String -> Maybe String -> Maybe String -> Either Error Person
    makePerson (Just firstNameA) (Just lastNameA) (Just ageA) =
      case (firstNameA, lastNameA, ageA) of
        ('=' : ' ' : firstName, '=' : ' ' : lastName, '=' : ' ' : age) ->
          case reads age of
            [(i, "")] -> Right $ Person firstName lastName i
            _ -> Left $ IncorrectDataError age
        _ -> Left ParsingError
    makePerson _ _ _ = Left IncompleteDataError
-}

--https://stepik.org/lesson/5746/step/9?discussion=1221408&thread=solutions&unit=1256
{-
import Data.List.Split
import Text.Read

data Error = ParsingError | IncompleteDataError | IncorrectDataError String

data Person = Person {firstName :: String, lastName :: String, age :: Int}

parsePerson :: String -> Either Error Person
parsePerson = parse_person_list . map (splitOn " = ") . lines
  where
    parse_person_list :: [[String]] -> Either Error Person
    parse_person_list list@(x : xs)
      | (and $ map (== 2) $ (map length) list) == False = Left ParsingError
      | ["firstName", s1] : ["lastName", s2] : ["age", s3] : (xs) <- list =
        if (readMaybe s3 :: Maybe Int) == Nothing
          then Left (IncorrectDataError s3)
          else Right (Person s1 s2 (read s3 :: Int))
      | otherwise = Left IncompleteDataError
    parse_person_list _ = Left ParsingError
-}

--https://stepik.org/lesson/5746/step/9?discussion=1014309&thread=solutions&unit=1256
{-
import Data.Char (isDigit)
import Data.List.Split (splitOn)

data Error = ParsingError | IncompleteDataError | IncorrectDataError String

data Person = Person {firstName :: String, lastName :: String, age :: Int}

parsePerson :: String -> Either Error Person
parsePerson str = case validators of
  [True, True, True] -> Right $ Person firstName lastName $ read age
  [False, _, _] -> Left ParsingError
  [_, False, _] -> Left IncompleteDataError
  [_, _, False] -> Left $ IncorrectDataError age
  where
    parsedList = map (splitOn " = ") $ splitOn "\n" str
    parsingOk = all ((2 ==) . length) parsedList
    (completeData, firstName, lastName, age) = case parsedList of
      (["firstName", fn] : ["lastName", ln] : ["age", age] : _) -> (True, fn, ln, age)
      _ -> (False, "", "", "")
    correctData = all isDigit age
    validators = [parsingOk, completeData, correctData]
-}

--4.4.11
{-
Укажите вид конструктора типов Either (Maybe Int).
-}
-- * -> *

--4.4.12
{-
Исправьте ошибку в приведенном коде.
eitherToMaybe :: Either a -> Maybe a
eitherToMaybe (Left a) = Just a
eitherToMaybe (Right _) = Nothing
-}

eitherToMaybe :: Either a b-> Maybe a
eitherToMaybe (Left a) = Just a
eitherToMaybe (Right _) = Nothing

-- 4.4.13
{-
Укажите все выражения, имеющие вид *
(Maybe Int, Either (Int -> (Char, Char)) Int)
Either (Int -> Int) Maybe
Maybe (Int -> Either Int Int)
Maybe Int -> Int
Maybe -> Int
Nothing
Either True False
Int -> Int
Either (Int -> (,)) Int 
-}

-- Prelude> :k Int -> (Char, Char)
-- Int -> (Char, Char) :: *

{-
(Maybe Int, Either (Int -> (Char, Char)) Int) :k=*
Either (Int -> Int) Maybe : k = error
Maybe (Int -> Either Int Int) :k=*
Maybe Int -> Int :k=*
Maybe -> Int :k=error
Nothing :k=error
Either True False :k=error
Int -> Int :k=*
Either (Int -> (,)) Int :k=error
-}

--4.4.15
{-
Допустим тип Coord определен следующим образом:
data Coord a = Coord a !a
Пусть определены следующие функции:
getX :: Coord a -> a
getX (Coord x _) = x
getY :: Coord a -> a
getY (Coord _ y) = y
Какие из следующих вызовов  вернут число 3?
Выберите все подходящие ответы из сп
-}

-- getY undefined
-- getX (Coord 3 3) //++
-- getY (Coord undefined 3) //++
-- getX (Coord 3 undefined)
-- getX (Coord undefined undefined)
-- getY (Coord 3 7)
-- getY (Coord 3 undefined)
-- getX (Coord undefined 3) 

--4.5.3
{-
Тип List, определенный ниже, эквивалентен определению списков из стандартной библиотеки в том смысле, 
что существуют взаимно обратные функции, преобразующие List a в [a] и обратно. Реализуйте эти функции.
data List a = Nil | Cons a (List a)
fromList :: List a -> [a]
fromList = undefined
toList :: [a] -> List a
toList = undefined
-}
data List a = Nil | Cons a (List a)
fromList :: List a -> [a]
fromList Nil = []
fromList (Cons a  Nil) = [a]
fromList (Cons b (Cons a Nil)) = b : [a]
fromList (Cons a b) = a : (fromList b)
toList :: [a] -> List a
toList [] = Nil
toList [a] = Cons a  Nil
toList (b : [a]) = Cons b (Cons a Nil)
toList (b : a) = Cons b (toList a)

--4.5.4
{-
Рассмотрим еще один пример рекурсивного типа данных:
data Nat = Zero | Suc Nat
Элементы этого типа имеют следующий вид: Zero, Suc Zero, Suc (Suc Zero), Suc (Suc (Suc Zero)), и так далее. 
Таким образом мы можем считать, что элементы этого типа - это натуральные числа в унарной системе счисления.
Мы можем написать функцию, которая преобразует Nat в Integer следующим образом:
fromNat :: Nat -> Integer
fromNat Zero = 0
fromNat (Suc n) = fromNat n + 1
Реализуйте функции сложения и умножения этих чисел, а также функцию, вычисляющую факториал.
-- https://stepik.org/lesson/7009/step/3?discussion=764083&unit=1472
-- Возвращаемые результаты:
-- *DT> fromList (Cons 'w' (Cons 'x' (Cons 'y' (Cons 'z' Nil))))
-- "wxyz"
-- *DT> :t fromList (Cons 'w' (Cons 'x' (Cons 'y' (Cons 'z' Nil))))
-- fromList (Cons 'w' (Cons 'x' (Cons 'y' (Cons 'z' Nil)))) :: [Char]
-- *DT> toList [1,2,3,4,5]
-- Cons 1 (Cons 2 (Cons 3 (Cons 4 (Cons 5 Nil))))
-- *DT> :t toList [1,2,3,4,5]
-- toList [1,2,3,4,5] :: Num a => List a
-}
data Nat = Zero | Suc Nat

fromNat :: Nat -> Integer
fromNat Zero = 0
fromNat (Suc n) = fromNat n + 1
-- *Main> fromNat (Suc (Suc (Suc (Suc Zero))))
-- 4

instance Show Nat where
  show = show . fromNat

fromInt :: Integer -> Nat
fromInt 0 = Zero
fromInt x = Suc ( fromInt (x -1))

add :: Nat -> Nat -> Nat
add Zero Zero = Zero
add Zero x =  x
add x Zero = x
add  x  y = fromInt (fromNat x + fromNat y)

--https://stepik.org/lesson/7009/step/4?discussion=517596&unit=1472
-- instance Show Nat where
--   show = show . fromNat
--https://stepik.org/lesson/7009/step/4?discussion=517596&reply=517604&unit=1472
-- fromInt :: Integer -> Nat
-- fromInt 0 = Zero
-- fromInt x = Suc $ fromInt (x -1)

mul :: Nat -> Nat -> Nat
mul Zero Zero =  Zero
mul Zero x = Zero
mul x (Suc Zero) = x
mul (Suc Zero) x = x
mul x Zero = Zero
mul x y = fromInt (fromNat x * fromNat y)

fac :: Nat -> Nat
fac Zero = Suc Zero
fac (Suc Zero) = Suc Zero
fac (Suc x) = fromInt (fromNat (Suc x) * fromNat (fac x))

--https://stepik.org/lesson/7009/step/4?discussion=764192&unit=1472
-- Возвращаемые значения:
-- *DT> add (Suc (Suc (Suc (Suc (Suc Zero))))) (Suc (Suc (Suc (Suc (Suc Zero)))))
-- Suc (Suc (Suc (Suc (Suc (Suc (Suc (Suc (Suc (Suc Zero)))))))))
-- *DT> mul (Suc (Suc (Suc Zero))) (Suc (Suc (Suc Zero)))
-- Suc (Suc (Suc (Suc (Suc (Suc (Suc (Suc (Suc Zero))))))))
-- *DT> fac (Suc (Suc (Suc Zero)))
-- Suc (Suc (Suc (Suc (Suc (Suc Zero)))))
-- {-
-- Anatomy:
-- fromNat (Suc (Suc (Suc (Suc (Suc Zero))))) = fromNat Suc (Suc (Suc (Suc Zero))) + 1
-- fromNat       Suc (Suc (Suc (Suc Zero))) + 1 = fromNat Suc (Suc (Suc Zero)) + 1 + 1
-- fromNat            Suc (Suc (Suc Zero)) + 1 + 1 = fromNat Suc (Suc Zero) + 1 + 1 + 1
-- fromNat                 Suc (Suc Zero) + 1 + 1 + 1 = fromNat Suc Zero + 1 + 1 + 1 + 1
-- fromNat                      Suc Zero + 1 + 1 + 1 + 1 = fromNat Zero + 1 + 1 + 1 + 1 + 1
--                                                                    0 + 1 + 1 + 1 + 1 + 1
-- -}

--4.5.5
{-
Тип бинарных деревьев можно описать следующим образом:
data Tree a = Leaf a | Node (Tree a) (Tree a)
Реализуйте функцию height, возвращающую высоту дерева, и функцию size, возвращающую количество узлов в дереве (и внутренних, и листьев). 
Считается, что дерево, состоящее из одного листа, имеет высоту 0.
-}

data Tree a = Leaf a | Node (Tree a) (Tree a) -- deriving Show

height :: Tree a -> Int
height (Leaf a) = 0
height (Node (Leaf b) (Leaf a)) = 1
height (Node x y) = max (height x) (height y) +1

size :: Tree a -> Int
size (Leaf a)=1
size (Node (Leaf b) (Leaf a)) = 3
size (Node x y) = size x + size y +1

--https://stepik.org/lesson/7009/step/5?discussion=764664&unit=1472
-- Результаты возвращаемые программой:
-- *DT> height (Leaf 1)
-- 0
--              _0______________________
--              _1______________________
-- *DT> height (Node (Leaf 1) (Leaf 1))
-- 1
--             _0______________________________________
--             _1___                          _1_______
--                    _2_____________________
-- *DT> height (Node (Node (Leaf 1) (Leaf 1)) (Leaf 1))
-- 2
--             _0______________________________________________________
--             _1____
--                   _2________________________________________________
-- *DT> height (Node (Node (Leaf 1) (Leaf 1)) (Node (Leaf 1) (Leaf 1)))
-- 2
--             _0_____________________________________________________________________
--             _1___                                           _1_____________________
--                   _2____                          _2_____
--                          _3____________________
-- *DT> height (Node (Node (Node (Leaf 1) (Leaf 1)) (Leaf 1)) (Node (Leaf 1) (Leaf 1)))
-- 3
-- *DT> height (Node (Node (Node (Leaf 1) (Leaf 1)) (Node (Leaf 1) (Leaf 1))) (Node (Leaf 1) (Leaf 1)))
-- 3
-- *DT> height (Node (Node (Node (Leaf 1) (Leaf 1)) (Node (Leaf 1) (Leaf 1))) (Node (Node (Leaf 1) (Leaf 1)) (Leaf 1)))
-- 3
-- *DT> height (Node (Node (Node (Leaf 1) (Leaf 1)) (Node (Leaf 1) (Leaf 1))) (Node (Node (Leaf 1) (Leaf 1)) (Node (Leaf 1) (Leaf 1))))
-- 3
-- *DT> height (Node (Node (Node (Node (Leaf 1) (Leaf 1)) (Leaf 1)) (Node (Leaf 1) (Leaf 1))) (Node (Node (Leaf 1) (Leaf 1)) (Node (Leaf 1) (Leaf 1))))
-- 4
-- *DT> size (Leaf 1)
-- 1
-- *DT> size (Node (Leaf 1) (Leaf 1))
-- 3
-- *DT> size (Node (Node (Leaf 1) (Leaf 1)) (Leaf 1))
-- 5
-- *DT> size (Node (Node (Leaf 1) (Leaf 1)) (Node (Leaf 1) (Leaf 1)))
-- 7

--4.5.6
{-
Теперь нам нужно написать функцию avg, которая считает среднее арифметическое всех значений в дереве. 
И мы хотим, чтобы эта функция осуществляла только один проход по дереву. 
Это можно сделать при помощи вспомогательной функции, возвращающей количество листьев и сумму значений в них. 
Реализуйте эту функцию.
-}

--data Tree a = Leaf a | Node (Tree a) (Tree a)

avg :: Tree Int -> Int
avg t =
  let (c, s) = go t
   in s `div` c
  where
    go :: Tree Int -> (Int, Int)
    go (Leaf a) = (1, a)
    go (Node (Leaf a) (Leaf b)) = (2, a+b)
    go (Node x y) = (fst (go x) + fst (go y), snd (go x) + snd (go y))

--4.5.8
{-
Исправьте определение функции expand
infixl 6 :+:
infixl 7 :*:
data Expr = Val Int | Expr :+: Expr | Expr :*: Expr
    deriving (Show, Eq)
expand :: Expr -> Expr
expand ((e1 :+: e2) :*: e) = expand e1 :*: expand e :+: expand e2 :*: expand e
expand (e :*: (e1 :+: e2)) = expand e :*: expand e1 :+: expand e :*: expand e2
expand (e1 :+: e2) = expand e1 :+: expand e2
expand (e1 :*: e2) = expand e1 :*: expand e2
expand e = e
так, чтобы она, используя дистрибутивность (а также, возможно, ассоциативность и коммутативность), 
всегда возвращала значение, эквивалентное данному и являющееся суммой произведений числовых значений. 
Например,
GHCi> expand $ (Val 1 :+: Val 2 :+: Val 3) :*: (Val 4 :+: Val 5)
Val 1 :*: Val 4 :+: (Val 1 :*: Val 5 :+: (Val 2 :*: Val 4 :+: (Val 2 :*: Val 5 :+: (Val 3 :*: Val 4 :+: Val 3 :*: Val 5))))
Примечание. 
Скобки в ответе могут быть расставлены по-другому или вообще отсутствовать, поскольку сложение ассоциативно. 
Слагаемые могут идти в другом порядке, поскольку сложение коммутативно.
-}
infixl 6 :+:

infixl 7 :*:

data Expr = Val Int | Expr :+: Expr | Expr :*: Expr
  deriving (Show, Eq)

--fooo :: Expr -> Expr-> Expr
-- fooo e1 e2 =
--   if (expand e1 :*: expand e2) == expand (expand e1 :*: expand e2)
--     then expand e1 :*: expand e2
--     else expand (expand e1 :*: expand e2)
-- fooo e1 e2 =  if (expand e1 :*: expand e2) == expand (expand e1 :*: expand e2)
--     then expand e1 :*: expand e2
--     else expand (expand e1 :*: expand e2)

expand :: Expr -> Expr
expand = expand1 . expand1
expand1 :: Expr -> Expr
-- expand ((e1 :+: e2) :*: (e3 :+: e4)) =
--   expand e1 :*: expand e3 :+: expand e2 :*: expand e3 :+: expand e1 :*: expand e4 :+: expand e2 :*: expand e4
expand1 ((e1 :+: e2) :*: e) = expand e1 :*: expand e :+: expand e2 :*: expand e
expand1 (e :*: (e1 :+: e2)) = expand e :*: expand e1 :+: expand e :*: expand e2
-- expand ((e1 :+: e2) :*: e) = expand (e1 :*:  e) :+: expand (e2 :*: e)
-- expand (e :*: (e1 :+: e2)) = expand (e :*: e1) :+: expand (e :*: e2)
-- expand ((e1 :+: e2) :*: e) = fooo e1 e :+: fooo e2 e
-- expand (e :*: (e1 :+: e2)) = fooo e1 e :+: fooo e2 e
-- expand ((e1 :+: e2) :*: (e3 :+: e4)) = expand (e1 :*: e3 ):+: expand (e2 :*: e3) :+: 
--  expand (e1 :*: e4) :+: expand (e2 :*:  e4)
-- expand (((Val s1) :+: (Val s2)) :*: e) = expand e :*: Val s1 :+: expand e :*: Val s2
-- expand (((Val s1) :+: e2) :*: e) = expand e :*: Val s1 :+: expand(expand e :*: expand e2)
-- expand ((e1 :+: (Val s2)) :*: e) = expand(expand e :*: expand e1) :+: expand e :*: Val s2
-- ++ expand ((e1 :+: e2) :*: e) = expand (expand e1 :*: expand e) :+: expand(expand e2 :*: expand e)
-- expand (e :*: ((Val s1) :+: (Val s2))) = expand e :*: Val s1 :+: expand e :*: Val s2
-- expand (e :*: ((Val s1) :+: e2)) = expand e :*: Val s1 :+: expand(expand e :*: expand e2)
-- expand (e :*: (e1 :+: (Val s2))) = expand(expand e :*: expand e1) :+: expand e :*: Val s2
-- ++ expand (e :*: (e1 :+: e2)) = expand (expand e :*: expand e1) :+: expand(expand e :*: expand e2)
-- expand ((e1 :+: e2) :*: e) = expand (e1 :*: e) :+: expand (e2 :*: e)
-- expand (e :*: (e1 :+: e2) ) = expand (e1 :*: e) :+: expand (e2 :*: e)
-- expand ((e1 :+: e2 :+: e3) :*: e) = expand e1 :*: expand e :+: expand e2 :*: expand e :+: expand e3 :*: expand e
--expand (e :*: (e1 :+: e2)) = expand e :*: expand e1 :+: expand e :*: expand e2
expand1 (e1 :+: e2) = expand e1 :+: expand e2
expand1 (e1 :*: e2) = expand e1 :*: expand e2
----expand (Val s1 :*: Val s2) = Val s1 :*: Val s2 --expand e1 :*: expand e2
----expand (e1 :*: e2) = expand(expand e1 :*: expand e2)
----expand (e1 :*: e2) = fooo e1 e2
--expand(expand e1 :+: expand e2)=expand (e1 :+: e2)
--expand a@(e1 :*: e2) = if a == expand a then  expand e1 :*: expand e2 else expand( expand e1 :*: expand e2)
--expand (e1 :*: e2) = if e1 ==  then expand e1 :*: expand e2 else expand (expand e1 :*: expand e2)
--expand ((Val s1) :*: (Val s2) ) = if e1 ==  then expand e1 :*: expand e2 else expand (expand e1 :*: expand e2)
expand1 e = e

-- *Main>  expand $ (Val 1 :+: Val 2 :+: Val 3) :*: (Val 4 :+: Val 5)
-- (Val 1 :+: Val 2) :*: (Val 4 :+: Val 5) :+: Val 3 :*: (Val 4 :+: Val 5)

-- *Main>  expand $ Val 1 :*: ((Val 2 :+: Val 3) :*: Val 4)
-- Val 1 :*: (Val 2 :*: Val 4 :+: Val 3 :*: Val 4)

-- *Main> expand $ (Val 1 :+: Val 2 :+: Val 3) :*: (Val 4 :+: Val 5):*: (Val 6 :+: Val 7)
-- (((Interrupted.

-- *Main> ((Val 2 :+: Val 3) :*: Val 4)==(Val 2:*: Val 4 :+: Val 3 :*: Val 4)
-- False

--https://stepik.org/lesson/7009/step/8?discussion=347051&thread=solutions&unit=1472
-- Определим (как обычно, по индукции) функцию `expandList`, которая принимает выражение и представляет его 
-- в виде суммы множителей, но только не суммирует их сразу, а возвращает в виде списка слагаемых 
-- (в таком виде очень легко определить эту функцию по индукции).
-- Тогда искомый ответ — это просто собранная сумма результата вызова функции `expandList`.
-- expand :: Expr -> Expr
-- expand = foldr1 (:+:) . expandList
--   where
--     expandList :: Expr -> [Expr]
--     expandList (Val i) = [Val i]
--     expandList (l :+: r) = expandList l ++ expandList r
--     expandList (l :*: r) = [e1 :*: e2 | e1 <- expandList l, e2 <- expandList r]

--https://stepik.org/lesson/7009/step/8?discussion=379208&thread=solutions&unit=1472
-- expand :: Expr -> Expr
-- expand = until (\x -> expand' x == x) expand'
-- expand' ((e1 :+: e2) :*: e) = expand e1 :*: expand e :+: expand e2 :*: expand e
-- expand' (e :*: (e1 :+: e2)) = expand e :*: expand e1 :+: expand e :*: expand e2
-- expand' (e1 :+: e2) = expand e1 :+: expand e2
-- expand' (e1 :*: e2) = expand e1 :*: expand e2
-- expand' e = e

--https://stepik.org/lesson/7009/step/8?discussion=1603513&thread=solutions&unit=1472
-- expand :: Expr -> Expr
-- expand ((e1 :+: e2) :*: e) = expand (e1 :*: e) :+: expand (e2 :*: e)
-- expand (e :*: (e1 :+: e2)) = expand (e :*: e1) :+: expand (e :*: e2)
-- expand (e1 :+: e2) = expand e1 :+: expand e2
-- expand (e1 :*: e2) = if a == e1 :*: e2 then a else expand a
--   where
--     a = (expand e1 :*: expand e2)
-- expand e = e

--https://stepik.org/lesson/7009/step/8?discussion=371899&thread=solutions&unit=1472
-- expand' :: Expr -> Expr
-- expand' ((e1 :+: e2) :*: e) = expand' e1 :*: expand' e :+: expand' e2 :*: expand' e
-- expand' (e :*: (e1 :+: e2)) = expand' e :*: expand' e1 :+: expand' e :*: expand' e2
-- expand' (e1 :+: e2) = expand' e1 :+: expand' e2
-- expand' (e1 :*: e2) = expand' e1 :*: expand' e2
-- expand' e = e
-- expand e
--   | expand' e == e = e
--   | otherwise = expand (expand' e)

--https://stepik.org/lesson/7009/step/8?discussion=586553&thread=solutions&unit=1472
-- В таких задачах мне нравится брать в помощники мощную систему типов Haskell. 
-- Нас просят преобразовать произвольное арифметическое выражение Expr к виду полинома. 
-- Если ввести вспомогательный тип данных Polynomial, то вся содержательная работа будет в функции
--  toPoly :: Expr -> Polynomial
-- Причём теперь эту функцию практически невозможно написать неправильно -- "недоупрощенные" выражения Expr просто нельзя поместить в тип Polynomial! Обратная функция
--  fromPoly :: Polynomial -> Expr
-- пишётся тривиально, а требуемая функция expand - это просто expand = fromPoly . toPoly

-- data Monomial = ValM Int | Monomial ::*:: Monomial
--   deriving (Show, Eq)

-- data Polynomial = Poly Monomial | Polynomial ::+:: Polynomial
--   deriving (Show, Eq)

-- toPoly :: Expr -> Polynomial
-- toPoly (Val i) = Poly (ValM i)
-- toPoly (e1 :+: e2) = toPoly e1 ::+:: toPoly e2
-- toPoly ((e1 :+: e2) :*: e) = toPoly (e1 :*: e :+: e2 :*: e)
-- toPoly (e :*: (e1 :+: e2)) = toPoly (e :*: e1 :+: e :*: e2)
-- toPoly (e1 :*: e2) = case (toPoly e1, toPoly e2) of
--   (Poly m1, Poly m2) -> Poly (m1 ::*:: m2)
--   (p1, p2) -> toPoly (fromPoly p1 :*: fromPoly p2)

-- fromMono :: Monomial -> Expr
-- fromMono (ValM i) = Val i
-- fromMono (m1 ::*:: m2) = fromMono m1 :*: fromMono m2

-- fromPoly :: Polynomial -> Expr
-- fromPoly (Poly m) = fromMono m
-- fromPoly (p1 ::+:: p2) = fromPoly p1 :+: fromPoly p2

-- expand :: Expr -> Expr
-- expand = fromPoly . toPoly

--4.6.3
{-
Пусть синоним типа Endo определен следующим образом:
type Endo a = a -> a
Выберите из списка типы, эквивалентные Endo (Endo Int).
 ++ (Int -> Int) -> Int -> Int
Int -> Int -> Int -> Int
 ++ (Int -> Int) -> (Int -> Int)
Int -> Int -> Int
(Int -> Int) -> Int
Int -> Int
Int -> (Int -> Int)
Int -> Int -> (Int -> Int)
Int 
-}

--4.6.5
{-
Выберите корректные объявления типов.
- newtype A a = A
+ newtype A a b = A a
+ newtype A a b = A b
- newtype A = A A A
- newtype A a b = A a b
- newtype A a = A a a
+ newtype A a = A a
- newtype A = A
+ newtype A = A A
- newtype A = A a 
-}

newtype A = A A deriving (Show)
-- ee :: A -> A
ee= A ee
-- *Main>  take 100 $ show ee
-- "A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A"

--https://stepik.org/lesson/7602/step/5?discussion=3035730&reply=3035760&unit=1473
-- newtype A = A A deriving (Show)
-- a = A a
-- GHCi > take 100 $ show a
-- "A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A (A"
-- newtype A = A B deriving (Show)
-- newtype B = B C deriving (Show)
-- newtype C = C A deriving (Show)
-- a = A b
-- b = B c
-- c = C a
-- GHCi > take 100 $ show a
-- "A (B (C (A (B (C (A (B (C (A (B (C (A (B (C (A (B (C (A (B (C (A (B (C (A (B (C (A (B (C (A (B (C (A"

--4.6.7
{-
Реализуйте представителя класса типов Monoid для типа Xor, в котором mappend выполняет операцию xor.
newtype Xor = Xor {getXor :: Bool}
  deriving (Eq, Show)
instance Monoid Xor where
  mempty = undefined
  mappend = undefined
-}
newtype Xor = Xor {getXor :: Bool}
  deriving (Eq, Show)

instance Semigroup Xor where
  Xor False <> Xor False = Xor False
  Xor False <> Xor True = Xor True
  Xor True <> Xor False = Xor True
  Xor True <> Xor True = Xor False

instance Monoid Xor where
  mempty = Xor False
  mappend (Xor False) (Xor False) = Xor False
  mappend (Xor False) (Xor True) = Xor True
  mappend (Xor True) (Xor False) = Xor True
  mappend (Xor True) (Xor True) = Xor False

--https://stepik.org/lesson/7602/step/7?discussion=346641&thread=solutions&unit=1473  
-- newtype Xor = Xor {getXor :: Bool}
--   deriving (Eq, Show)
-- instance Monoid Xor where
--   mempty = Xor False
--   Xor x `mappend` Xor y = Xor (x /= y)

--https://stepik.org/lesson/7602/step/7?discussion=639817&thread=solutions&unit=1473
-- newtype Xor = Xor {getXor :: Bool}
--   deriving (Eq, Show)
-- instance Monoid Xor where
--   mempty = Xor False
--   mappend = (Xor .) . (/=)


--https://stepik.org/lesson/7602/step/8?discussion=391295&reply=391315&unit=1473
--f (g (h x)) = f $ g $ h x = (f . g . h) x

--4.6.9
{-
Реализуйте представителя класса типов Monoid для Maybe' a так, чтобы mempty не был равен Maybe' Nothing. 
Нельзя накладывать никаких дополнительных ограничений на тип a, кроме указанных в условии.
newtype Maybe' a = Maybe' {getMaybe :: Maybe a}
  deriving (Eq, Show)
instance Monoid a => Monoid (Maybe' a) where
  mempty = undefined
  mappend = undefined
-}

newtype Maybe' a = Maybe' {getMaybe :: Maybe a}
  deriving (Eq, Show)

instance Semigroup a => Semigroup (Maybe' a)
(<>) a b = undefined

-- instance Semigroup a => Semigroup (Maybe' a) where 
--  (Maybe' Nothing) <> (Maybe' Nothing) = Maybe' Nothing
--  (Maybe' Nothing) <> _ = Maybe' Nothing
--  _ <> (Maybe' Nothing) = Maybe' Nothing
--  (Maybe' a) <> (Maybe' b) = mempty --Maybe' (a <> b) --a<>b = Maybe' Nothing


 --Maybe' (Just a) <> Maybe' (Just b) = mempty --(Just a <> Just b)
  -- Maybe' a <> Maybe' Nothing = Maybe' Nothing
  -- Maybe' a <> Maybe' b = Maybe' b

instance Monoid a => Monoid (Maybe' a) where
  mappend (Maybe' Nothing) (Maybe' a) = Maybe' Nothing
  mappend (Maybe' a) (Maybe' Nothing) = Maybe' Nothing
  mappend (Maybe' a) (Maybe' b) = Maybe'(mappend a b) --mempty --Maybe' Nothing
  mempty = Maybe' (Just mempty)
--works!!!! but why and how??????????


--  mempty = Maybe' (Just mempty) --mempty
-- *Main> test0
-- "passed"

--instance Semigroup a => Semigroup (Maybe' a) where
--(<>) a b = undefined

-- instance Semigroup (Maybe' a) where
--  Maybe' Nothing <> Maybe' a = Maybe' Nothing  
--  Maybe' a <> Maybe' Nothing = Maybe' Nothing
--  Maybe' a <> Maybe'  b = Maybe' b

-- instance Monoid a => Monoid (Maybe' a) where  
-- --  mappend (Maybe' Nothing) (Maybe' a) = Maybe' Nothing  
-- --  mappend (Maybe' a) (Maybe' Nothing) = Maybe' Nothing
-- --  mappend (Maybe' a) (Maybe' b) = Maybe' a
--  mempty = Maybe' (Just mempty)


-- newtype Maybe' a = Maybe' {getMaybe :: Maybe a}
--   deriving (Eq, Show)
-- instance Monoid a => Monoid (Maybe' a) where
--   mappend (Maybe' Nothing) (Maybe' a) = Maybe' Nothing
--   mappend (Maybe' a) (Maybe' Nothing) = Maybe' Nothing
--   mappend (Maybe' a) (Maybe' b) = Maybe' b
--   mempty = Maybe' mempty
--Failed . Wrong answer

-- newtype Maybe' a = Maybe' {getMaybe :: Maybe a}
--   deriving (Eq, Show)
-- instance Monoid a => Monoid (Maybe' a) where
--   mappend (Maybe' Nothing) (Maybe' Nothing) = Maybe' Nothing
--   mappend (Maybe' Nothing) _ = Maybe' Nothing
--   mappend _ (Maybe' Nothing) = Maybe' Nothing
--   mappend (Maybe' (Just a)) (Maybe' (Just b)) = Maybe' Nothing
--   mempty = Maybe' mempty
-- Failed. Runtime error
-- Error: main: boo

-- newtype Maybe' a = Maybe' {getMaybe :: Maybe a}
--   deriving (Eq, Show)
-- instance Monoid a => Monoid (Maybe' a) where
--   mappend (Maybe' Nothing) _ = Maybe' Nothing
--   mappend _ (Maybe' Nothing) = Maybe' Nothing
--   mappend (Maybe' a) (Maybe' b) = mempty --Maybe'(mappend a b)
--   mempty = Maybe' mempty --(Maybe Nothing)
-- Failed. Runtime error
-- Error: main: boo

-- newtype Maybe' a = Maybe' {getMaybe :: Maybe a}
--   deriving (Eq, Show)
-- instance Monoid a => Monoid (Maybe' a) where
--   mappend (Maybe' Nothing) _ = Maybe' Nothing
--   mappend _ (Maybe' Nothing) = Maybe' Nothing
--   mappend (Maybe' a) (Maybe' b) = Maybe' (mappend a b)
--   mempty = mempty 
--Failed. Runtime error
--Error:main: <<loop>>

-- newtype Maybe' a = Maybe' {getMaybe :: Maybe a}
--   deriving (Eq, Show)
-- instance Monoid a => Monoid (Maybe' a) where
--   mappend (Maybe' Nothing) _ = Maybe' Nothing
--   mappend _ (Maybe' Nothing) = Maybe' Nothing
--   mappend (Maybe' a) (Maybe' b) = Maybe' (mappend a b)
--   mempty = Maybe' mempty
--Failed. Wrong answer




--https://stepik.org/lesson/7602/step/9?discussion=344814&unit=1473
test0 = if (mempty :: Maybe' [Int]) == Maybe' Nothing then "failed" else "passed"
-- newtype Maybe' a = Maybe' {getMaybe :: Maybe a}
--   deriving (Eq, Show)
-- instance Semigroup a => Semigroup (Maybe' a)
-- (<>) a b = undefined
-- instance Monoid a => Monoid (Maybe' a) where
--   mappend (Maybe' Nothing) (Maybe' a) = Maybe' Nothing
--   mappend (Maybe' a) (Maybe' Nothing) = Maybe' Nothing
--   mappend (Maybe' a) (Maybe' b) = Maybe' (mappend a b) --mempty --Maybe' Nothing
--   mempty = Maybe' (Just mempty)
-- *Main> test0
-- "passed"

--https://stepik.org/lesson/7602/step/9?discussion=1079348&unit=1473
{-
Видел много комментариев в духе "не понял как, но решил".

Внушительное количество успешных решений 1156 на данный момент говорит о том, что задача многим по силам, но процент удачных попыток 13% говорит, что среди них много решивших задачу наугад.

Забудем пока про Maybe' a, возьмем тип Ordering, у него всего 3 возможных значения: EQ, LT, GT, и он является моноидом с нейтральным элементом EQ
mappend обладает таким поведением (распишу все варианты для наглядности)

EQ `mappend` EQ = EQ
EQ `mappend` LT = LT
EQ `mappend` GT = GT
LT `mappend` EQ = LT
LT `mappend` LT = LT
LT `mappend` GT = LT
GT `mappend` EQ = GT
GT `mappend` LT = GT
GT `mappend` GT = GT

Теперь возьмем тип Maybe Ordering, у него уже 4 возможных значения: 
Just EQ, Just LT, Just GT и Nothing, и он так же является моноидом с нейтральным элементом Nothing

Nothing `mappend` Nothing = Nothing
Nothing `mappend` Just EQ = Just EQ
Nothing `mappend` Just LT = Just LT
Nothing `mappend` Just GT = Just GT
Just EQ `mappend` Nothing = Just EQ
Just EQ `mappend` Just EQ = Just EQ
Just EQ `mappend` Just LT = Just LT
Just EQ `mappend` Just GT = Just GT
Just LT `mappend` Nothing = Just LT
Just LT `mappend` Just EQ = Just LT
Just LT `mappend` Just LT = Just LT
Just LT `mappend` Just GT = Just LT
Just GT `mappend` Nothing = Just GT
Just GT `mappend` Just EQ = Just GT
Just GT `mappend` Just LT = Just GT
Just GT `mappend` Just GT = Just GT

Может показаться, что Just EQ по прежнему является нейтральным элементом.
Очень просто доказать, что это не так:

Just EQ `mappend` Nothing = Just EQ

Если бы Just EQ был нейтральным элементом, результатом был бы Nothing

Является ли Maybe Ordering несмотря на это законным моноидом? Безусловно.
Он не нарушает ни одного закона моноидов, а большего от него и не требуется.
Он вообще мог бы иметь другой нейтральный элемент, например Just GT, главное - соблюдение законов.

Вернемся к Maybe'. Maybe' Ordering так же содержит 4 возможных значения:
Maybe' (Just EQ), Maybe' (Just LT), Maybe' (Just GT) и Maybe' Nothing

Какое из них выбрать нейтральным элементом? Maybe' Nothing отпадает по условию.
Конечно, если бы требовалось реализовать именно моноид Maybe' Ordering, 
можно было бы выбрать любое из оставшихся значений.

Но требуется реализовать представителя Monoid для Maybe' a, 
но о типе a нам ничего неизвестно кроме того, что он также является моноидом.
Значит мы можем только сконструировать нейтральный элемент Maybe' a на основе 
единственного известного значения a - его нейтрального элемента.
В случае Maybe' Ordering это мог бы быть Maybe' (Just EQ)

Как должен себя вести mappend для нашего моноида?
Как угодно, пока это не нарушает законы моноидов. 
Он не обязан ни в коей мере повторять поведение mappend для типа a, 
хоть у вас и не получится реализовать mappend для Maybe' a не используя mappend для a, 
по тем же причинам что и с mempty

Подумайте, как можно реализовать mappend, если мы даже не можем отличить нейтральный 
элемент a от остальных его значений, при этом Maybe' Nothing не является нейтральными, а значит

Maybe' Nothing `mappend` mempty ≡ mempty `mappend` Maybe' Nothing ≡ Maybe' Nothing  
-}

{-
--https://stepik.org/lesson/7602/step/9?discussion=115663&reply=115704&unit=1473
я тут поигрался немного, и сдается мне, что любой моноид, обернутый аппликативным функтором естественным образом 
сам формирует моноид - его нейтральный элемент будет pure от нейтрального элемента внутреннего моноида, 
а бинарная операция - так же операция внутреннего моноида, примененная к содержимому аппликативного функтора. 
Таким образом можно неограниченно оборачивать моноиды в любое количество любых аппликативных функторов. 
Удовлетворение аксиомам моноида такой конструкции я не проверял, но сильно подозреваю, что там все хорошо. 
--https://stepik.org/lesson/7602/step/9?discussion=115663&reply=115903&unit=1473
Покрутил немного с законами моноидов, функторов и аппликативов и вроде (если нигде не наврал и не ошибся) 
доказал левую и правую единицы образованного таким образом моноида, ассоциативность бинарной операции пока не осилил. 
Выкладки (после знаков равенства записаны номера тождеств, которые применялись при преобразованиях, тождества выписаны не все, 
а только которые потребовались для доказательства):
--https://stepik.org/lesson/7602/step/9?discussion=115663&reply=115904&unit=1473
Моноид:
m1) mappend mempty = id

Функтор:
f1) fmap id = id
f2) fmap (f . g) = fmap f . fmap g

Аппликативный функтор:
a1) pure f <*> pure x = pure (f x)
a2) u <*> pure y = pure ($ y) <*> u
a3) pure f <*> x = fmap f x

Новый моноид:
1) mempty' = pure mempty
2) mappend' a b = fmap mappend a <*> b (mappend' = liftA2 mappend)

Левая единица:
mappend' mempty' x = (1,2)
fmap mappend (pure mempty) <*> x = (a3)
(pure mappend) <*> (pure mempty) <*> x = (a1)
pure (mappend mempty) <*> x = (5)
fmap (mappend mempty) x = (m1)
fmap id x = (f1)
x

Правая единица:
mappend' x mempty' = (1,2)
fmap mappend x <*> (pure mempty) = (a2)
pure ($ mempty) <*> (fmap mappend x) = (a3)
fmap ($ mempty) (fmap mappend x) = (f2)
fmap ($ mempty . mappend) x =
fmap (mappend mempty) x = (m1)
fmap id x = (f1)
x
--https://stepik.org/lesson/7602/step/9?discussion=115663&reply=116431&unit=1473
Вести с полей - если я нигде не ошибся, то я доказал и ассоциативность такого аппликативно обернутого моноида (через многабукаф), 
и теперь он полностью законен для любого уровня вложенности :)
--https://stepik.org/lesson/7602/step/9?discussion=115663&reply=116435&unit=1473
Моноид:
m1) mappend mempty = id
m2) mappend (mappend x y) z = mappend x (mappend y z)

Функтор:
f1) fmap id = id
f2) fmap (f . g) = fmap f . fmap g

Аппликативный функтор:
a1) pure f <*> pure x = pure (f x)
a2) u <*> pure y = pure ($ y) <*> u
a3) pure f <*> x = fmap f x
a4) pure (.) <*> u <*> v <*> w = u <*> (v <*> w)

Новый моноид:
1) mempty' = pure mempty
2) mappend' a b = fmap mappend a <*> b (mappend' = liftA2 mappend)

Левая единица:

mappend' mempty' x = (1,2)
fmap mappend (pure mempty) <*> x = (a3)
(pure mappend) <*> (pure mempty) <*> x = (a1)
pure (mappend mempty) <*> x = (a3)
fmap (mappend mempty) x = (m1)
fmap id x = (f1)
x

Правая единица:

mappend' x mempty' = (1,2)
fmap mappend x <*> (pure mempty) = (a2)
pure ($ mempty) <*> (fmap mappend x) = (a3)
fmap ($ mempty) (fmap mappend x) = (f2)
fmap ($ mempty . mappend) x =
fmap (mappend mempty) x = (m1)
fmap id x = (f1)
x

Ассоциативность бинарной операции:

Левый порядок:
mappend' (mappend' x y) z = (1,2)
fmap mappend (fmap mappend x <*> y) <*> z = (a3)
pure mappend <*> (pure mappend <*> x <*> y) <*> z = (a4)
pure (.) <*> pure mappend <*> (pure mappend <*> x) <*> y <*> z = (a4)
pure (.) <*> (pure (.) <*> pure mappend) <*> pure mappend <*> x <*> y <*> z = (a4)
pure (.) <*> pure (.) <*> pure (.) <*> pure mappend <*> pure mappend <*> x <*> y <*> z = (a1)
pure (((((.) (.)) (.)) mappend) mappend) <*> x <*> y <*> z = (lambdabot pl)
pure ((mappend .) . mappend) <*> x <*> y <*> z = (lambdabot unpl)
pure (\ x y z -> mappend (mappend x y) z) <*> x <*> y <*> z = 
pure f <*> x <*> y <*> z where f = \ x y z -> mappend (mappend x y) z

Правый порядок:
mappend' x (mappend' y z) = (1,2)
fmap mappend x <*> (fmap mappend y <*> z) = (a3)
pure mappend <*> x <*> (pure mappend <*> y <*> z) = (a4)
pure (.) <*> (pure mappend <*> x) <*> (pure mappend <*> y) <*> z = (a4)
pure (.) <*> (pure (.) <*> (pure mappend <*> x)) <*> pure mappend <*> y <*> z = (a4)
pure (.) <*> pure (.) <*> pure (.) <*> (pure mappend <*> x) <*> pure mappend <*> y <*> z = (a4)
pure (.) <*> (pure (.) <*> pure (.) <*> pure (.)) <*> pure mappend <*> x <*> pure mappend <*> y <*> z = (a4)
pure (.) <*> pure (.) <*> (pure (.) <*> pure (.)) <*> pure (.) <*> pure mappend <*> x <*> pure mappend <*> y <*> z = (a4)
pure (.) <*> (pure (.) <*> pure (.)) <*> pure (.) <*> pure (.) <*> pure (.) <*> pure mappend <*> x <*> pure mappend <*> y <*> z = (a4)
pure (.) <*> pure (.) <*> pure (.) <*> pure (.) <*> pure (.) <*> pure (.) <*> pure (.) <*> pure mappend <*> x <*> pure mappend <*> y <*> z = (a1)
pure ((((((((.) (.)) (.)) (.)) (.)) (.)) (.)) mappend) <*> x <*> pure mappend <*> y <*> z = (a2)
pure ($ mappend) <*> (pure ((((((((.) (.)) (.)) (.)) (.)) (.)) (.)) mappend) <*> x) <*> y <*> z = (lambdabot pl)
pure ($ mappend) <*> (pure ((((.) . (.)) .) mappend) <*> x) <*> y <*> z = (a4)
pure (.) <*> pure ($ mappend) <*> pure ((((.) . (.)) .) mappend) <*> x <*> y <*> z = (lambdabot pl)
pure ((. mappend) . (.) . mappend) <*> x <*> y <*> z = (lambdabot unpl)
pure (\ x y z -> mappend x (mappend y z)) <*> x <*> y <*> z =
pure g <*> x <*> y <*> z where g = \ x y z -> mappend x (mappend y z)

f = g (m2) => ч.т.д.
-}

--https://stepik.org/lesson/7602/step/9?discussion=117180&unit=1473
{-
Подскажите, пожалуйста, что не так с решением 6633913? Идея такая: 
пусть дано множество с операцией (S,⋅) (S, \cdot) (S,⋅) и инволюция σ \sigma σ на нём. 
Рассмотрим на S новую операцию a∘b:=σ(σa⋅σb) a \circ b := \sigma(\sigma a \cdot \sigma b) a∘b:=σ(σa⋅σb). 
Я утверждаю, что из коммутативности и ассоциативности ⋅ \cdot ⋅ следуют эти свойства для ∘ \circ ∘:
    1) b∘a=σ(σb⋅σa)=//⋅коммутативна//=σ(σa⋅σb)=a∘b 
    2) a∘(b∘c)=σ(σa⋅σ(b⋅c))=σ(σa⋅σσ(σb⋅σc))=σ(σa⋅(σb⋅σc))==//⋅ассоциативна//=σ((σa⋅σb)⋅σc)=(a⋅b)⋅c 
Если 1∈S  1 \in S  1∈S  -- это нейтральный элемент для ⋅ , то σ(1)    будет нейтральным элементом для ∘ .
Отсюда идея решения: взять в качестве ⋅ сложение в Maybe a, а в качестве σ   взять перестановку элементов Nothing и Just mempty.
P.S. Пока писал комментарий, тесты к задаче поменялись, и от типа a нельзя требовать принадлежности Eq a :( .
-}

--4.6.10
{-
Ниже приведено определение класса MapLike типов, похожих на тип Map. 
Определите представителя MapLike для типа ListMap, определенного ниже как список пар ключ-значение. 
Для каждого ключа должно храниться не больше одного значения. 
Функция insert заменяет старое значение новым, если ключ уже содержался в структуре.
import qualified Data.List as L
import Prelude hiding (lookup)
class MapLike m where
  empty :: m k v
  lookup :: Ord k => k -> m k v -> Maybe v
  insert :: Ord k => k -> v -> m k v -> m k v
  delete :: Ord k => k -> m k v -> m k v
  fromList :: Ord k => [(k, v)] -> m k v
  fromList [] = empty
  fromList ((k, v) : xs) = insert k v (fromList xs)
newtype ListMap k v = ListMap {getListMap :: [(k, v)]}
  deriving (Eq, Show)
-}
-- import qualified Data.List as L
-- import Prelude hiding (lookup)

class MapLike m where
  empty1 :: m k v
  lookup1 :: Ord k => k -> m k v -> Maybe v
  insert1 :: Ord k => k -> v -> m k v -> m k v
  delete1 :: Ord k => k -> m k v -> m k v
  fromList1 :: Ord k => [(k, v)] -> m k v
  fromList1 [] = empty1
  fromList1 ((k, v) : xs) = insert1 k v (fromList1 xs)

newtype ListMap k v = ListMap {getListMap :: [(k, v)]}
  deriving (Eq, Show)

instance MapLike ListMap where
  -- toList :: Ord k => m k v-> [(k, v)] 
  -- toList empty = []  

 empty1 = ListMap []
  -- empty1 = fromList1 (ListMap [])
  -- Couldn't match expected type ‘[(k, v)]’ with actual type ‘ListMap k0 v0’

 --lookup1 _ (ListMap []) = Nothing
 lookup1 x (ListMap []) = Nothing
 -- ????? why doesnt work?????? -- lookup1 x empty1 = Nothing
 --lookup1 x (ListMap [(k,v)]) = if x == k then Just v else Nothing
 lookup1 x (ListMap ((k, v) : xs)) = if x == k then Just v else lookup1 x (ListMap xs)
  --lookup1 x (fromList1   h) = if x == k then Just v else Nothing

 delete1 x (ListMap []) = ListMap []
 delete1 k (ListMap xs) = case (lookup1 k (ListMap xs)) of --look of
   Nothing -> ListMap xs --fromList1((k, v) : xs)
   Just w -> ListMap  [x | x <- xs, fst x /= k]

--  delete1 x (ListMap ((k, v) : xs)) = fst (helper x (k,v):xs) [] where
--    helper :: a ->[a]->[a]

    --  helper :: Ord k => k -> ListMap k v -> ListMap k v -> ListMap k v -> ListMap k v
    --  helper k (ListMap []) b = (ListMap b) []
    --  helper x (ListMap ((k, v) : xs)) b = if x == k then b ++ ListMap xs else (b ++ [(k, v)] ++ helper x (ListMap  xs) b)


 --insert1 k v (ListMap []) = ListMap [(k, v)]
--  insert1 k v (ListMap xs) = if (lookup k xs) == Nothing then ListMap ((k, v) :xs) else
--    ListMap ((takeWhile (/=(k,loo)) xs)++[(k,v)]++(dropWhile (==(k,loo)) (takeWhile (/=(k,loo)) xs))) 
--     where  Just loo = lookup k xs

 insert1 k v (ListMap []) = ListMap [(k, v)] --fromList1 [(k, v)]
 insert1 k v (ListMap xs) = case (lookup1 k (ListMap xs)) of --look of
   Nothing ->  ListMap  ((k, v) : xs)--fromList1((k, v) : xs)
   Just w ->  ListMap  ([(k, v)] ++ [ x | x <- xs, fst x/=k])--(filter (/= (k, _)) xs))
   --Just w ->  fromList1 ((takeWhile (/= (k, w)) xs) ++ [(k, v)] ++ (dropWhile (== (k, w)) (takeWhile (/= (k, w)) xs)))

--  insert1 k v (ListMap xs) = case (lookup k xs) of --look of
--    Nothing -> ListMap ((k, v) : xs)
--    Just w ->  ListMap ((takeWhile (/= (k, w)) xs) ++ [(k, v)] ++ (dropWhile (== (k, w)) (takeWhile (/= (k, w)) xs)))

    --  look = lookup1 k (ListMap xs)
    --  Just loo = lookup k xs

--  insert1 k v (ListMap xs) = if look == Nothing then ListMap ((k, v) :xs) else
--     ListMap ((takeWhile (/=(k,loo)) xs)++[(k,v)]++(dropWhile (==(k,loo)) (takeWhile (/=(k,loo)) xs))) 
--         where  
--           look = lookup1 k (ListMap xs)
--           Just loo = lookup k xs


-- GHCi> let list = [(3,"a"),(1,"x"),(8,"q"),(6,"qwerty"),(7,"bar")]
-- GHCi> getListMap $ insert 6 "" (ListMap list)
-- [(3,"a"),(1,"x"),(6,""),(8,"q"),(6,"qwerty"),(7,"bar")]

--https://stepik.org/lesson/7602/step/10?discussion=349554&thread=solutions&unit=1473
-- instance MapLike ListMap where
--   empty = ListMap []
--   lookup key (ListMap list) = L.lookup key list
--   insert key value map = ListMap $ (key, value) : (getListMap $ delete key map)
--   delete key (ListMap list) = ListMap $ filter (\(x, _) -> x /= key) list

--4.6.12
{-
Реализуйте представителя MapLike для типа ArrowMap, определенного ниже.
import Prelude hiding (lookup)
class MapLike m where
  empty :: m k v
  lookup :: Ord k => k -> m k v -> Maybe v
  insert :: Ord k => k -> v -> m k v -> m k v
  delete :: Ord k => k -> m k v -> m k v
  fromList :: Ord k => [(k, v)] -> m k v
newtype ArrowMap k v = ArrowMap {getArrowMap :: k -> Maybe v}
-}
--import Prelude hiding (lookup)

class MapLike2 m where
  empty2 :: m k v
  lookup2 :: Ord k => k -> m k v -> Maybe v
  insert2 :: Ord k => k -> v -> m k v -> m k v
  delete2 :: Ord k => k -> m k v -> m k v
  fromList2 :: Ord k => [(k, v)] -> m k v
newtype ArrowMap k v = ArrowMap {getArrowMap :: k -> Maybe v}
b :: ArrowMap k v -> k -> Maybe v
b = getArrowMap
-- fooo :: ArrowMap k v
-- fooo = ArrowMap (\k -> Maybe v)
instance MapLike2 ArrowMap where
  empty2 = ArrowMap (\x -> Nothing ) --const nothing --undefined :: m k v
    --lookup2 k m = getArrowMap  m k
  -- lookup2 :: Ord k => k -> m k v -> Maybe v
  -- lookup2 x (ArrowMap m) = if x == k1 then v1 else lookup2 x m1 where
  --   v1 = getArrowMap (ArrowMap m) k1
  --   m1 k1 v1 =  m
  -- lookup2 x m22@(ArrowMap m33) = if x == kk1 then vv1 else lookup2 x m33
  --lookup2 x ArrowMap (\x -> Nothing) = Nothing
  ---lookup2 x m22@(ArrowMap m33) = getArrowMap m22 x
  lookup2 k  m = getArrowMap m k--(ArrowMap m) $ k
  ----lookup2 x m22@(ArrowMap m33) = if getArrowMap m22 x == Nothing then lookup2 x m33 else getArrowMap m22 x --if getArrowMap m22 x ==Nothing then lookup2 x m33 else vv1
    -- where
      --mm1 :: m k v
      --mm1  = m k v
      --v1 = getArrowMap m k1   
      -- vv1 = getArrowMap m22 x
      --mm1 kk1 vv1 = getArrowMap m22 x
     --v1 = getArrowMap (ArrowMap m) k1
   --d = ArrowMap (kx -> vx) 
   
  --lookup2 kx (ArrowMap m) =  Nothing --(\kx empty2 -> Nothing) -- undefined :: Maybe v
  --lookup2 k (ArrowMap m) = getArrowMap (ArrowMap m) k --(\kx (ArrowMap m) -> if getArrowMap == kx then (Maybe v) else Nothing)
  --lookup1 x (ListMap ((k, v) : xs)) = if x == k then Just v else lookup1 x (ListMap xs)

  --insert2 k v (ArrowMap m) = ArrowMap (\k -> Just v)
  --insert2 k v (ArrowMap m) = if (getArrowMap (ArrowMap m) k )== (Just v) then ArrowMap m else ArrowMap m where (Just g) = getArrowMap (ArrowMap m) k
  -- insert2 k v (ArrowMap m) = if foo == Nothing then ArrowMap (\k -> Just v) else ArrowMap (\k -> Just v) where
  --   foo = lookup2 k (ArrowMap m) --ArrowMap m
  --insert2 k v empty2 = helper k v (ArrowMap m) empty2 where

  insert2 k v m = ArrowMap (\k' -> if k' == k then Just v else getArrowMap m k') 
  -- insert2 k v m = m k'
  ---- insert2 xx vw m22@(ArrowMap m44)  = case lookup2 xx m22 of 
  ----   Nothing -> ArrowMap m22 
  ----   _ -> undefined--if lookup2 x m22 == Nothing then create x v m22 empty2 else
    
    -- create  xx vw m22 empty2 where --ArrowMap (\k -> Just v) where
    -- -- me :: p -> p1 -> ArrowMap k v -> ArrowMap k v
    -- -- me k v (ArrowMap m) = ArrowMap m
    -- create :: MapLike2 m=>Ord k => k -> v -> m k v -> m k v -> m k v
    -- create  k v m prev = case lookup2 k m of
    --   Nothing ->m 
    --   _ ->  undefined 

  -- insert1 k v (ListMap []) = ListMap [(k, v)] --fromList1 [(k, v)]
  -- insert1 k v (ListMap xs) = case (lookup1 k (ListMap xs)) of --look of
  --  Nothing ->  ListMap  ((k, v) : xs)--fromList1((k, v) : xs)
  --  Just w ->  ListMap  ([(k, v)] ++ [ x | x <- xs, fst x/=k])

--insert2 k v m = if (lookup2 k m) == Nothing then m  else m  --undefined :: m k v
  delete2 k m = ArrowMap (\k' -> if k' == k then Nothing else getArrowMap m k')
  --delete2 k m = empty2 -- ArrowMap (\x -> Nothing )  --undefined :: m k v
  -- delete1 x (ListMap []) = ListMap [] 
  -- delete1 k (ListMap xs) = case (lookup1 k (ListMap xs)) of --look of
  --  Nothing -> ListMap xs --fromList1((k, v) : xs)
  --  Just w -> ListMap  [x | x <- xs, fst x /= k]

  fromList2 [] = empty2 --ArrowMap (\x -> Nothing)
  fromList2 ((k,v):xs) = insert2 k v (fromList2 xs)
  -- fromList2 [(k, v)] = ArrowMap (\k -> Just v)
  -- fromList2 ((k, v):s) = ArrowMap (\k-> Just v)  --undefined :: m k v

-- +++++https://stackoverflow.com/questions/56674254/understanding-key-value-constructors-in-haskell/56674377#56674377
--empty = const Nothing
--type KVS a b = a -> Maybe b
--insert k v kvs = \k' -> if k' == k then Just v else kvs k'
-- !!!works!!!
{-
newtype ArrowMap k v = ArrowMap {getArrowMap :: k -> Maybe v}
instance MapLike ArrowMap where
  empty = ArrowMap (\x -> Nothing)
  lookup k m = getArrowMap m k
  insert k v m = ArrowMap (\k' -> if k' == k then Just v else getArrowMap m k')
  delete k m = ArrowMap (\k' -> if k' == k then Nothing else getArrowMap m k')
  fromList [] = empty
  fromList ((k, v) : xs) = insert k v (fromList xs)
-}

--https://stepik.org/lesson/7602/step/12?discussion=3828006&unit=1473
simpleMap = ArrowMap (\k -> if mod k 2 == 0 then Just k else Nothing)
testr :: Bool
testr = (lookup2 2 $ insert2 1 1 simpleMap) == Just 2

  --Failed. Runtime error Error: main: main.hs:(16,3)-(17,45): Non-exhaustive patterns in function fromList
  --fromList ((k,v):xs) = insert k v (fromList xs)    

  --lookup k (insert k v empty) == Just v
  --lookup 2 $ insert 1 1 simpleMap --правильный ответ  Just 2
testMap :: ArrowMap Int String
testMap = fromList2 [(1, "one"), (2, "two"), (3, "three")]
test1 = lookup2 1 testMap == Just "one"
test2 = lookup2 4 testMap == Nothing

testf x = case x of {42 -> Just "ANSWER"; 0 -> Just "ZERO"; _ -> Nothing}
testam = ArrowMap testf

{-
https://stepik.org/lesson/7602/step/12?discussion=349243&unit=1473
import Prelude hiding (fromList, lookup)
https://stepik.org/lesson/7602/step/12?discussion=349243&reply=349346&unit=1473
testMap :: ArrowMap Int String
testMap = fromList [(1, "one"), (2, "two"), (3, "three")]
test1 = lookup 1 testMap == Just "one"
test2 = lookup 4 testMap == Nothing
--
GHCi> let f x = case x of {42 -> Just "ANSWER"; 0 -> Just "ZERO"; _ -> Nothing}
GHCi> let am = ArrowMap f
GHCi> lookup 43 am
Nothing
GHCI> lookup 0 am
Just "ZERO"
GHCi> lookup 42 am
Just "ANSWER"
GHCi> let am' = insert 43 "SUCC ANSWER" am
GHCi> lookup 43 am'
Just "SUCC ANSWER"
GHCI> lookup 0 am'
Just "ZERO"
GHCi> lookup 42 am'
Just "ANSWER"
-}

{-
am = fromList [(1, "a"), (2, "b"), (3, "c")]
lookup 1 am -- Just "a"
lookup 4 $ insert 4 "d" am -- Just "d"
lookup 1 $ delete 2 am -- Nothing
-}
{-
arrowMap :: ArrowMap Int String
arrowMap = ArrowMap $ \k ->
  case k of
    1 -> Just "Petya"
    2 -> Just "Vasya"
    _ -> Nothing
-}

{--
--Lambda-calculus:
--https://habr.com/ru/post/215807/
--https://habr.com/ru/post/215991/
tru = λt.λf.t 	Двухаргументная функция, всегда возвращающая первый аргумент
fls = λt.λf.f 	Двухаргументная функция, всегда возвращающая второй аргумент
Оператор if под такие булевы константы будет имеет вид:
if = λb. λx. λy.b x y
Здесь b — tru или fls, x — ветка then, y — ветка else.
Посмотрим, как это будет работать:
if fls t e
Поскольку условие if ложно (fls), то должно возвращаться выражение из ветки else (e в нашем случае).
--
конъюнкция (логическое «и») будет выглядеть так:
and = λx. λy. x y fls
and получает два булевых значения x и y. 
Первым подставляется x (каррирование). 
Если он является tru (tru y fls после редукции), то вернётся y, который затем тоже «проверится» на истинность. 
Таким образом, итоговое tru мы получим только в случае, когда и x, и y «истинны». Во всех других вариантах ответом будет fls.
--
or = λx.λy. x tru y
--
not = λx. x fls tru
--}

--http://neerc.ifmo.ru/wiki/index.php?title=%D0%9B%D1%8F%D0%BC%D0%B1%D0%B4%D0%B0-%D0%B8%D1%81%D1%87%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5
--succ = λn . λs . λz . s (n s z)
--plus = λn . λm . λs . λz . n s (m s z)
--(plus 3 ¯ 3 ¯) (+ 1) 0 ≡ 6
--(plus ((plus 2 5) (+ 1) 0) 4) (+ 1) 0 ≡ 11
--mult=λn . λm . λs . λz . n (m s) z
-- power = λn . λm . λs . λz . m n s z
-- (power 3 ¯ (succ 3 ¯)) (+ 1) 0 ≡ 81
-- minus = λn . λm . m pred n -----if n>m
-- true = λa . λb . a
-- false = λa . λb . b
-- if=λp . λt . λe . p t e
-- and=λn . λm . if n m false
-- or=λn . λm . if n true m
-- not=λb . if b false true
-- isZero = λn . n (λc . false) true
-- pair = λa . λb . λt . t a b
-- fst = λp . p true
-- snd = λp . p false
-- pred=λn . λs . λz. snd (n (λp . pair (s (fstp)) (fstp)) (pair z z))
-- le = λn . λm . isZero (minus n m)
-- less = λn . λm . le n (predm)
-- eq = λn . λm . and (isZero (minus n m)) (isZero (minus m n))


--https://bmsdave.github.io/blog/y-combinator/
-- fix f = f (fix f)
-- const42 x = 42
-----
--Y = λf.(λx.f(x x)) (λx.f(x x))
--yCombinator
-- --
-- const factorialRecursive = g => (f => n => n === 0 ? 1 : n * f(n - 1))((x) => g(g)(x));
-- const factorial = factorialRecursive(factorialRecursive)
-- console.log(factorial(5)); // 120
-- --
-- const factorial = (
--     g => (f => n => n === 0 ? 1 : n * f(n - 1))((x) => g(g)(x)) // функция
-- )(g => (f => n => n === 0 ? 1 : n * f(n - 1))((x) => g(g)(x))); // сам снова в качестве аргумента
-- console.log(factorial(5)); // 120
-- --
-- // `g => g(g)` - функция, которая берет функцию и вызывает ее сама с собой
-- const factorial = (g => g(g))(g => (f => n => n === 0 ? 1 : n * f(n - 1))((x) => g(g)(x)))
-- console.log(factorial(5)); // 120
-- --
-- const factorialGenerator = f => n => n === 0 ? 1 : n * f(n - 1);
-- const yCombinator = f => (g => g(g))(g => f((x) => g(g)(x)))
-- const factorial = yCombinator(factorialGenerator);
-- console.log(factorial(5)); // 120

--https://gist.github.com/decorator-factory/da2e1e8d3bec4b967bffaaf4a5578e07
{--
import Prelude hiding (lookup)

class MapLike m where
    empty :: m k v
    lookup :: Ord k => k -> m k v -> Maybe v
    insert :: Ord k => k -> v -> m k v -> m k v
    delete :: Ord k => k -> m k v -> m k v
    fromList :: Ord k => [(k,v)] -> m k v


newtype ArrowMap k v = ArrowMap { getArrowMap :: k -> Maybe v }

{-
  define ArrowMap as an instance of MapLike here
-}

data TestCase a = TestCase {name::String, actual::a, expected::a} deriving Show
data TestResult a = Success | Fail (TestCase a) deriving Show

run :: Eq a => [TestCase a] -> TestResult a
run = mconcat . map test where
    test :: Eq a => TestCase a -> TestResult a
    test (TestCase s x y) | x == y    = Success
                        | otherwise = Fail (TestCase s x y)

assert :: Bool -> String -> TestCase Bool
assert bool name' = TestCase {name = name', actual = bool, expected = True}

instance Semigroup (TestResult a) where
    Success <> x = x
    x <> Success = x
    (Fail testCase) <> _ = (Fail testCase)

instance Monoid (TestResult a) where
    mempty = Success

tests :: [TestCase Bool]
tests = [
    assert (
        isNothing $ lookup 42 (empty::ArrowMap Int ())
    ) "Empty map is empty",

    assert (
        let m = (fromList [('a', 1), ('b', 2), ('c', 3)])::ArrowMap Char Int in
        and [
            isJust 1 $ lookup 'a' m,
            isJust 2 $ lookup 'b' m,
            isJust 3 $ lookup 'c' m,
            isNothing $ lookup 'd' m
        ]
    ) "fromList works",

    assert (
        let m = ArrowMap (\x -> if x `elem` "ab" then Just x else Nothing) in
            and [
                isJust 'a' $ lookup 'a' m,
                isJust 'b' $ lookup 'b' m,
                isNothing $ lookup 'c' m
            ]
    ) "simple lookup works",

    assert (
        let m1 = (fromList [('a', 1), ('b', 2), ('c', 3)])::ArrowMap Char Int
            m2 = (delete 'b' m1)
        in
            and [
                isJust 1 $ lookup 'a' m2,
                isNothing $ lookup 'b' m2,
                isJust 3 $ lookup 'c' m2,
                isNothing $ lookup 'd' m2
            ]
    ) "you can't look up an element after deleting it",

    assert(
        let m1 = (fromList [('a', 1), ('b', 2), ('c', 3)])::ArrowMap Char Int
            m2 = (insert 'd' 4 m1)
        in
            and [
                isJust 1 $ lookup 'a' m2,
                isJust 2 $ lookup 'b' m2,
                isJust 3 $ lookup 'c' m2,
                isJust 4 $ lookup 'd' m2
            ]
    ) "after inserting a new pair, you can look it up"
  ]
  where
    isNothing Nothing  = True
    isNothing _        = False
    isJust x  (Just y) = (x == y)
    isJust _  _        = False
--}