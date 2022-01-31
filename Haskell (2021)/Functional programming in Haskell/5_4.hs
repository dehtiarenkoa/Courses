import Data.Char
import Data.List
import System.Directory

import Control.Monad (ap, liftM)
import Data.Monoid
--5.4.4
{-
Рассмотрим язык арифметических выражений, которые состоят из чисел, скобок, операций сложения и вычитания. 
Конструкции данного языка можно представить следующим типом данных:
data Token = Number Int | Plus | Minus | LeftBrace | RightBrace 
    deriving (Eq, Show)
Реализуйте лексер арифметических выражений. Для начала реализуйте следующую функцию:
asToken :: String -> Maybe Token
Она проверяет, является ли переданная строка числом (используйте функцию isDigit из модуля Data.Char), знаком "+" или "-", 
открывающейся или закрывающейся скобкой. Если является, то она возвращает нужное значение обёрнутое в Just, в противном случае - Nothing:
GHCi> asToken "123"
Just (Number 123)
GHCi> asToken "abc"
Nothing
Далее, реализуйте функцию tokenize:
tokenize :: String -> Maybe [Token]
Функция принимает на вход строку и если каждое слово является корректным токеном, то она возвращает список этих токенов, завёрнутый в Just. 
В противном случае возвращается Nothing. 
Функция должна разбивать входную строку на отдельные слова по пробелам (используйте библиотечную функцию words). 
Далее, полученный список строк должен быть свёрнут с использованием функции asToken и свойств монады Maybe:
GHCi> tokenize "1 + 2"
Just [Number 1,Plus,Number 2]
GHCi> tokenize "1 + ( 7 - 2 )"
Just [Number 1,Plus,LeftBrace,Number 7,Minus,Number 2,RightBrace]
GHCi> tokenize "1 + abc"
Nothing
Обратите внимание, что скобки отделяются пробелами от остальных выражений!
---
data Token = Number Int | Plus | Minus | LeftBrace | RightBrace
  deriving (Eq, Show)
asToken :: String -> Maybe Token
asToken = undefined
tokenize :: String -> Maybe [Token]
tokenize input = undefined
-}
data Token = Number Int | Plus | Minus | LeftBrace | RightBrace
    deriving (Eq, Show)

asToken :: String -> Maybe Token
asToken str
  | all ((== True) . isDigit) str = Just (Number (read str :: Int))--  all (== True) (map isDigit str) ==  all ((== True) . isDigit) str
  | str == "+" = Just Plus
  | str == "-" = Just Minus
  | str == "(" = Just LeftBrace
  | str == ")" = Just RightBrace
  | otherwise = Nothing

tokenize :: String -> Maybe [Token]
tokenize input = if all (/= Nothing) w then Just (map (\(Just x)-> x) w) else Nothing where
  w ::  [Maybe Token]
  w = map asToken (words input)

--https://stepik.org/lesson/8439/step/4?discussion=680267&thread=solutions&unit=1574
{-
asToken :: String -> Maybe Token
asToken t
  | t == "+" = Just Plus
  | t == "-" = Just Minus
  | t == "(" = Just LeftBrace
  | t == ")" = Just RightBrace
  | all isDigit t && any isDigit t = Just $ Number (read t :: Int)
  | otherwise = Nothing

tokenize :: String -> Maybe [Token]
--tokenize s = sequence $ map asToken $ words s where
tokenize s = cvt $ words s
  where
    cvt :: [String] -> Maybe [Token]
    cvt [] = Just []
    cvt (w : ws) = do
      t <- asToken w -- Token
      ts <- cvt ws -- [Token]
      return (t : ts)      
-}

--https://stepik.org/lesson/8439/step/4?discussion=380040&thread=solutions&unit=1574
{-
...
tokenize :: String -> Maybe [Token]
tokenize = traverse asToken . words
-}

--https://stepik.org/lesson/8439/step/4?discussion=634870&thread=solutions&unit=1574
{-
...
tokenize :: String -> Maybe [Token]
tokenize = mapM asToken . words
-}

--https://stepik.org/lesson/8439/step/4?discussion=515251&thread=solutions&unit=1574
{-
...
tokenize :: String -> Maybe [Token]
tokenize = sequence . map (asToken) . words
-}

--https://stepik.org/lesson/8439/step/4?discussion=3557680&thread=solutions&unit=1574
{-
...
tokenize :: String -> Maybe [Token]
tokenize = foldrM f [] . words
  where
    f word tokenList = do
      token <- asToken word
      return $ token : tokenList
-}

-- 5.4.6
{-
Пусть имеется тип данных, который описывает конфигурацию шахматной доски:
data Board = ...
Кроме того, пусть задана функция
nextPositions :: Board -> [Board]
которая получает на вход некоторую конфигурацию доски и возвращает все возможные конфигурации, которые могут получиться, 
если какая-либо фигура сделает один ход. Напишите функцию:
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
которая принимает конфигурацию доски, число ходов n, предикат и возвращает все возможные конфигурации досок, 
которые могут получиться, если фигуры сделают n ходов и которые удовлетворяют заданному предикату. 
При n < 0 функция возвращает пустой список. 
--
--Тип Board и функция nextPositions заданы, реализовывать их не нужно
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN b n pred = do undefined
-}
data Board = Board Int
nextPositions :: Board -> [Board]
nextPositions x = [x]
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN b n pred
  | n < 0 = [] :: [Board]
  | n == 0 = if pred b then [b] else []
  | n == 1 = filter pred (nextPositions b)
  | n > 0 = do
    z <- nextPositions b
    nextPositionsN z (n -1) pred

--testing functions
testf :: Int -> [Int]
testf b = [0,b]

testt :: Int -> Int -> [Int]
testt b n
 |n<0 =[9]
 |n==0 =[b]
 |n==1 =testf b
 |n>0 =do
  z <- testf b
  testt z (n -1)

-- *Main> testt 2 3
-- [0,0,0,0,0,0,0,2]
-- *Main> testt 2 2
-- [0,0,0,2]
-- *Main> testt 2 1
-- [0,2]
-- *Main> testt 2 0
-- [2]

{-
nextPositionsN b n pred   
  |n<0 = [] :: [Board]   
  |n==0 = nextPositions b
  -- |n==1 = filter pred (nextPositions b)
  |n>0 = filter pred (do          
    z<- nextPositionsN b (n-1) pred
    return z)
--Failed. "Haskell: test #2 failed"

nextPositionsN b n pred   
  |n<0 = [] :: [Board]   
  |n==0 = [b]
  |n>0 = filter pred (do          
    z<- nextPositionsN b (n-1) pred
    return z)
--Failed. "Haskell: test #1 failed"

nextPositionsN b n pred   
  |n<0 = [] :: [Board]   
  |n==0 = nextPositions b
  |n>0 = filter pred (nextPositionsN1 b n 0 ) where
      nextPositionsN1 :: Board -> Int -> Int -> [Board]
      nextPositionsN1 b n n1 
       |n<0 = [] :: [Board]  
       |n==0 = nextPositions b
       |n>0 = do
         z<- concat (map nextPositions (nextPositionsN1 b (n-1) (n1+1)))
         return z
--Failed. "Haskell: test #1 failed"

nextPositionsN b n pred   
  |n<0 = [] :: [Board]   
  |n==0 = nextPositions b
  |n==1 = filter pred (nextPositions b)
  |n>1 = filter pred (nextPositionsN1 b n ) where
      nextPositionsN1 :: Board -> Int -> [Board]
      nextPositionsN1 b n  
       |n<0 = [] :: [Board]  
       |n==0 = nextPositions b
       |n>0 = do
         z<- concat (map nextPositions (nextPositionsN1 b (n-1) ))
         return z
--Failed. "Haskell: test #2 failed"

nextPositionsN b n pred   
  |n<0 = [] :: [Board]   
  |n==0 = filter pred (nextPositions b)
  |n>0 =  do
         z<- concat (map nextPositions (nextPositionsN b (n-1) pred))
         return z
--Failed. "Haskell: test #1 failed"

nextPositionsN b n pred   
  |n<0 = [] :: [Board]   
  |n==0 = if pred b then nextPositions b else [] :: [Board]   
  |n>0 =  do
         z<- nextPositions b 
         y<-nextPositionsN z (n -1) pred
         return y
--Failed. "Haskell: test #1 failed"

nextPositionsN b n pred
  |n<0 = [] :: [Board]
  |n==0 = if pred b then nextPositions b else [] :: [Board]-- map pred b
  -- |n==1 = filter pred (nextPositions b)
  |n>0 =  do
         z<- nextPositions b
         nextPositionsN z (n -1) pred
--Failed. "Haskell: test #1 failed"

nextPositionsN b n pred
  | n < 0 = [] :: [Board]
  | n == 0 = nextPositions b
  | n == 1 = filter pred (nextPositions b)
  | n > 0 = do
    z <- nextPositions b
    nextPositionsN z (n -1) pred
--Failed . "Haskell: test #8 failed"

nextPositionsN b n pred
  | n < 0 = [] :: [Board]
  | n == 0 = filter pred (nextPositions b)
  | n == 1 = filter pred (nextPositions b)
  | n > 0 = do
    z <- nextPositions b
    nextPositionsN z (n -1) pred
--Failed . "Haskell: test #8 failed"

nextPositionsN b n pred
  |n<0 = [] :: [Board]
  |n==0 =  filter pred (nextPositions b)
  -- |n==1 = filter pred (nextPositions b)
  |n>0 =  do
         z<- nextPositions b
         nextPositionsN z (n -1) pred
--Failed. "Haskell: test #1 failed"

nextPositionsN b n pred
  | n < 0 = [] :: [Board]
  | n == 0 =[b]
  | n == 1 = filter pred (nextPositions b)
  | n > 0 = do
    z <- nextPositions b
    nextPositionsN z (n -1) pred
--Failed . "Haskell: test #11 failed"

-}



--https://stepik.org/lesson/8439/step/6?discussion=3832212&unit=1574
{-
Можно взять для теста
nextPositions :: [[Int]] -> [[[Int]]]
 nextPositions x = [[(t:xs)] | xs <- x, t <- [1, 2, 3]]
и реализовывать
nextPositionsN :: [[Int]] -> Int -> ([[Int]] -> Bool) -> [[[Int]]]
дальше сами тесты
> nextPositionsN [[1]] (-1) (\x ->  (sum . (fmap sum)) x < 7)
[]
> nextPositionsN [[1]] 0 (\x ->  (sum . (fmap sum)) x < 7)
[[1]]
> nextPositionsN [[1]] 1 (\x ->  (sum . (fmap sum)) x < 7) 
[[[1,1]],[[2,1]],[[3,1]]
> nextPositionsN [[1]] 2 (\x ->  (sum . (fmap sum)) x < 7)
[[[1,1,1]],[[2,1,1]],[[3,1,1]],[[1,2,1]],[[2,2,1]],[[3,2,1]],[[1,3,1]],[[2,3,1]]]
Также, учитывайте, что фильтрацию нужно применять только в конце, а не на промежуточных шагах.
То есть конечная позиция входит в ответ, если предикат для нее выполняется, и не важно, 
как к этой позиции пришли - может, и через "невалидные" позиции.
-}

--https://stepik.org/lesson/8439/step/6?discussion=2962757&unit=1574
{-
пример реализации Board
data Pown = Pown
  deriving (Show, Eq)
data Square = Empty | Square Pown
  deriving (Show, Eq)
data Board = Board [Square]
  deriving (Show, Eq)
-- only forward moves is legal
nextPositions :: Board -> [Board]
nextPositions brd = helper [] brd []
  where
    helper prevBoard (Board []) acc = acc
    helper prevBoard (Board (b : [])) acc = acc
    helper prevBoard (Board (b : bx)) acc = case b of
      Empty -> helper (prevBoard ++ [b]) (Board bx) acc
      Square Pown ->
        if (head bx == Empty)
          then
            let prevBoardNew = prevBoard ++ [b]
                bBx = Board bx
                accNew = ((Board (prevBoard ++ [Empty, b] ++ (tail bx))) : acc)
             in helper prevBoardNew bBx accNew
          else helper (prevBoard ++ [b]) (Board bx) acc
predicat :: Board -> Bool
predicat (Board []) = False
predicat (Board (b : bx)) = case b of
  Square Pown -> True
  Empty -> False
-}

--https://stepik.org/lesson/8439/step/6?discussion=338569&thread=solutions&unit=1574
{-
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN b n pred
  | n < 0 = []
  | otherwise = filter pred $ foldl (>>=) [b] (replicate n nextPositions)
-}

-- https://stepik.org/lesson/8439/step/6?discussion=356009&thread=solutions&unit=1574
{-
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN b n pred
  | n < 0 = []
  | n == 0 = filter pred [b]
  | otherwise =
    do
      x <- nextPositions b
      nextPositionsN x (n - 1) pred
-}

--https://stepik.org/lesson/8439/step/6?discussion=590904&thread=solutions&unit=1574
--Решение в ленивом стиле: 1) Сначала формируем бесконечный список списков конфигураций [[Board]], 
--достижимых за 0, 1, 2, 3 и т.д. ходов с помощью функции iterate 2) 
--Достаем n-ый элемент этого списка, потом фильтруем - и вуаля!
{-
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN b n pred
  | n >= 0 = filter pred $ (iterate (>>= nextPositions) [b]) !! n
  | otherwise = []
-}

--https://stepik.org/lesson/8439/step/6?discussion=830920&thread=solutions&unit=1574
{-
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN _ n _ | n < 0 = []
nextPositionsN b n pred = filter pred . (!! n) . iterate (>>= nextPositions) $ [b]
-}

--https://stepik.org/lesson/8439/step/6?discussion=1073682&thread=solutions&unit=1574
{-
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN board n predicate
  | n < 0 = []
  | otherwise = filter predicate $ foldl (>>=) [board] (replicate n nextPositions)
-}

--https://stepik.org/lesson/8439/step/6?discussion=423049&thread=solutions&unit=1574
{-
import Control.Monad
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN _ n _ | n < 0 = []
nextPositionsN b n pred = filter pred $ (foldl (>=>) (return) $ replicate n $ nextPositions) b
-}

--https://stepik.org/lesson/8439/step/6?discussion=3013091&thread=solutions&unit=157
{-
nextPositionsN :: Board -> Int -> (Board -> Bool) -> [Board]
nextPositionsN b n pred = do
  if n < 0 then [] else [1]
  filter pred (foldr (\init bx -> concatMap nextPositions bx) [b] [n, n -1 .. 1])
-}

--5.4.8
{-
Используя монаду списка и do-нотацию, реализуйте функцию
pythagoreanTriple :: Int -> [(Int, Int, Int)]
которая принимает на вход некоторое число x x x и возвращает список троек (a,b,c) (a, b, c) (a,b,c), таких что
a2+b2=c2,  a>0,  b>0,  c>0,  c≤x,  a<b a^2 + b^2 = c^2, \; a \gt 0, \; b \gt 0, \; c \gt 0, \; c \leq x, \; a \lt b a2+b2=c2,a>0,b>0,c>0,c≤x,a<b  
Число x x x может быть ≤0 \leq 0 ≤0 , на таком входе должен возвращаться пустой список.
GHCi> pythagoreanTriple 5
[(3,4,5)]
GHCi> pythagoreanTriple 0
[]
GHCi> pythagoreanTriple 10
[(3,4,5),(6,8,10)]
--
pythagoreanTriple :: Int -> [(Int, Int, Int)]
pythagoreanTriple x = do undefined
-}

pythagoreanTriple :: Int -> [(Int, Int, Int)]
pythagoreanTriple x = do
  a<-[1..x]
  b<-[1..x]
  c<-[1..x]
  True<- return ((a<b)&&(a^2+b^2==c^2))
  if x <= 0 then [] else [(a,b,c)]

--
{-
На этом шаге вы будете работать с монадой IO, а значит, ваша программа будет взаимодействовать с операционной системой. 
Чтобы тестирующая система смогла оценить вашу программу, пожалуйста, используйте только функции, 
осуществляющие ввод/вывод на терминал: getChar, putChar, putStr, putStrLn, getLine. 
Все эти функции уже будут находиться в области видимости, так что вам не следует их импортировать. 
По той же причине, главная функция вашей программы будет называться не main, а main' (со штрихом).
Напишите программу, которая будет спрашивать имя пользователя, а затем приветствовать его по имени. 
Причем, если пользователь не ввёл имя, программа должна спросить его повторно, и продолжать спрашивать, 
до тех пор, пока пользователь не представится.
Итак, первым делом, программа спрашивает имя:
What is your name?
Name: 
Пользователь вводит имя и программа приветствует его:
What is your name?
Name: Valera
Hi, Valera!
Если же пользователь не ввёл имя, необходимо отобразить точно такое же приглашение ещё раз:
What is your name?
Name: 
What is your name?
Name: 
What is your name?
Name: Valera
Hi, Valera!
Пожалуйста, строго соблюдайте приведенный в примере формат вывода. Особое внимание уделите пробелам и переводам строк! 
Не забудьте про пробел после Name:, а также про перевод строки в самом конце (ожидается, что вы будете использовать 
putStrLn для вывода приветствия пользователя).
--
main' :: IO ()
main' = ?
-}
main' :: IO ()
main' = do
  putStrLn "What is your name?"
  putStr "Name: "
  n <- getLine
  if null n then main' else putStrLn ("Hi, " ++ n ++ "!")

-- https://stepik.org/lesson/8443/step/3?discussion=1143294&thread=solutions&unit=1578
{-
main' :: IO ()
main' = putStrLn "What is your name?" >> putStr "Name: " >> getLine >>= (\name -> if name /= "" then putStrLn ("Hi, " ++ name ++ "!") else main')
-}

--5.5.8
{-
На этом шаге вы будете работать с монадой IO, а значит, ваша программа будет взаимодействовать с операционной системой. 
Чтобы тестирующая система смогла оценить вашу программу, пожалуйста, используйте только функции, работающие с файлами и директориями: 
getDirectoryContents, removeFile. Все эти функции уже будут находиться в области видимости, так что вам не следует их импортировать. 
По той же причине, главная функция вашей программы будет называться не main, а main' (со штрихом).
В этом задании ваша программа должна попросить пользователя ввести любую строку, а затем удалить все файлы в текущей директории, 
в именах которых содержится эта строка, выдавая при этом соответствующие сообщения.
Substring: 
Пользователь вводит любую строку:
Substring: hell
Затем программа удаляет из текущей директории файлы с введенной подстрокой в названии. К примеру, если в текущей директории 
находились файлы thesis.txt, kitten.jpg, hello.world, linux_in_nutshell.pdf, то вывод будет таким:
Substring: hell
Removing file: hello.world
Removing file: linux_in_nutshell.pdf
Если же пользователь ничего не ввёл (просто нажал Enter), следует ничего не удалять и сообщить об этом:
Substring: 
Canceled
Для получения списка файлов в текущей директории используйте функцию getDirectoryContents, передавая ей в качестве аргумента строку, 
состоящую из одной точки  ("."), что означает «текущая директория». 
Для удаления файлов используйте функцию removeFile (считайте, что в текущей директории нет поддиректорий — только простые файлы). 
В выводимых сообщениях удаленные файлы должны быть перечислены в том же порядке, в котором их возвращает функция getDirectoryContents.
Пожалуйста, строго соблюдайте приведенный в примере формат вывода. Особое внимание уделите пробелам и переводам строк! 
Не забудьте про пробел после Substring:, а также про перевод строки в конце (ожидается, что вы будете использовать putStrLn для вывода 
сообщений об удалении).
--
main' :: IO ()
main' = ?
-}
main1' :: IO ()
main1' = do
  putStr "Substring: "
  --hFlush stdout --necessary, but needs import System.IO
  n<-getLine
  if null n then do
  putStrLn "Canceled"  else do
  xi <- getDirectoryContents "./77"--"."
  mapM_ (\f -> if isInfixOf n f  then do
    --removeFile f --necessary
    putStr ("Removing file: " ++ f ++ "\n") else do
      --hFlush stdout --necessary, but needs import System.IO
      putStr "") (reverse xi)



--https://stepik.org/lesson/8443/step/7?discussion=186591&unit=1578
--foldr (+) 0 [1,2,3,4] = (1 + (2 + (3 + (4 + 0))))
--https://stepik.org/lesson/8443/step/7?discussion=186591&reply=190128&unit=1578
{-
putChar 'a' >> (putChar 'b' >> return ())
~>   putChar 'a' >>= \_ -> putChar 'b' >> return () 
~>   \w -> case putChar 'a' w of (w',a) -> (\_ -> putChar 'b' >> return ()) a w'
Сначала раскрывается головной редекс (>>), потом раскрывается  головной редекс (>>=), потом (в рантайме) 
при передаче значения w  магического типа RealWorld срабатывает сопоставление с образцом case, требующее 
вызова  putChar 'a' w, который и выводит 'a'. 
-}

--https://stepik.org/lesson/8443/step/8?discussion=3077179&unit=1578
{-


Объясните, пожалуйста, почему программа
main :: IO ()
main = do
  putStr "Substring: "
  s <- getLine
  putStr s
Сначала запрашивает строку, а потом выводит Substring: Хотя при интерпритации всё норм
--https://stepik.org/lesson/8443/step/8?discussion=3077179&reply=3077781&unit=1578
Из за буферизации вывод по возможности происходит только по достижении очередного символа перевода строки. 
Чтобы неполные строки выводились без задержки, нужно отключить буферизацию (в ghci она уже отключена)
import System.IO
main :: IO ()
main = do
  hSetBuffering stdout NoBuffering
  putStr "Substring: "
  s <- getLine
  putStr s
либо можно сбрасывать буфер вручную после неполного вывода
import System.IO
main :: IO ()
main = do
  putStr "Substring: "
  hFlush stdout
  s <- getLine
  putStr s
-}

--https://stepik.org/lesson/8443/step/8?discussion=831238&thread=solutions&unit=1578
{-
import Data.List
main' :: IO ()
main' = do
  putStr "Substring: "
  l <- getLine
  if null l
    then putStrLn "Canceled"
    else
      getDirectoryContents "."
        >>= return . filter (l `isInfixOf`)
        >>= mapM_ (\x -> putStrLn ("Removing file: " ++ x) >> removeFile x)
-}

--https://stepik.org/lesson/8443/step/3?discussion=120312&reply=120347&unit=1578
{-
Это стандартная буферизация потока вывода. Действительно, в обычном случае поток вывода буферизуется построчно, 
так что без перевода строки ничего на экран не выведется и придется вызвать `hFlush stdout`
-}

--https://stepik.org/lesson/8441/step/2?discussion=124189&reply=124548&unit=1576
{-
У меня вопрос. В выражении: 
m >>= k = \e -> k (m e) e
буквы m e, которые в скобках - это один аргумент функции k, представляющий из себя обёрнутое в ReaderT окружение? 
Или это какое-то m, применённое к e? 
https://stepik.org/lesson/8441/step/2?discussion=124189&reply=124590&unit=1576
Это один аргумент функции `k`, представляющий из себя `m`, примененное к `e`. 
Потому что `m :: (->) e a` (т.е. `e -> a`), а `k :: a -> (->) e b`.
m >>= k = \e -> k (m e) e
m :: (->) e a            --(т.е. `e -> a`)
k :: a -> (->) e b
--
(->) e a >>= a -> (->) e b
\e ->e -> a >>=\e ->k 
-}

-- 5.6.3
{-
Не используя интерпретатор, вычислите значение следующего выражения:
return 2 >>= (+) >>= (*) $ 4
--24
-}

-- 5.6.4
{-
При работе с монадой Reader, каков смысл оператора (>>)?
Этот оператор позволяет передать одно и то же значение (окружение) в качестве аргумента нескольким функциям в цепочке композиций
Этот оператор позволяет изменить окружение
Этот оператор позволяет вычислить произвольную функцию от окружения
+В сочетании с монадой Reader этот оператор бесполезен
-}

-- 5.6.7
{-
В последнем видео мы познакомились с функцией local, позволяющей произвести некоторое вычисление во временно измененном окружении. 
При этом значение, задающее новое окружение, имело тот же тип, что и исходное.
Если попытаться обобщить эту функцию таким образом, чтобы новое окружение потенциально имело другой тип, какая сигнатура будет у 
обобщенной функции local'?
--
local' :: (r' -> r) -> Reader r' a -> Reader r a
+local' :: (r -> e) -> Reader e a -> Reader r a
local' :: (r -> r') -> Reader r' a -> Reader r ()
local' :: (r -> e) -> Reader r a -> Reader e a
local' :: (r -> r') -> Reader r a -> Reader r' () 
-}

--5.6.8
{-
Реализуйте функцию local' из прошлого задания.
Считайте, что монада Reader определена так, как на видео:
data Reader r a = Reader { runReader :: (r -> a) }
instance Monad (Reader r) where
  return x = Reader $ \_ -> x
  m >>= k  = Reader $ \r -> runReader (k (runReader m r)) r
-- 
local' :: (r -> r') -> Reader r' a -> Reader r a
local' f m = ?
-}
--
instance Functor (Reader r) where
  fmap = liftM
instance Applicative (Reader r) where
  pure = return
  (<*>) = ap
--

data Reader r a = Reader {runReader :: (r -> a)}
-- runReader:: Reader r a -> r -> a

instance Monad (Reader r) where
  return x = Reader $ \_ -> x
  m >>= k = Reader $ \r -> runReader (k (runReader m r)) r

local :: (r -> r) -> Reader r a -> Reader r a
local f m = Reader $ \e -> runReader m (f e)

local' :: (r -> r') -> Reader r' a -> Reader r a
--f :: (r -> r')
--m :: Reader r' a
local' f m = Reader $ \e -> runReader m (f e)

--5.6.9
{-
Вспомним пример с базой пользователей и паролей:
type User = String
type Password = String
type UsersTable = [(User, Password)]
Реализуйте функцию, принимающую в качестве окружения UsersTable и возвращающую список пользователей, 
использующих пароль "123456" (в том же порядке, в котором они перечислены в базе).
GHCi> runReader usersWithBadPasswords [("user", "123456"), ("x", "hi"), ("root", "123456")]
["user","root"]
--
usersWithBadPasswords :: Reader UsersTable [User]
usersWithBadPasswords = ?
-}

ask:: Reader r r
ask = Reader id

asks :: (r->a)->Reader r a
asks = Reader

type User = String
type Password = String
type UsersTable = [(User, Password)]
usersWithBadPasswords :: Reader UsersTable [User]
usersWithBadPasswords = do
  e<- ask
  return $ map fst (filter (\x -> (snd x) =="123456") e)

--5.7.3
{-
Функция execWriter запускает вычисление, содержащееся в монаде Writer, и возвращает получившийся лог, 
игнорируя сам результат вычисления. Реализуйте функцию evalWriter, которая, наоборот, игнорирует накопленный лог 
и возвращает только результат вычисления.
--
evalWriter :: Writer w a -> a
evalWriter = ?
-}
evalWriter :: Writer w a -> a
evalWriter m = fst (runWriter m)


--
instance (Monoid w) => Functor (Writer w) where
  fmap = liftM
instance (Monoid w) => Applicative (Writer w) where
  pure = return
  (<*>) = ap
--

newtype Writer w a = Writer {runWriter :: (a,w)}
--runWriter :: Writer w a -> (a, w)
instance (Monoid w) => Monad (Writer w) where
  return x = Writer (x, mempty)
  m>>=k =
    let (x, u) = runWriter m
        (y, v) = runWriter $ k x
   in Writer( y, u `mappend` v)

writer :: (a,w)-> Writer w a
writer = Writer


--5.7.4
{-
Выберите все верные утверждения про монаду Writer.
+В качестве типа лога можно использовать произвольную группу
+В качестве типа лога можно использовать произвольный моноид
Тип результата вычисления и тип лога должны совпадать
Тип результата вычисления и тип лога не могут совпадать
+В качестве типа результата вычисления можно использовать произвольный моноид
В качестве типа лога можно использовать произвольный тип
-}

--5.7.6
{-
Давайте разработаем программное обеспечение для кассовых аппаратов одного исландского магазина. 
Заказчик собирается описывать товары, купленные покупателем, с помощью типа Shopping следующим образом:
type Shopping = Writer (Sum Integer) ()
shopping1 :: Shopping
shopping1 = do
  purchase "Jeans"   19200
  purchase "Water"     180
  purchase "Lettuce"   328
Последовательность приобретенных товаров записывается с помощью do-нотации. 
Для этого используется функция purchase, которую вам предстоит реализовать. Эта функция принимает наименование товара, 
а также его стоимость в исландских кронах (исландскую крону не принято делить на меньшие единицы, потому используется целочисленный 
тип Integer). Кроме того, вы должны реализовать функцию total:
GHCi> total shopping1 
19708
--
purchase :: String -> Integer -> Shopping
purchase item cost = ?

total :: Shopping -> Integer
total = ?
-}
tell :: Monoid w=> w-> Writer w ()
tell w = writer ((),w)

tell1 :: Monoid w => w ->a-> Writer w a
tell1 w s= writer (s, w)

type Shopping = Writer (Sum Integer) ()

shopping1 :: Shopping
shopping1 = do
  purchase "Jeans" 19200
  purchase "Water" 180
  purchase "Lettuce" 328

purchase :: String -> Integer -> Shopping
purchase item cost = tell (Sum cost)

total :: Shopping -> Integer
total sho = do getSum $ snd  $ runWriter sho

--5.7.7
{-
Измените определение типа Shopping и доработайте функцию purchase из предыдущего задания таким образом, 
чтобы можно было реализовать функцию items, возвращающую список купленных товаров (в том же порядке, 
в котором они были перечислены при покупке):
shopping1 :: Shopping
shopping1 = do
  purchase "Jeans"   19200
  purchase "Water"     180
  purchase "Lettuce"   328
GHCi> total shopping1 
19708
GHCi> items shopping1
["Jeans","Water","Lettuce"]
Реализуйте функцию items и исправьте функцию total, чтобы она работала как и прежде.
--
type Shopping = Writer ? ()
purchase :: String -> Integer -> Shopping
purchase item cost = ?
total :: Shopping -> Integer
total = ?
items :: Shopping -> [String]
items = ?
-}
-- type Shopping2 = Writer ((Sum Integer),String) ()

-- shopping2 :: Shopping2
-- shopping2 = do
--   purchase2 "Jeans" 19200
--   purchase2 "Water" 180
--   purchase2 "Lettuce" 328

-- purchase2 :: String -> Integer -> Shopping2
-- purchase2 item cost = tell ((Sum cost), item)

-- total2 :: Shopping2 -> Integer
-- total2 sho = do getSum $ fst $ snd $ runWriter sho

-- items :: Shopping2 -> [String]
-- items = ?

type Shopping2 = Writer (Sum Integer,[String]) ()

writer2 :: (a, w) -> Writer w a
writer2 = Writer

tell2 :: (Monoid w) => w ->  Writer w ()--(Sum Integer, String) () --Shopping2
tell2 w  = Writer ((), w)

--t :: (a, w)
t = runWriter
--t = runWriter (Writer (s, w))
--runWriter :: Writer w a -> (a, w)

shopping2 :: Shopping2
shopping2 = do
  purchase2 "Jeans" 19200
  purchase2 "Water" 180
  purchase2 "Lettuce" 328

purchase2 :: String -> Integer -> Shopping2
purchase2 item cost = Writer ((), ((Sum cost), [item])) --tell2 ((Sum cost), [item])

total2 :: Shopping2 -> Integer
total2 sho = do getSum $ fst $ snd $ runWriter sho --19708

items :: Shopping2 -> [String]
items sho = do snd $ snd $ runWriter sho --["Jeans","Water","Lettuce"]

--5.8.3 
{-
Выберите все верные утверждения про монаду State:
Монада State является частным случаем монады Writer
+Монада Reader является частным случаем монады State
+Монада Writer является частным случаем монады State
-}

--5.8.4
{-
Где реализована монада State?
+Монада State реализована в одном из пакетов Haskell Platform
Монада State встроена в компилятор GHC, поскольку позволяет осуществлять вычисления с изменяемым состоянием, что невозможно в «чистом» Хаскеле 
-}

--
instance  Functor (State s) where
  fmap = liftM

instance  Applicative (State s) where
  pure = return
  (<*>) = ap

--

newtype State s a = State {runState ::s -> (a,s)}
instance Monad (State s) where
  return a = State $ \st -> (a,st)
  m>>=k = State $ \st ->
    let (a, st')= runState m st
        m' = k a
    in runState m' st'

-- runState ::State s a -> s -> (a,s)

execState :: State s a -> s -> s
execState m s = snd (runState m s)

evalState :: State s a -> s -> a
evalState m s = fst (runState m s)

get:: State s s
get = State $ \st->(st,st)

put :: s -> State s ()
put st = State $ \_ -> ((), st)

tick :: State Int Int
tick = do
  n<- get
  put (n+1)
  return n

--5.8.6
{-
Давайте убедимся, что с помощью монады State можно эмулировать монаду Reader.
Напишите функцию readerToState, «поднимающую» вычисление из монады Reader в монаду State:
GHCi> evalState (readerToState $ asks (+2)) 4
6
GHCi> runState (readerToState $ asks (+2)) 4
(6,4) 
--
readerToState :: Reader r a -> State r a
readerToState m = ?
-}
-- runReader:: Reader r a -> r -> a
-- runState ::State s a -> s -> (a,s)
readerToState :: Reader r a -> State r a
readerToState (Reader a) = State (\st -> (runReader (Reader a) st, st))
--readerToState (Reader a) =  State (runReader (Reader a) )
-- • Occurs check: cannot construct the infinite type: a ~ (a, r)
  --Expected type: r -> (a, r)
    --Actual type: r -> a

-- readerToState (Reader r) = do
--  x<-Reader r   -- ::a  -- ::State r a
--  y<-runReader (Reader r) r
--  State r y  
  --State (runReader (Reader r) r, Reader r) 
 --Reader r::Reader r a
 --r::r->a
 --(runReader (Reader r) r)::a
 --(runReader (Reader r) st)::a
  --State (\r -> (runReader (Reader r) (runReader (Reader r ) st), st))
  --State (\st -> (runReader (Reader r) (runReader (Reader r ) st), st))
  --State (\st ->(runReader (Reader r) r,st ))
 --put (State r (runReader (Reader r) r))
 --State r (runReader (Reader r) r)
 --let a = (runReader (Reader r) r) in (State r ww)
  --put (runReader (Reader r) r)
  --State (runReader (Reader r))
  --let k = (runReader (Reader r) r a) in State r 

--5.8.7
{-
Теперь убедимся, что с помощью монады State можно эмулировать монаду Writer.
Напишите функцию writerToState, «поднимающую» вычисление из монады Writer в монаду State:
GHCi> runState (writerToState $ tell "world") "hello,"
((),"hello,world")
GHCi> runState (writerToState $ tell "world") mempty
((),"world")
Обратите внимание на то, что при работе с монадой Writer предполагается, 
что изначально лог пуст (точнее, что в нём лежит нейтральный элемент моноида), 
поскольку интерфейс монады просто не позволяет задать стартовое значение. 
Монада State же начальное состояние (оно же стартовое значение в логе) задать позволяет.
--
writerToState :: Monoid w => Writer w a -> State w a
writerToState m = ?
-}

--runWriter :: Writer w a -> (a, w)
-- runState ::State s a -> s -> (a,s)
writerToState :: Monoid w => Writer w a -> State w a
writerToState (Writer a) =
  ---put (snd (runWriter (Writer a))) ---+++
  State (\w -> (x,w`mappend` y)) where
    (x,y)=runWriter (Writer a)
  --writerToState (Writer a) = State (\w -> runWriter (Writer a))
--   *Main> runState (writerToState $ tell "world") "hello," -- ((),"world")
   --b = (snd (runWriter (Writer a))) -- ::w
   --a::(a,w)
  --State (\w -> a (w `mappend` a))
  --Occurs check: cannot construct the infinite type: w ~ (a, w)  Expected type: State w a    Actual type: State (a, w) a
  --State (\w -> (a,w `mappend` a))
  --Occurs check: cannot construct the infinite type: w ~ (a, w)  Expected type: State w a    Actual type: State (a, w) (a, w)
--writerToState (Writer a) = State (\w -> a `mappend` w) a
--Couldn't match expected type ‘(a, w) -> State w a’ with actual type ‘State (a, w) a’
--writerToState (Writer a) = State (a,(\w -> a `mappend` w)) 
--Couldn't match expected type ‘w -> (a, w)’ with actual type ‘((a, w), (a, w) -> (a, w))’
--writerToState (Writer a) = State (a `mappend` w) 
-- Couldn't match expected type ‘w -> (a, w)’  with actual type ‘(a, w)’
---writerToState (Writer a) = State (\w -> a `mappend` w) 
--cannot construct the infinite type: w ~ (a, w)  Expected type: (a, (a, w))    Actual type: (a, w)
--writerToState (Writer a) = State (\w -> a) --Failed. "Haskell: test #2 failed"where
  --(a, w) = State a
  --b = runWriter (Writer a) -- :: (a,w)
  --State a -- Couldn't match expected type ‘w -> (a, w)’ with actual type ‘(a, w)’


--tick без использования do-нотации
--tick = get >>= \n -> put (n+1) >> return n


  --5.8.9
  {-
Если бы мы хотели вычислить n n n-е число Фибоначчи на императивном языке программирования, мы бы делали это с помощью 
двух переменных и цикла, обновляющего эти переменные:
def fib(n):
  a, b = 0, 1
  for i in [1 .. n]:
    a, b = b, a + b
  return a
С точки зрения Хаскеля, такую конструкцию удобно представлять себе как вычисление с состоянием. Состояние в данном случае — 
это два целочисленных значения.
Императивный алгоритм действует очень просто: он совершает n n n шагов, каждый из которых некоторым образом изменяет текущее состояние. 
Первым делом, реализуйте функцию fibStep, изменяющую состояние таким же образом, как и один шаг цикла в императивном алгоритме:
GHCi> execState fibStep (0,1)
(1,1)
GHCi> execState fibStep (1,1)
(1,2)
GHCi> execState fibStep (1,2)
(2,3)
После этого останется лишь применить этот шаг n  раз к правильному стартовому состоянию и выдать ответ. 
Реализуйте вспомогательную функцию execStateN, которая принимает число шагов n, вычисление с состоянием и 
начальное состояние, запускает вычисление n раз и выдает получившееся состояние (игнорируя сами результаты вычислений). 
Применяя эту функцию к fibStep, мы сможем вычислять числа Фибоначчи:
fib :: Int -> Integer
fib n = fst $ execStateN n fibStep (0, 1)
--
fibStep :: State (Integer, Integer) ()
fibStep = ?
execStateN :: Int -> State s a -> s -> s
execStateN n m = ?
  -}

fibStep :: State (Integer, Integer) ()
fibStep = do
  (x,y)<- get
  put (y,x+y)
  return ()

execStateN :: Int -> State s a -> s -> s
execStateN n m = execState (sequence $ replicate n m )

fib :: Int -> Integer
fib n = fst $ execStateN n fibStep (0, 1)

--5.8.10
{-
Некоторое время назад мы определили тип двоичных деревьев, содержащих значения в узлах:
data Tree a = Leaf a | Fork (Tree a) a (Tree a)
В этой задаче вам дано значение типа Tree (), иными словами, вам задана форма дерева. 
Требуется пронумеровать вершины дерева данной формы, обойдя их in-order (то есть, сначала обходим левое поддерево, 
затем текущую вершину, затем правое поддерево):
GHCi> numberTree (Leaf ())
Leaf 1
GHCi> numberTree (Fork (Leaf ()) () (Leaf ()))
Fork (Leaf 1) 2 (Leaf 3)
-}
data Tree a = Leaf a | Fork (Tree a) a (Tree a) deriving Show
numberTree :: Tree () -> Tree Integer
numberTree t = fst (helper2 t 1)
helper2 :: Tree () -> Integer -> (Tree Integer,Integer)
helper2 (Leaf ()) n = (Leaf n, n+1)
helper2 (Fork ta a tb) n = (Fork x1 y1 x2, y2) where
 (x1,y1)=helper2 ta n
 (x2, y2) = helper2 tb (y1 + 1)

-- data Tree a = Leaf a | Fork (Tree a) a (Tree a) deriving Show
-- numberTree :: Tree () -> Tree Integer
-- numberTree t = fst (helper2 t 1) 
-- helper2 :: Tree () -> Integer -> (Tree Integer,Integer)
-- helper2 (Leaf ()) n = (Leaf n, n+1)
-- helper2 (Fork ta a tb) n = ((Fork x1 (y1 + 1) x2),y2) where
--  (x1,y1)=helper2 ta n
--  (x2, y2) = helper2 tb (n+1)
-- *Main> numberTree (Fork (Fork (Leaf ()) () (Leaf ())) () (Leaf ()))
-- Fork (Fork (Leaf 1) 3 (Leaf 2)) 4 (Leaf 2)

-- data Tree a = Leaf a | Fork (Tree a) a (Tree a) deriving (Show)
-- numberTree :: Tree () -> Tree Integer
-- numberTree t = helper2 t 1
-- helper2 :: Tree () -> Integer -> Tree Integer
-- helper2 (Leaf ()) n = (Leaf n)
-- helper2 (Fork ta a tb) n = (Fork (helper2 ta n) (n + 1) (helper2 tb (n + 2)))
-- * Main> numberTree (Fork (Fork (Leaf ()) () (Leaf ())) () (Leaf ()))
-- Fork (Fork (Leaf 1) 2 (Leaf 3)) 2 (Leaf 3)

-- data Tree a = Leaf a | Fork (Tree a) a (Tree a)
-- numberTree :: Tree () -> Tree Integer
-- numberTree t =  do 
--   put 1
--   helper1 (runState ) t
-- helper1 :: State s Integer -> Tree () -> Tree Integer
-- helper1 (State n) (Leaf ()) = do
--   x<-get
--   put (n+1)
--   return x
--   Leaf (execState x)
-- helper1 (State n) (Fork ta a tb) = do (Fork (helper1 ta) n (helper1 tb))

---https://stepik.org/lesson/8444/step/10?discussion=434736&thread=solutions&unit=1579
{-
numberTree :: Tree () -> Tree Integer
numberTree tree = evalState (number tree) 1
  where
    number :: Tree () -> State Integer (Tree Integer)
    number (Leaf ()) = get >>= \i -> modify (+ 1) >> return (Leaf i)
    number (Fork l () r) = do
      la <- number l
      i <- get
      modify (+ 1)
      ra <- number r
      return $ Fork la i ra
-}

--https://stepik.org/lesson/8444/step/10?discussion=1272047&thread=solutions&unit=1579
{-
import Control.Applicative
import Data.Foldable
import Data.Traversable

instance Functor Tree where
  fmap = fmapDefault

instance Foldable Tree where
  foldMap = foldMapDefault

instance Traversable Tree where
  traverse g (Leaf x) = Leaf <$> g x
  traverse g (Fork l x r) = Fork <$> traverse g l <*> g x <*> traverse g r

numberTree :: Tree () -> Tree Integer
numberTree tree = evalState (traverse (const $ modify succ >> get) tree) 0
-}

--https://stepik.org/lesson/8444/step/10?discussion=381384&thread=solutions&unit=1579
{-
tick :: State Integer Integer
tick = do
  n <- get
  put (n + 1)
  return n

numberTree :: Tree () -> Tree Integer
numberTree tree = evalState (numberTree' tree) 1

numberTree' (Leaf _) = tick >>= return . Leaf
numberTree' (Fork l _ r) = do
  l' <- numberTree' l
  t <- tick
  r' <- numberTree' r
  return $ Fork l' t r'
-}

--https://stepik.org/lesson/8444/step/10?discussion=686742&thread=solutions&unit=1579
{-
numberTree :: Tree () -> Tree Integer
numberTree tree = evalState (helper tree) 0
  where
    helper (Leaf _) = liftM Leaf step
    helper (Fork left _ right) = liftM3 Fork (helper left) step (helper right)
    step = modify succ >> get
-}

--https://stepik.org/lesson/8444/step/10?discussion=351189&thread=solutions&unit=1579

{-
numberTree :: Tree () -> Tree Integer
numberTree tree = evalState (indepth tree) 1
  where
    indepth (Leaf _) = do
      x <- get
      put (x + 1)
      return (Leaf x)
    indepth (Fork l _ r) = do
      ll <- indepth l
      x <- get
      put (x + 1)
      rr <- indepth r
      return (Fork ll x rr)
-}

--https://stepik.org/lesson/8444/step/10?discussion=1696126&thread=solutions&unit=1579
{-
numberTree :: Tree () -> Tree Integer
numberTree tree = evalState (node tree) 1

node :: Tree () -> State Integer (Tree Integer)
node (Leaf _) = do
  n <- get
  put (n + 1)
  return $ Leaf n
node (Fork l _ r) = do
  l' <- node l
  n <- get
  put (n + 1)
  r' <- node r
  return $ Fork l' n r'
-}

---https://stepik.org/lesson/8444/step/10?discussion=1207957&thread=solutions&unit=1579
{-
numberTree :: Tree () -> Tree Integer
numberTree tree = (evalState (tickTree tree)) 1

tickTree :: Tree () -> State Integer (Tree Integer)
tickTree (Leaf _) = do
  n <- get
  modify (+ 1)
  return (Leaf n)
tickTree (Fork left root right) = do
  left' <- tickTree left
  n <- get
  modify (+ 1)
  let root' = n
  right' <- tickTree right
  return (Fork left' root' right')
-}

--https://stepik.org/lesson/8444/step/10?discussion=1073588&thread=solutions&unit=1579
{-
numberTree :: Tree () -> Tree Integer
numberTree tree = evalState (helper tree) 1
  where
    helper :: Tree () -> State Integer (Tree Integer)
    helper (Leaf ()) = do
      st <- get
      _ <- modify (+ 1)
      return $ Leaf st
    helper (Fork l _ r) = do
      newL <- helper l
      st <- get
      _ <- modify (+ 1)
      newR <- helper r
      return $ Fork newL st newR
-}

--https://stepik.org/lesson/8444/step/10?discussion=557034&thread=solutions&unit=1579
{-
numberTree' (Leaf _) = do
  n <- get
  modify ((+ 1))
  return (Leaf n)
numberTree' (Fork a _ b) = do
  l <- numberTree' a
  n <- get
  modify ((+ 1))
  r <- numberTree' b
  return (Fork l n r)
numberTree tree = fst (runState (numberTree' tree) 1)
-}