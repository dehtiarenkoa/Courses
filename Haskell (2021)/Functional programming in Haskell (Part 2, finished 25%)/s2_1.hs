import Data.Foldable (Foldable)
import Distribution.Simple.Utils (xargs)
--2.1
{-
Сделайте тип
data Triple a = Tr a a a  deriving (Eq,Show)
представителем класса типов Foldable:
GHCi> foldr (++) "!!" (Tr "ab" "cd" "efg")
"abcdefg!!"
GHCi> foldl (++) "!!" (Tr "ab" "cd" "efg")
"!!abcdefg"
-}

data Triple a = Tr a a a deriving (Eq, Show)
instance Foldable Triple where
    --foldr :: (a -> b -> b) -> b -> Triple a -> b
    foldr f bb (Tr a b c) = a `f` (b `f` (c `f` bb))
    --foldl :: (b -> a -> b) -> b -> Triple a -> b
    foldl f bb (Tr a b c) =   ((bb `f` a )`f`b) `f` c


--2.2
{-
Для реализации свертки двоичных деревьев нужно выбрать алгоритм обхода узлов дерева (см., например, http://en.wikipedia.org/wiki/Tree_traversal).
Сделайте двоичное дерево
data Tree a = Nil | Branch (Tree a) a (Tree a)   deriving (Eq, Show)
представителем класса типов Foldable, реализовав симметричную стратегию (in-order traversal). 
Реализуйте также три другие стандартные стратегии (pre-order traversal, post-order traversal и level-order traversal), сделав типы-обертки
newtype Preorder a   = PreO   (Tree a)    deriving (Eq, Show)
newtype Postorder a  = PostO  (Tree a)    deriving (Eq, Show)
newtype Levelorder a = LevelO (Tree a)    deriving (Eq, Show)
представителями класса Foldable.
GHCi> tree = Branch (Branch Nil 1 (Branch Nil 2 Nil)) 3 (Branch Nil 4 Nil)
GHCi> foldr (:) [] tree
[1,2,3,4]
GHCi> foldr (:) [] $ PreO tree
[3,1,2,4]
GHCi> foldr (:) [] $ PostO tree
[2,1,4,3]
GHCi> foldr (:) [] $ LevelO tree
[3,1,4,2]
--

-}

data Tree a = Nil | Branch (Tree a) a (Tree a) deriving (Eq, Show)
instance Foldable Tree where
  --foldr :: (a -> b -> b) -> b -> Tree a -> b
  foldr f b Nil = b
  foldr f b (Branch left x right) = foldr f (f x (foldr f b right))  left

newtype Preorder a = PreO (Tree a) deriving (Eq, Show)
instance Foldable Preorder where
  foldr f b (PreO Nil) = b
  foldr f b (PreO (Branch left x right)) = f x (foldr f (foldr f b (PreO right)) (PreO left))

newtype Postorder a = PostO (Tree a) deriving (Eq, Show)
instance Foldable Postorder where
  foldr f b (PostO Nil) = b
  foldr f b (PostO (Branch left x right)) = foldr f (foldr f (f x b) (PostO right)) (PostO left)
    --foldr f (foldr f (f x b) left) right  --foldr f (f x (foldr f b left)) right

--https://doisinkidney.com/posts/2018-12-18-traversing-graphs.html
{-
bfe :: Tree a -> [a]
bfe r = f r b []   where
    f (Node x xs) fw bw = x : fw (xs : bw)
    b [] = []
    b qs = foldl (foldr f) b qs []
-}
--https://doisinkidney.com/posts/2018-06-03-breadth-first-traversals-in-too-much-detail.html
{-
data Tree a = Node
  { rootLabel :: a,
    subForest :: [Tree a]
  }
breadthFirst :: Applicative f => (a -> f b) -> Tree a -> f (Tree b)
-}

-- data Treee a = Noode 
--   { rootLabel :: a,
--     subForest :: [Treee a]
--   } deriving (Eq, Show)

-- trr:: Tree a-> Treee a
-- trr (Branch left a right) = Noode a [trr left, trr right]
-- --trr Nil = Noode []


--https://stepik.org/lesson/30427/step/6?discussion=371248&reply=371425&unit=11044
{-
Moжно не строить промежуточный список, для level-order достаточно заметить, что мы можем разделить дерево на голову значение в узле и список хвостов (поддеревьев). 
Далее просто рекурсивно сворачиваем получившийся список, получается весьма компактно и с хорошими вычислительными свойствами.
-}
newtype Levelorder a = LevelO (Tree a) deriving (Eq, Show)
instance Foldable Levelorder where
 --foldr :: (a -> b -> b) -> b -> Levelorder a -> b
 foldr f b (LevelO Nil) = b
 foldr f b x =  foldr f b (fuu ([], [x])) 
 --foldr f b (LevelO (Branch left x right)) = f x (foldr f (foldr f b (LevelO right)) (LevelO left))
--  foldr f b (LevelO (Branch left x right)) = f x ( (fst d) ++ (snd d)) where --foldr (f x ((fst d) ++ (snd d))) where
--    d = fun (LevelO left : [LevelO right])
--    fun :: [Levelorder a ] ->([a],[Levelorder a ])
--    fun [LevelO Nil] = ([],[])   
--    fun [LevelO (Branch left1 x right1)] = ([x], [LevelO left1 , LevelO right1])
--    fun ((LevelO (Branch left1 x1 right1)) : xs) = (x1 : fst (fun xs), LevelO left1 : LevelO right1:snd(fun xs))

heads31 :: Levelorder a -> ([a], [Levelorder a])
heads31 (LevelO Nil) = ([], [])
heads31 (LevelO (Branch Nil y Nil)) = ([y], [])
heads31 (LevelO (Branch l y Nil)) = ([y], [LevelO l])
heads31 (LevelO (Branch Nil y r)) = ([y], [LevelO r])
heads31 (LevelO (Branch l y r)) = ([y], [LevelO l] ++ [LevelO r])

-- * Main> heads3 (LevelO testTree)

-- ([1],[LevelO (Branch (Branch Nil 4 Nil) 2 (Branch (Branch Nil 7 Nil) 5 (Branch Nil 8 Nil))),LevelO (Branch Nil 3 (Branch (Branch Nil 9 Nil) 6 Nil))])

fuu :: ([a], [Levelorder a]) -> [a]
fuu ([], []) = []
fuu ([], [LevelO Nil]) = []
fuu (x, [LevelO Nil]) = x
fuu (x, []) = x
fuu (x, [LevelO (Branch Nil y Nil)]) = x ++ [y]
fuu (x, [LevelO (Branch l y Nil)]) = x ++ [y]
fuu (x, y) = x ++ concatMap fst (map heads31 y) ++ fuu ([], concatMap snd (map heads31 y))

--

fun :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
fun (x, []) = (x, [])
fun (x, [LevelO Nil]) = (x, [])
fun (x, [LevelO (Branch Nil y right)]) = fun (x ++ [y], [LevelO right])
fun (x, [LevelO (Branch left y Nil)]) = fun (x ++ [y], [LevelO left])
fun (x, [LevelO (Branch left y right)]) = fun (x ++ [y], [LevelO left, LevelO right])
fun (x, y : ys) = fun (fst d ++ fst f, snd d ++ snd f)
  where
    d = fun (x, [y])
    f = fun ([], ys)

heads :: [Levelorder a] ->[[a]]
heads [] = [[]]
heads [LevelO Nil] = [[]]
heads [LevelO (Branch Nil y Nil)] = [[y]]
heads [LevelO (Branch Nil y (Branch l z r))] = [[z]] ++ heads [LevelO l] ++ heads [LevelO r]
heads [LevelO (Branch (Branch l z r) y Nil)] = [[z]] ++ heads [LevelO l] ++ heads [LevelO r]
heads [LevelO (Branch (Branch l1 z r1) y (Branch l2 z2 r2))] = [[z, z2]] ++ heads [LevelO l1] ++ heads [LevelO r1] ++ heads [LevelO l2] ++ heads [LevelO r2]
heads (y : ys) = heads [y] ++ heads ys
-- *Main> heads [LevelO testTree]
-- [[2,3],[4],[7,8],[],[],[],[],[],[9],[],[]]

heads1 :: [Levelorder a] -> [a]
heads1 [] = []
heads1 [LevelO Nil] = []
heads1 [LevelO (Branch Nil y Nil)] = [y]
heads1 [LevelO (Branch Nil y (Branch l z r))] = [z]
heads1 [LevelO (Branch (Branch l z r) y Nil)] = [z]
heads1 [LevelO (Branch (Branch l z r) y (Branch l2 z2 r2))] = [z, z2]
heads1 (y : ys) = heads1 [y] ++ heads1 ys
-- *Main> heads1 [LevelO testTree]
-- [2,3]

heads2 :: Levelorder a -> [[a]]
heads2 (LevelO Nil) = [[]]
heads2 (LevelO (Branch Nil y Nil)) = [[y]]
heads2 (LevelO (Branch Nil y (Branch l z r))) = [y] : (heads2 (LevelO l)++ heads2 (LevelO r))
heads2 (LevelO (Branch (Branch l z r) y Nil)) = [y] : (heads2 (LevelO l) ++ heads2 (LevelO r))
heads2 (LevelO (Branch (Branch l z r) y (Branch l1 z1 r1))) = [y] : (heads2 (LevelO l) ++ heads2 (LevelO r) ++ heads2 (LevelO l1) ++ heads2 (LevelO r1))
-- *Main> heads2 (LevelO testTree)
-- [[1],[4],[5],[],[],[],[],[],[6],[],[]]


heads3 :: Levelorder a -> [[a]]
heads3 (LevelO Nil) = [[]]
heads3 (LevelO (Branch Nil y Nil)) = [[y]]
heads3 (LevelO (Branch Nil y r)) = [y] :  heads3 (LevelO r)
heads3 (LevelO (Branch l y Nil)) = [y] : heads3 (LevelO l)
heads3 (LevelO (Branch l y r)) = [y] : (heads3 (LevelO l) ++ heads3 (LevelO r))
-- *Main> heads3 (LevelO testTree)
-- [[1],[2],[4],[5],[7],[8],[3],[6],[9]]

heads4 :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
heads4 (x, [LevelO Nil]) = (x, [])
heads4 (x, [LevelO (Branch Nil y Nil)]) = (x ++ [y], [])
heads4 (x, [LevelO (Branch l y Nil)]) = (x ++ [y], [LevelO l])
heads4 (x, [LevelO (Branch Nil y r)]) = (x ++ [y], [LevelO r])
heads4 (x, [LevelO (Branch l y r)]) = (x ++ [y], [LevelO l] ++ [LevelO r])
-- * Main> heads4 ([],[LevelO testTree])
-- ([1],[LevelO (Branch (Branch Nil 4 Nil) 2 (Branch (Branch Nil 7 Nil) 5 (Branch Nil 8 Nil))),LevelO (Branch Nil 3 (Branch (Branch Nil 9 Nil) 6 Nil))])

-- *Main> fuu ([],[LevelO testTree])
-- [1,2,3,4,5,6,7,8,9]
-- *Main> fuu ([],[LevelO testTree2])
-- [1,2,3,4,5,6,7,8,9]
-- *Main> fuu ([],[LevelO tree])
-- [3,1,4,2]

--foldr :: forall (t :: * -> *) a b.Foldable t =>(a -> b -> b) -> b -> t a -> b
--foldr f z [x1, x2, ..., xn] == x1 \`f\` (x2 \`f\` ... (xn \`f\` z)...)

--foldl :: forall (t :: * -> *) b a.Foldable t =>(b -> a -> b) -> b -> t a -> b
--foldl f z [x1, x2, ..., xn] == (...((z \`f\` x1) \`f\` x2) \`f\`...) \`f\` xn

--tt = foldl heads4 ([], [])  ([], [LevelO testTree])

--ma = map heads4 ( (heads4 ([], [LevelO testTree])))

heads5 :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
heads5 (x, [LevelO Nil]) = (x, [])
heads5 (x, [LevelO (Branch Nil y Nil)]) = (x ++ [y], [])
heads5 (x, [LevelO (Branch l y Nil)]) = (x ++ [y] ++ fst (heads5 ([], [LevelO l])), snd (heads5 ([], [LevelO l])))
heads5 (x, [LevelO (Branch Nil y r)]) = (x ++ [y] ++ fst (heads5 ([], [LevelO r])), snd (heads5 ([], [LevelO r])))
heads5 (x, [LevelO (Branch l y r)]) = (x ++ [y] ++ fst (heads5 ([], [LevelO l])) ++ fst (heads5 ([], [LevelO r])), snd (heads5 ([], [LevelO l])) ++ snd (heads5 ([], [LevelO r])))
-- *Main> heads5 ([],[LevelO testTree])
-- ([1,2,4,5,7,8,3,6,9],[])

heads6 :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
heads6 ([], []) = ([], [])
heads6 (x, []) = (x, [])
heads6 (x, [LevelO Nil]) = (x, [])
heads6 (x, [LevelO (Branch Nil y Nil)]) = (x ++ [y], [])
heads6 (x, [LevelO (Branch l y Nil)]) = heads6 (x ++ [y], [LevelO l])
heads6 (x, [LevelO (Branch Nil y r)]) = heads6 (x ++ [y], [LevelO r])
heads6 (x, [LevelO (Branch l y r)]) = heads6 (x ++ [y], [LevelO l] ++ [LevelO r])
heads6 (x, (y : ys)) = heads6 (x ++ fst (heads6 ([], [y])) ++ fst (heads6 ([],ys)),snd (heads6 ([], [y]))++snd (heads6 ([], ys)))
-- *Main> heads6 ([],[LevelO testTree])
-- ([1,2,4,5,7,8,3,6,9],[])


-- re::[Levelorder a]->[[a]]
-- re [] = [[]]
-- re [LevelO Nil] =  [[]]
-- re [LevelO (Branch Nil y right)] = [[y],concat ( re [LevelO right])]
-- re [LevelO (Branch left y Nil)] = [[y], concat (re [LevelO left])]
-- re [LevelO (Branch left y right)] = [[y], concat (re [LevelO left]) ++ concat (re [LevelO right])]
-- *Main> re [LevelO testTree]
-- [[1],[2,4,5,7,8,3,6,9]]

re :: [Levelorder a] -> [[a]]
re [] = [[]]
re [LevelO Nil] = [[]]
re [LevelO (Branch Nil y right)] = [[y], concat (re [LevelO right])]
re [LevelO (Branch left y Nil)] = [[y], concat (re [LevelO left])]
re [LevelO (Branch left y right)] = [[y], concat (re [LevelO left])++ concat (re [LevelO right])]



-- fun :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
-- fun (x, []) = (x, [])
-- fun (x, [LevelO Nil]) = (x, [])
-- fun (x, [LevelO (Branch Nil y right)]) = fun (x ++ [y], [LevelO right])
-- fun (x, [LevelO (Branch left y Nil)]) =  fun (x ++ [y], [LevelO left])
-- fun (x, [LevelO (Branch left y right)]) = fun (x ++ [y], [LevelO left, LevelO right])
-- fun (x, y : ys) = fun (fst d++fst f,snd d++ snd f )
--   where
--     d = fun (x, [y])
--     f = fun ([],ys)
-- *Main> fun ([],[LevelO testTree2])
-- ([1,2,4,7,8,5,3,6,9],[])

-- fun :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
-- fun (x, []) = (x, [])
-- fun (x, [LevelO Nil]) = (x, [])
-- fun (x, [LevelO (Branch Nil y right)]) = fun (x ++ [y], [LevelO right])
-- fun (x, [LevelO (Branch left y Nil)]) = fun (x ++ [y], [LevelO left])
-- fun (x, [LevelO (Branch left y right)]) = fun (x ++ [y], [LevelO left, LevelO right])
-- fun (x, y : ys) = fun (fst f, snd d ++ snd f)
--   where
--     d = fun (x, [y])
--     f = fun (fst d, ys) --f = fun (fst d, ys++snd d)
-- *Main> fun ([],[LevelO tree])
-- ([3,1,2,4],[])
-- *Main> fun ([],[LevelO testTree])
-- ([1,2,4,5,7,8,3,6,9],[])

-- fun :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
-- fun (x, []) = (x, [])
-- fun (x, [LevelO Nil]) = (x, [])
-- fun (x, [LevelO (Branch Nil y right)]) = fun (x ++ [y], [LevelO right])
-- fun (x, [LevelO (Branch left y Nil)]) = fun (x ++ [y], [LevelO left])
-- fun (x, [LevelO (Branch left y right)]) = fun (x ++ [y], [LevelO left, LevelO right])
-- fun (x, y : ys) = fun (x++fst d++fst f,  snd d++snd f) where
--   d= fun (x,[y])
--   f = fun (x, ys)
-- * Main> fun ([],[LevelO tree])
-- ([3,3,1,2,3,4],[])


-- fun :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
-- fun (x, []) = (x, [])
-- fun (x, [LevelO Nil]) = (x, [])
-- fun (x, [LevelO (Branch Nil y right)]) = fun (x ++ [y], [LevelO right])
-- fun (x, [LevelO (Branch left y Nil)]) = fun (x ++ [y], [LevelO left])
-- fun (x, [LevelO (Branch left y right)]) = fun (x ++ [y], [LevelO left, LevelO right]) 
-- fun (x, (LevelO (Branch left y right)) : ys) = (fst (fun (x ++ [y], ys)), LevelO left : LevelO right : snd (fun ([], ys)))
-- * Main> fun ([],[LevelO tree])
-- ([3,1,4],[LevelO Nil,LevelO (Branch Nil 2 Nil)])


-- fun :: ([a], [Levelorder a]) -> ([a], [Levelorder a])
-- fun (x, []) = (x, [])
-- fun (x, [LevelO Nil]) = (x, [])
-- fun (x, [LevelO (Branch Nil y right)]) = (fst (fun (x ++ [y], [LevelO right])), snd (fun (x ++ [y], [LevelO right])))
-- fun (x, [LevelO (Branch left y Nil)]) = (fst (fun (x ++ [y], [LevelO left])), snd (fun (x ++ [y], [LevelO left])))
-- fun (x, [LevelO (Branch left y right)]) = ((fst (fun (x ++ [y], [LevelO left, LevelO right]))), snd (fun (x ++ [y], [LevelO left, LevelO right])))
-- fun (x, (LevelO (Branch left y right)) : ys) = (x ++ [y] ++ fst (fun ([], ys)), LevelO left : LevelO right : snd (fun ([], ys)))
-- *Main> fun ([],[LevelO tree])
-- ([3,1,4],[LevelO Nil,LevelO (Branch Nil 2 Nil)])

-- fun ::  ([a], [Levelorder a]) ->([a], [Levelorder a])
-- fun (x,[]) = ( x,[])
-- fun (x, [LevelO Nil]) = (x, [])
-- fun (x, [LevelO (Branch left y right)]) = (x ++ [y], [LevelO left, LevelO right])
-- fun (x, (LevelO (Branch left y right)) : ys) = (x ++ [y] ++ fst (fun ([],ys)), LevelO left:LevelO right:snd (fun ([],ys)))
-- *Main> fun ([],[LevelO tree])
-- ([3],[LevelO (Branch Nil 1 (Branch Nil 2 Nil)),LevelO (Branch Nil 4 Nil)])

--    fun [LevelO Nil] = ([],[])
--    fun [LevelO (Branch left1 x right1)] = ([x], [LevelO left1 , LevelO right1])
--    fun ((LevelO (Branch left1 x1 right1)) : xs) = (x1 : fst (fun xs), LevelO left1 : LevelO right1:snd(fun xs))

  --  fun :: [Levelorder a] -> ([a], [Levelorder a])
  --  fun [LevelO Nil] = ([], [])
  --  fun [LevelO (Branch left1 x right1)] = ([x], [LevelO left1, LevelO right1])
  --  --fun ((LevelO (Branch left1 x1 right1)) : (LevelO (Branch left2 x2 right2))) = ([x1, x2], [fun xs])
  --  fun ((LevelO (Branch left1 x1 right1)) : xs) = (x1 : fst (fun xs), LevelO left1 : LevelO right1 : snd (fun xs))


   --fun ((LevelO left) :[(LevelO right)]) b = foldr f (foldr f b right) left -- where
    --  (Branch left1 x1 right1) = left
    --  (Branch left2 x2 right2) = right

     --(f (f b (LevelO right)) (LevelO left))
 --foldr f b (LevelO (Branch left x right)) = f bftree (bflist (LevelO (Branch left x right)))
 --foldr f b (LevelO (Branch left x right)) = bftree (bflist (LevelO (Branch left x right)))

-- bftree :: [a] -> Levelorder a
-- bftree xs = LevelO t
--   where
--     LevelO t :  q = go xs q
--     go [] _ = repeat (LevelO Nil)
--     go (x : ys) ~(l : ~(r : q)) = LevelO (Branch l x r) : go ys q

-- bflist :: Levelorder a -> [a]
-- bflist t = [x | LevelO (Branch  _ x _) <- q]
--   where
--     q =  t :  go 1 q
--     go 0 _ = []
--     go i (LevelO Nil : q) = go (i -1) q
--     go i (LevelO (Branch  l _ r : q)) = l : r : go (i + 1) q

newtype Levelorder1 a = LevelO1 (Tree a) deriving (Eq, Show)
instance Foldable Levelorder1 where
  -- +++foldr f b (LevelO1 Nil) = b
  -- +++foldr f b (LevelO1 (Branch left x right)) = f x (foldr f (foldr f b (LevelO1 right)) (LevelO1 left))
  -- foldr f b (LevelO (Branch Nil x Nil)) = f x b
  -- foldr f b (LevelO (Branch Nil x right)) = f x (foldr f b (LevelO right))
  -- foldr f b (LevelO (Branch left x Nil)) = f x (foldr f b (LevelO left))
    -- foldr f b (LevelO (Branch left x right)) = foldr1 f b (LevelO (Branch left x right)) arr where
  --   foldr1 f b (LevelO Nil) arr = f arr b
  --   foldr1 f b (LevelO (Branch left x right)) arr = f x (foldr f (foldr f b (LevelO right)) (LevelO left))

  -- foldr f b (LevelO (Branch Nil a right)) = f a (foldl (foldr f) b (LevelO right))
  -- foldr f b (LevelO (Branch left x Nil)) = f x (foldl (foldr f) b (LevelO left :: Levelorder a))
  -- foldr f b (LevelO (Branch left x right)) = f x (foldl (foldr f) (foldl (foldr f) b (LevelO right)) (LevelO left))

--
  -- foldr f b (LevelO1 Nil) = b
  -- foldr f b (LevelO1 (Branch Nil _ Nil)) = b
  -- foldr f b (LevelO1 (Branch Nil _ x)) = foldr f b (LevelO1 x)
  -- foldr f b (LevelO1 (Branch x _ Nil)) = foldr f b (LevelO1 x)
  -- foldr f b (LevelO1 (Branch left x right)) = foldr f (foldr f b (LevelO1 right)) (LevelO1 left)
  -- --foldr f b (LevelO1 (Branch left x right)) = f x (foldr f (foldr f b (LevelO1 right)) (LevelO1 left))
  foldr f b (LevelO1 Nil) = b
  foldr f b (LevelO1 (Branch Nil x Nil)) = f x b
  foldr f b (LevelO1 (Branch Nil x right)) = f x (foldr f b (LevelO1 right))
  foldr f b (LevelO1 (Branch left x Nil)) = f x (foldr f b (LevelO1 left) )
  foldr f b (LevelO1 (Branch left x right)) = foldr f (foldr f b (LevelO1 right)) (LevelO1 left)
  --foldr f b (LevelO1 (Branch left x right)) = f x (foldr f (foldr f b (LevelO1 right)) (LevelO1 left))



--https://stepik.org/lesson/30427/step/6?discussion=371248&unit=11044
{-
Раскрытие магии вложенных foldr на примере преобразования дерева в список.
Через рекурсию порядки обхода дерева описываются просто:
flatTree :: Tree a -> [a]
flatTree Nil = []
flatTree (Branch l x r) =
   flatTree l ++ [x] ++ flatTree r  -- In-order
   [x] ++ flatTree l ++ flatTree r  -- Pre-order
   flatTree l ++ flatTree r ++ [x]  -- Post-order

Далее принцип такой -- самое правое слагаемое идёт как ini в очередной foldr.
Начинаем преобразовывать по слагаемым справа-налево для In-order:
f = (:)
ini = []
flatTree r = foldr f ini r
[x] ++ flatTree r = f x (foldr f ini r)

flatTree l ++ [x] ++ flatTree r
= foldr f ([x] ++ flatTree r) l
= foldr f (f x (foldr f ini r)) l

Теперь для Pre-order:
flatTree l ++ flatTree r = foldr f (foldr f ini r) l
[x] ++ flatTree l ++ flatTree r = f x (foldr f (foldr f ini r) l)

Post-order, начало:
flatTree l ++ flatTree r ++ [x]
= flatTree l ++ flatTree r ++ (f x ini)
= ...
-}

--https://doisinkidney.com/posts/2018-12-18-traversing-graphs.html
{-
bfe :: Tree a -> [a]
bfe r = f r b []
  where
    f (Node x xs) fw bw = x : fw (xs : bw)

    b [] = []
    b qs = foldl (foldr f) b qs []

--
lwe :: Tree a -> [[a]]
lwe r = f b r [] []
  where
    f k (Node x xs) ls qs = k (x : ls) (xs : qs)

    b _ [] = []
    b k qs = k : foldl (foldl f) b qs [] []

--
lwe :: Tree a -> [[a]]
lwe r = f r []
  where
    f (Node x xs) (q : qs) = (x : q) : foldr f qs xs
    f (Node x xs) [] = [x] : foldr f [] xs
-}

-- +++ https://stackoverflow.com/questions/60516485/building-a-binary-tree-not-bst-in-haskell-breadth-first
{-
data Tree a = Empty | Node a (Tree a) (Tree a) deriving (Show)

bft :: [a] -> Tree a
bft xs = head nodes -- Breadth First Tree
  where
    nodes =
      zipWith
        g
        (map Just xs ++ repeat Nothing)
        -- true length of Empty leaves: |xs| + 1
        (pairs $ tail nodes)
    g (Just x) (lt, rt) = Node x lt rt
    g Nothing _ = Empty
    pairs ~(a : ~(b : c)) = (a, b) : pairs c

--
bftree :: [a] -> Tree a
bftree xs = t
  where
    t : q = go xs q
    go [] _ = repeat Empty
    go (x : ys) ~(l : ~(r : q)) = Node x l r : go ys q
--
For comparison, the opposite operation of breadth-first enumeration of a tree is

bflist :: Tree a -> [a]
bflist t = [x | Node x _ _ <- q]
    where
    q  =  t : go 1 q
    go 0  _                =          []
    go i (Empty      : q)  =          go (i-1) q
    go i (Node _ l r : q)  =  l : r : go (i+1) q
-}

--https://stepik.org/lesson/30427/step/6?discussion=3062351&unit=11044
{-
Может поможет кому решить level - order
treenods Nil = []
treenods (Branch Nil _ Nil) = []
treenods (Branch Nil _ b) = [b]
treenods (Branch a _ Nil) = [a]
treenods (Branch a _ b) = [a, b]
-}

-- treenods :: Tree a -> [Tree a]
-- treenods Nil = []
-- treenods (Branch Nil _ Nil) = []
-- treenods (Branch Nil _ b) = [b]
-- treenods (Branch a _ Nil) = [a]
-- treenods (Branch a _ b) = [a, b]

-- --
-- treenods :: Tree a -> [a]
-- treenods Nil = []
-- treenods (Branch Nil _ Nil) = []
-- treenods (Branch Nil _ (Branch l b r)) = [b]
-- treenods (Branch (Branch l a r) _ Nil) = [a]
-- treenods (Branch (Branch l0 a r0) _ (Branch l b r)) = [a, b]

treenods :: Tree a -> [a]
treenods Nil = []
treenods (Branch Nil _ Nil) = []
treenods (Branch Nil _ (Branch l b r)) = [b]
treenods (Branch (Branch l a r) _ Nil) = [a]
treenods (Branch (Branch l0 a r0) _ (Branch l b r)) = [a, b]


tree = Branch (Branch Nil 1 (Branch Nil 2 Nil)) 3 (Branch Nil 4 Nil)
testTree =
   Branch ( Branch (Branch Nil 4 Nil)  
                   2  
                   (Branch (Branch Nil 7 Nil) 
                          5 
                          (Branch Nil 8 Nil))      )
           1
           (Branch Nil 
                   3 
                  (Branch (Branch Nil 9 Nil) 
                           6 
                            Nil))
testTree2 =
  Branch  ( Branch ( Branch (Branch Nil 7 Nil)
                            4
                            (Branch Nil 8 Nil))
                   2
                  (Branch Nil 5 Nil))
           1
         ( Branch  Nil  
                   3
                   (Branch  (Branch Nil 9 Nil)
                            6            
                            Nil  ) )

-- +++ https://stepik.org/lesson/30427/step/6?discussion=373993&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tree) = levelorder [tree]
    where
      levelorder [] = ini
      levelorder (Nil : xs) = levelorder xs
      levelorder ((Branch l x r) : xs) = f x (levelorder (xs ++ [l, r]))
-}

-- +++ https://stepik.org/lesson/30427/step/6?discussion=539007&thread=solutions&unit=11044
{-
next Nil = []
next (Branch l v r) = [l, r]

value Nil = []
value (Branch l v r) = [v]

instance Foldable Levelorder where
  foldr f x (LevelO tree) =
    let calc [] = x
        calc level = foldr f (calc (level >>= next)) (level >>= value)
     in calc [tree]
-}

--https://stepik.org/lesson/30427/step/6?discussion=554128&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
foldr f ini (LevelO Nil) = ini
foldr f ini (LevelO tree) = helper [tree]
  where
    helper [] = ini
    helper (Nil : xs) = helper xs
    helper ((Branch ln v rn) : xs) = f v (helper $ xs ++ [ln, rn])
-}

--https://stepik.org/lesson/30427/step/6?discussion=372546&thread=solutions&unit=11044
{-
values :: Levelorder a -> [a]
values (LevelO Nil) = []
values (LevelO (Branch l x r)) = [x]

children :: Levelorder a -> [Levelorder a]
children (LevelO Nil) = []
children (LevelO (Branch l x r)) = [LevelO l, LevelO r]

instance Foldable Levelorder where
  -- unfoldr :: ([node] -> Maybe ([value], [node])) -> [node] -> [[value]]
  foldr f i tree =
    foldr f i $
      concat $
        unfoldr g [tree]
    where
      g :: [Levelorder a] -> Maybe ([a], [Levelorder a])
      g a =
        if null v && null n
          then Nothing
          else Just (v, n)
        where
          v = concatMap values a
          n = concatMap children a
-}

--https://stepik.org/lesson/30427/step/6?discussion=371837&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldMap f tree = helper f tree []
    where
      helper f (LevelO Nil) [] = mempty
      helper f (LevelO Nil) (q : qs) = helper f q qs
      helper f (LevelO (Branch l x r)) [] = f x <> helper f (LevelO l) [LevelO r]
      helper f (LevelO (Branch l x r)) (q : qs) = f x <> helper f q (qs ++ [LevelO l, LevelO r])
-}

--https://stepik.org/lesson/30427/step/6?discussion=3401382&thread=solutions&unit=11044
{-
g :: Foldable t => (a -> b -> b) -> t a -> b -> b
g = flip . foldr

instance Foldable Levelorder where
  foldr f ini (LevelO tree) =
    foldr f ini $ flatten [tree]

flatten :: [Tree a] -> [a]
flatten [] = []
flatten (Nil : xs) = flatten xs
flatten ((Branch l a r) : xs) = a : flatten (xs ++ [l, r])
-}

--https://stepik.org/lesson/30427/step/6?discussion=373263&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f acc (LevelO Nil) = acc
  foldr f acc (LevelO (Branch l x r)) = foldForest f acc [Branch l x r]

foldForest f acc [] = acc
foldForest f acc (Nil : as) = foldForest f acc as
foldForest f acc ((Branch l x r) : as) = f x $ foldForest f acc (as ++ [l, r])
-}

--https://stepik.org/lesson/30427/step/6?discussion=372852&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f b (LevelO tree) = go b [tree]
    where
      go acc [] = acc
      go acc (t : ts) = case t of
        Nil -> go b ts
        Branch l a r -> f a (go acc (ts ++ [l, r]))
-}

--https://stepik.org/lesson/30427/step/6?discussion=1106763&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO t) = lvl [t]
    where
      lvl [] = ini
      lvl (Nil : ts) = lvl ts
      lvl ((Branch l n r) : ts) = f n $ lvl (ts ++ [l, r])

  foldl f ini (LevelO t) = lvl [t] ini
    where
      lvl [] res = res
      lvl (Nil : ts) res = lvl ts res
      lvl ((Branch l n r) : ts) res = lvl (ts ++ [l, r]) (f res n)
-}

--https://stepik.org/lesson/30427/step/6?discussion=370531&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini x = foldr' f ini [x]
    where
      foldr' _ ini [] = ini
      foldr' f ini ((LevelO Nil) : xs) = foldr' f ini xs
      foldr' f ini ((LevelO (Branch l x r)) : xs) = f x (foldr' f ini (xs ++ [LevelO l, LevelO r]))
-}

--https://stepik.org/lesson/30427/step/6?discussion=4331432&thread=solutions&unit=11044
{-
-- Я не понимаю, как это работает, взял levels отсюда https://doisinkidney.com/posts/2018-06-01-rose-trees-breadth-first-traversing.html
-- Там же десять постов об эффективной реализации BFS. 
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini (LevelO tr) = foldr f ini vals
    where
      vals = concat (levels tr)

      levels :: Tree a -> [[a]]
      levels tr = f tr []
        where
          f (Branch l x r) (y : ys) = (x : y) : foldr f ys [l, r]
          f (Branch l x r) [] = [x] : foldr f [] [l, r]
          f (Nil) (y : ys) = (y : ys)
          f (Nil) [] = []
-}

--https://stepik.org/lesson/30427/step/6?discussion=4315562&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr _ ini (LevelO Nil) = ini
  foldr f ini (LevelO tree) = foldr f ini (evalState bfs [tree])
    where
      bfs :: State [Tree a] [a]
      bfs = do
        nodes <- get
        case nodes of
          [] -> pure []
          (n : ns) -> case n of
            Branch l a r -> do
              let nonNil t = case t of Nil -> False; _ -> True
              put $ ns ++ filter nonNil [l, r]
              rest <- bfs
              pure $ a : rest
            _ -> pure []
-}

--https://stepik.org/lesson/30427/step/6?discussion=3799080&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tree) =
    let g [] xs = xs
        g ts xs = g (foldMap branches ts) $ xs ++ foldMap values ts

        branches Nil = []
        branches (Branch l x r) = [l, r]

        values Nil = []
        values (Branch _ x _) = [x]
     in foldr f ini $ g [tree] []
-}

--https://stepik.org/lesson/30427/step/6?discussion=3386854&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tree) = foldr f ini $ lo ([], [tree], [])
    where
      lo (_, [Nil], _) = []
      lo (nodes, [], []) = nodes
      lo (nodes, [], trees') = lo (nodes, trees', [])
      lo (nodes, (Branch Nil x Nil) : trees, trees') = lo (nodes ++ [x], trees, trees')
      lo (nodes, (Branch Nil x r) : trees, trees') = lo (nodes ++ [x], trees, trees' ++ [r])
      lo (nodes, (Branch l x Nil) : trees, trees') = lo (nodes ++ [x], trees, trees' ++ [l])
      lo (nodes, (Branch l x r) : trees, trees') = lo (nodes ++ [x], trees, trees' ++ [l, r])
-}

--https://stepik.org/lesson/30427/step/6?discussion=3249534&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO t) = foldrs f ini [t]
    where
      foldrs f ini [] = ini
      foldrs f ini xs = foldr (foo f) (foldrs f ini (chld xs)) xs
        where
          foo f (Nil) a = a
          foo f (Branch (t1) i (t2)) a = f i a
          chld [] = []
          chld ((Nil) : xs) = chld xs
          chld ((Branch (t1) i (t2)) : xs) = t1 : t2 : chld xs
-}

--https://stepik.org/lesson/30427/step/6?discussion=3197324&thread=solutions&unit=11044
{-
--Так и не смог найти mZip в библиотеках, пришлось писать самому
mZip :: (Monoid a, Monoid b) => [a] -> [b] -> [(a, b)]
mZip [] [] = []
mZip (x : xs) [] = (x, mempty) : mZip xs []
mZip [] (y : ys) = (mempty, y) : mZip [] ys
mZip (x : xs) (y : ys) = (x, y) : mZip xs ys

mZipWith :: (Monoid a, Monoid b) => (a -> b -> c) -> [a] -> [b] -> [c]
mZipWith f xs ys = map (uncurry f) $ mZip xs ys

levelOrder :: Tree a -> [a]
levelOrder t = concat $ lvls t
  where
    lvls Nil = []
    lvls (Branch l v r) = [v] : mZipWith (++) (lvls l) (lvls r)
-}

--https://stepik.org/lesson/30427/step/6?discussion=3129105&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini (LevelO (Branch l x r)) = foldr f ini (fun [Branch l x r] [])
    where
      fun :: [Tree a] -> [a] -> [a]
      fun [] lv = reverse lv
      fun lt lv = do
        case (last lt) of
          Nil -> fun (init lt) lv
          Branch l' x' r' -> fun (r' : l' : init lt) (x' : lv)
-}

--https://stepik.org/lesson/30427/step/6?discussion=3003166&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f acc (LevelO Nil) = acc
  foldr f acc (LevelO t@(Branch l x r)) = go f acc [t]

go :: (a -> b -> b) -> b -> [Tree a] -> b
go f acc [] = acc
go f acc bs = uncurry g $ unzip (map getValsBrs bs)
  where
    g vs brs = foldr f (go f acc $ concatMap (filter (not . nil)) brs) vs

getValsBrs (Branch l x r) = (x, [l, r])

nil Nil = True
nil _ = False
-}

--https://stepik.org/lesson/30427/step/6?discussion=2911309&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f i t = foldr f i $ lvl [t]
    where
      lvl [] = []
      lvl ((LevelO Nil) : xs) = lvl xs
      lvl (LevelO (Branch l x r) : xs) = x : lvl (xs ++ [LevelO l, LevelO r])
-}

--https://stepik.org/lesson/30427/step/6?discussion=2365738&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  --foldr :: (a -> b -> b) -> b -> Levelorder a -> b
  foldr f ini (LevelO tree) = foldr f ini $ helper tree [] [Nil]
    where
      helper _ list [] = reverse list
      helper Nil list (y : ys) = helper y list ys
      helper (Branch l x r) list (y : ys) = helper y (x : list) (ys ++ [l] ++ [r])
-}

--https://stepik.org/lesson/30427/step/6?discussion=2019325&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tr) = levelhelper [tr]
    where
      levelhelper [Nil] = ini
      levelhelper (Nil : xs) = levelhelper xs
      levelhelper (Branch l x r : xs) = f x (levelhelper (xs ++ [l, r]))
-}

--https://stepik.org/lesson/30427/step/6?discussion=2019140&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tr) = foldr f ini $ concatMap (magic tr) [0 .. height tr]

magic :: Tree a -> Integer -> [a]
magic Nil _ = []
magic (Branch _ x _) 0 = [x]
magic (Branch l _ r) n = magic l (n - 1) ++ magic r (n - 1)

height :: Tree a -> Integer
height Nil = 0
height (Branch l _ r) = 1 + max (height l) (height r)
-}

--https://stepik.org/lesson/30427/step/6?discussion=1925559&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO t) = foldl (flip f) ini (snd $ treeToList ([t], []))
    where
      treeToList ([], vs) = ([], vs)
      treeToList (ts, vs) = treeToList (foldr node ([], vs) ts)
        where
          node Nil x = x
          node (Branch l x r) (trees, vs') = (r : l : trees, x : vs')
-}

--https://stepik.org/lesson/30427/step/6?discussion=1797916&thread=solutions&unit=11044
{-
mycat [] ys = ys
mycat xs [] = xs
mycat (x : xs) (y : ys) = (x ++ y) : mycat xs ys

instance Foldable Levelorder where
  foldr f ini (LevelO t) = foldr f ini (concat $ subfoldr t)
    where
      subfoldr Nil = [[]]
      subfoldr (Branch l n r) =
        let le = subfoldr l
            ri = subfoldr r
            leri = mycat le ri
         in [n] : leri
-}

--https://stepik.org/lesson/30427/step/6?discussion=1580168&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini (LevelO tr) = foldr f ini (bfs [tr] [])
    where
      bfs [] [] = []
      bfs (c@(Branch l x r) : cs) ns = x : bfs cs (push r (push l ns))
        where
          push Nil xs = xs
          push b xs = b : xs
      bfs [] ns = bfs (reverse ns) []
-}

--https://stepik.org/lesson/30427/step/6?discussion=1484503&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini tree = foldr f ini (concat $ treeLevelToList tree [])

treeLevelToList :: Levelorder a -> [[a]] -> [[a]]
treeLevelToList (LevelO Nil) tree = tree
treeLevelToList (LevelO (Branch l x r)) ls = case ls of
  (lvl : lvls) -> (x : lvl) : (left lvls)
  [] -> [x] : (left [])
  where
    right lvls = treeLevelToList (LevelO r) lvls
    left lvls = treeLevelToList (LevelO l) (right lvls)
-}

--https://stepik.org/lesson/30427/step/6?discussion=1435868&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tree) = foldr f ini (rip [tree] [])
    where
      rip [] s = s
      rip (Nil : trees) s = rip trees s
      rip ((Branch l a r) : trees) s = rip (trees ++ [l, r]) (s ++ [a])
-}

--https://stepik.org/lesson/30427/step/6?discussion=1419230&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO t) = helper [t]
    where
      helper [] = ini
      helper (Nil : xs) = helper xs
      helper ((Branch l x r) : xs) = f x (helper (xs ++ [l, r]))
-}

--https://stepik.org/lesson/30427/step/6?discussion=1375590&thread=solutions&unit=11044
{-
current Nil = Nothing
current (Branch l n r) = Just n

siblings Nil = []
siblings (Branch l n r) = [l, r]

instance Foldable Levelorder where
  foldr f ini (LevelO t) = foldr f ini $ mapMaybe current $ concat $ takeWhile (not . null) $ iterate (concatMap siblings) [t]
-}

--https://stepik.org/lesson/30427/step/6?discussion=1365123&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f z (LevelO t) = go f z [t]
    where
      mapTree :: Tree a -> [a]
      mapTree Nil = []
      mapTree (Branch _ x _) = [x]
      go :: (a -> b -> b) -> b -> [Tree a] -> b
      go _ z [] = z
      go _ z [Nil] = z
      go f z ts =
        let values = ts >>= mapTree
            children = do
              (Branch l x r) <- ts
              [l, r]
            foldedChildren = go f z children
            foldedValue = foldr f foldedChildren values
         in foldedValue
-}

--https://stepik.org/lesson/30427/step/6?discussion=1342471&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini t = foldr f ini (S.viewl $ levelOrder $ S.viewl $ S.singleton t)
    where
      levelOrder S.EmptyL = S.empty
      levelOrder ((LevelO Nil) S.:< xs) = levelOrder $ S.viewl xs
      levelOrder ((LevelO (Branch l m r)) S.:< xs) = m S.<| (levelOrder $ S.viewl (xs S.|> (LevelO l) S.|> (LevelO r)))
-}

--https://stepik.org/lesson/30427/step/6?discussion=1221126&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini b = numr f ini b (height b) (height b)

numr f ini b n k = if k == -1 then ini else numr f (num f ini b k) b n (k - 1)

num f ini (LevelO Nil) k = ini
num f ini (LevelO (Branch l x r)) 0 = f x ini
num f ini (LevelO (Branch l x r)) k = num f (num f ini (LevelO r) (k - 1)) (LevelO l) (k - 1)

height (LevelO Nil) = 0
height (LevelO (Branch l x r)) = 1 + max (height (LevelO l)) (height (LevelO r))
-}

--https://stepik.org/lesson/30427/step/6?discussion=1177225&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini root = levelFold [root]
    where
      levelFold [] = ini
      --
      levelFold (LevelO tree : xs) = case tree of
        Nil -> levelFold xs
        Branch l x r -> f x $ levelFold (xs ++ map LevelO [l, r])
-}

--https://stepik.org/lesson/30427/step/6?discussion=1170939&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  --foldl = undefined
  foldr f ini (LevelO lo) = foldr f ini $ fxs [lo]
    where
      get_val Nil = []
      get_val (Branch _ x _) = [x]
      get_trees Nil = []
      get_trees (Branch Nil _ r) = r : []
      get_trees (Branch l _ Nil) = l : []
      get_trees (Branch l _ r) = l : r : []
      fxs :: [Tree a] -> [a]
      fxs [] = []
      fxs xs = (concatMap get_val xs) ++ (fxs $ concatMap get_trees xs)
-}

--https://stepik.org/lesson/30427/step/6?discussion=1133511&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr _ ini (LevelO Nil) = ini
  foldr f ini (LevelO r) = foldr f ini (fst (level ([], [r])))

level :: ([a], [Tree a]) -> ([a], [Tree a])
level (as, []) = (as, [])
level (as, nodes) = level (as ++ (nodeValues nodes), nextLevelNodes nodes)

nodeValues :: [Tree a] -> [a]
nodeValues nodes = nodes >>= (\node -> case node of (Branch _ n _) -> [n]; Nil -> [])

nextLevelNodes :: [Tree a] -> [Tree a]
nextLevelNodes nodes = nodes >>= (\node -> case node of (Branch l _ r) -> [l, r]; Nil -> [])
-}

--https://stepik.org/lesson/30427/step/6?discussion=1099617&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tree) = helper (D.singleton tree)
    where
      helper queue =
        let len = D.length queue
            lastIdx = len - 1
            lastEl = if (len == 0) then Nothing else Just $ D.index queue lastIdx
            queue' = fst $ D.splitAt lastIdx queue
         in case lastEl of
              Nothing -> ini
              (Just (Branch l x r)) -> f x (helper (r D.<| l D.<| queue'))
              (Just Nil) -> helper queue'
-}

--https://stepik.org/lesson/30427/step/6?discussion=977684&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini lt@(LevelO t) = foldr f ini $ q2l lt

q2l :: Levelorder a -> [a]
q2l t =
  let q = push t (Qu [])
   in reverse $ lvlfoldr q [] (:)

lvlfoldr :: Queue (Levelorder a) -> b -> (a -> b -> b) -> b
lvlfoldr q ini f = do
  let (t, q') = pop q
   in case t of
        LevelO Nil -> if (nullq q') then ini else lvlfoldr q' ini f
        LevelO (Branch l x r) -> (let q'' = push (LevelO r) $ push (LevelO l) q' in lvlfoldr q'' (f x ini) f)
-}

--https://stepik.org/lesson/30427/step/6?discussion=954843&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini (LevelO t) = foldr f ini (helper t)
    where
      helper tree =
        fst $
          until
            (\(_, b) -> (length b) == 0)
            (\(flat, leafs) -> (flat ++ gl (head leafs), (tail leafs) ++ brs (head leafs)))
            ([], [tree])
        where
          gl Nil = []
          gl (Branch _ x _) = [x]
          brs Nil = []
          brs (Branch l _ r) = [l, r]
-}

--https://stepik.org/lesson/30427/step/6?discussion=894514&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini t = helper f ini [t]
    where
      helper f ini [] = ini
      helper f ini ((LevelO Nil) : q) = helper f ini q
      helper f ini ((LevelO (Branch l v r)) : q) = v `f` (helper f ini (q ++ [LevelO l] ++ [LevelO r]))
-}

--https://stepik.org/lesson/30427/step/6?discussion=832969&thread=solutions&unit=11044
{-
nstance Foldable Levelorder where
    foldr f ini (LevelO Nil) = ini
    foldr f ini (LevelO tree) = foldLevel (visitNode f) ini [tree]
        where foldLevel :: (Tree a -> b -> b) -> b -> [Tree a] -> b
              foldLevel f ini [] = ini
              foldLevel f ini level = foldr f (foldLevel f ini $ level >>= neighbours) level 
              visitNode :: (a -> b -> b) -> Tree a -> b -> b
              visitNode f Nil ini = ini 
              visitNode f (Branch _  n _) ini = f n ini
-}

--https://stepik.org/lesson/30427/step/6?discussion=559292&thread=solutions&unit=11044
{-
foldNode :: Tree a -> ([a], [Tree a])
foldNode Nil = ([], [])
foldNode (Branch l x r) = ([x], [l, r])

foldLevel nodes =
  let (values, next) = unzip (map foldNode nodes)
   in (concat values, concat next)

foldTree f ini [] = ini
foldTree f ini l =
  let (values, next) = foldLevel l
   in foldr f (foldTree f ini next) values

instance Foldable Levelorder where
  foldr f ini (LevelO tree) = foldTree f ini [tree]
-}

--https://stepik.org/lesson/30427/step/6?discussion=559219&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini (LevelO t) =
    foldr f ini $
      map (\(x, _) -> x) $
        concat $
          takeWhile (not . null) $
            map (\(lvl, lst) -> filter (\(_, lvl') -> lvl == lvl') lst) $
              zip [0 ..] (repeat $ levelTree t 0)
    where
      levelTree Nil lvl = []
      levelTree (Branch l x r) lvl =
        [(x, lvl)] ++ (levelTree l (lvl + 1))
          ++ (levelTree r (lvl + 1))
-}

--https://stepik.org/lesson/30427/step/6?discussion=558504&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tree) =
    let foldr3 :: [[a]] -> Tree a -> [[a]]
        foldr3 [] Nil = [[]]
        foldr3 xs Nil = xs
        foldr3 [] (Branch tl n tr) = [n] : (foldr3 (foldr3 [[]] tl) tr)
        foldr3 (x : xs) (Branch tl n tr) = (x ++ [n]) : (foldr3 (foldr3 xs tl) tr)
     in foldr f ini (Prelude.concat (foldr3 [[]] tree))
-}

--https://stepik.org/lesson/30427/step/6?discussion=527497&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini (LevelO (Branch l v r)) = f v (h [l, r])
    where
      h [] = ini
      h ns = foldr f (h (foldr addChilds [] ns)) (foldr val [] ns)
      addChilds (Nil) rest = rest
      addChilds (Branch l _ r) rest = l : r : rest
      val Nil rest = rest
      val (Branch _ v _) rest = v : rest
-}

--https://stepik.org/lesson/30427/step/6?discussion=494358&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  -- foldr :: (a -> b -> b) -> b -> Preorder a -> b
  foldr _ ini (LevelO Nil) = ini
  foldr f ini (LevelO t) = foldr f ini (concat $ levels t)

levels :: Tree a -> [[a]]
levels Nil = []
levels (Branch l x r) = [x] : concatInOrder (levels l) (levels r)

concatInOrder :: [[a]] -> [[a]] -> [[a]]
concatInOrder xs [] = xs
concatInOrder [] ys = ys
concatInOrder (x : xs) (y : ys) = (x ++ y) : concatInOrder xs ys
-}

--https://stepik.org/lesson/30427/step/6?discussion=457139&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f acc (LevelO Nil) = acc
  foldr f acc (LevelO tree) =
    foldr f acc $
      enqueue (Sequence.singleton tree) Sequence.empty
    where
      enqueue :: Sequence.Seq (Tree a) -> Sequence.Seq a -> Sequence.Seq a
      enqueue seqtree result = case Sequence.viewl seqtree of
        (Branch left root right) Sequence.:< queue ->
          enqueue (queue Sequence.|> left Sequence.|> right) (result Sequence.|> root)
        Nil Sequence.:< queue -> enqueue queue result
        otherwise -> result
-}

--https://stepik.org/lesson/30427/step/6?discussion=393359&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO Nil) = ini
  foldr f ini (LevelO t) = foldr f ini (concat $ takeWhile (not . null) $ listize t)
    where
      listize Nil = repeat []
      listize (Branch l x r) = [x] : zipWith (++) (listize l) (listize r)
-}

--https://stepik.org/lesson/30427/step/6?discussion=375466&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr _ ini (LevelO Nil) = ini
  foldr f ini leveltree = reduce f ini $ traverseBF leveltree
    where
      reduce _ iv [] = iv
      reduce fun iv (x : xs) = f x (reduce fun iv xs)

traverseBF :: Levelorder a -> [a]
traverseBF (LevelO bftree) =
  map fromJust . filter notNothing . tbf $ [bftree]
  where
    tbf [] = []
    tbf xs = map nodeValue xs ++ tbf (concatMap leftAndRightNodes xs)

    nodeValue Nil = Nothing
    nodeValue (Branch _ a _) = Just a

    leftAndRightNodes Nil = []
    leftAndRightNodes (Branch Nil _ Nil) = []
    leftAndRightNodes (Branch Nil _ r) = [r]
    leftAndRightNodes (Branch l _ Nil) = [l]
    leftAndRightNodes (Branch l _ r) = [l, r]

    notNothing Nothing = False
    notNothing _ = True

    fromJust (Just x) = x
    fromJust _ = error "Function fromJust was applied to Nothing"
-}

--https://stepik.org/lesson/30427/step/6?discussion=373881&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr _ ini (LevelO Nil) = ini
  foldr f ini (LevelO t) = foldr f ini (concat . treeToList $ t)

treeToList :: Tree a -> [[a]]
treeToList Nil = [[]]
treeToList (Branch r x l) = [x] : (foldr (lzw (++)) [] (map treeToList [r, l]))

lzw f (a : x) (b : y) = f a b : lzw f x y
lzw f x [] = x
lzw f [] y = y
-}

--https://stepik.org/lesson/30427/step/6?discussion=372578&thread=solutions&unit=11044
{-
import Data.Monoid

foldMapInt :: Monoid b => (b -> b -> b -> b) -> (a -> b) -> Tree a -> b
foldMapInt g f = go
  where
    go Nil = mempty
    go (Branch l m r) = g (go l) (f m) (go r)

zipl :: Monoid a => [a] -> [a] -> [a]
zipl [] x = x
zipl x [] = x
zipl (x : xs) (y : ys) = x <> y : zipl xs ys

instance Foldable Levelorder where
  foldMap f (LevelO t) = foldMap id $ go t
    where
      go Nil = []
      go (Branch l m r) = f m : zipl (go l) (go r)
-}

--https://stepik.org/lesson/30427/step/6?discussion=371751&thread=solutions&unit=11044
{-
instance Foldable Levelorder where
  foldr f ini (LevelO tree) = layered f ini [tree]
    where
      layered :: (a -> b -> b) -> b -> [Tree a] -> b
      layered _ ini' [] = ini'
      layered f' ini' ns = foldr (layerFold f') (layered f' ini' (foldr nextLayer [] ns)) ns

      nextLayer :: Tree a -> [Tree a] -> [Tree a]
      nextLayer Nil acc = acc
      nextLayer (Branch l _ r) acc = l : r : acc

      layerFold :: (a -> b -> b) -> Tree a -> b -> b
      layerFold _ Nil acc = acc
      layerFold f' (Branch _ x _) acc = f' x acc
-}


--2.2.4
{-
Предположим для двоичного дерева
data Tree a = Nil | Branch (Tree a) a (Tree a)   deriving (Eq, Show)
реализован представитель класса типов Foldable, обеспечивающий стратегию обхода pre-order traversal. Какую строку вернет следующий вызов
GHCi> tree = Branch (Branch Nil 1 Nil) 2 (Branch (Branch Nil 3 Nil) 4 (Branch Nil 5 Nil))
GHCi> fst $ sequenceA_ $ (\x -> (show x,x)) <$> tree
-}
tree2 = Branch (Branch Nil 1 Nil) 2 (Branch (Branch Nil 3 Nil) 4 (Branch Nil 5 Nil))
instance Functor Tree where
  fmap f Nil = Nil
  fmap f (Branch l x r) = Branch (fmap f l) (f x) (fmap f r)

-- instance Foldable Tree where
--   foldr f ini Nil = ini
--   foldr f ini (Branch l x r) = f x $ foldr f (foldr f ini r) l


--2.1.8
{-
Предположим, что определены следующие функции
f = Just . getAny . foldMap Any . fmap ev
g = getLast . foldMap Last
h = Just . getAll . foldMap All . map isDigit
Сопоставьте их вызовы и результаты этих вызовов.
Предполагается, что загружены все модули, требующиеся для доступа к использованным функциям и конструкторам данных.
-}
{-
f [3, 5, 6]                        Just True
g [Just True,Just False,Nothing]   Just False   
h [3,5,6]                          error: ...
-}

--2.1.10
{-
Предположим, что у нас реализованы все свертки, основанные на разных стратегиях обхода дерева из предыдущей задачи. 
Какой из вызовов «лучше определен», то есть возвращает результат на более широком классе деревьев?
--
-elem 42 someTree
-elem 42 $ PreO someTree
-elem 42 $ PostO someTree
+elem 42 $ LevelO someTree
-}

