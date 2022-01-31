;--1.
;Nothing but the Truth
;; Complete the expression so it will evaluate to true.
;; (= __ true)
true

;--2.
;Simple Math
;; Innermost forms are evaluated first.
;; (= (- 10 (* 2 3)) __)
4

;--3.
;Strings
;; Clojure strings are Java strings, so you can use Java string methods on them.
;; (= __ (.toUpperCase "hello world"))
"HELLO WORLD"

;--4.
;Lists
;; Lists can be constructed with either a function or a quoted form.
;; (= (list __) '(:a :b :c))
:a :b :c

;--5.
;conj on lists
;; When operating on a list, the conj function will return a new list with one or more items "added" to the front.
;; (= __ (conj '(2 3 4) 1))
;; (= __ (conj '(3 4) 2 1))
'(1 2 3 4)

;--6.
;Vectors
;; Vectors can be constructed several ways. You can compare them with lists.
;; (= [__] (list :a :b :c) (vec '(:a :b :c)) (vector :a :b :c))
:a :b :c 

;--7.
;conj on vectors
;; When operating on a Vector, the conj function will return a new vector with one or more items "added" to the end.
;; (= __ (conj [1 2 3] 4))
;; (= __ (conj [1 2] 3 4))
[1 2 3 4] 

;--8.
;Sets
;; Sets are collections of unique values.
;; (= __ (set '(:a :a :b :c :c :c :c :d :d)))
;; (= __ (clojure.set/union #{:a :b :c} #{:b :c :d}))
#{:a :b :c :d} 

;--9.
;conj on sets
;; When operating on a set, the conj function returns a new set with one or more keys "added" .
;; (= #{1 2 3 4} (conj #{1 4 3} __))
2

;--10.
;Maps
;; Maps store key-value pairs. Both maps and keywords can be used as lookup functions. Commas are whitespace.
;; (= __ ((hash-map :a 10, :b 20, :c 30) :b))
;; (= __ (:b {:a 10, :b 20, :c 30}))
20 

;--11.
;conj on maps
;; When operating on a map, the conj function returns a new map with one or more key-value pairs "added" .
;; (= {:a 1, :b 2, :c 3} (conj {:a 1} __ [:c 3]))
{:b 2} 

;--12.
;Sequences
;; All Clojure collections support sequencing. You can operate on sequences with functions like first, second, and last.
;; (= __ (first '(3 2 1)))
;; (= __ (second [2 3 4]))
;; (= __ (last (list 1 2 3)))
3 

;--13.
;rest
;; The rest function will return all the items of a sequence except the first.
;; (= __ (rest [10 20 30 40]))
[20 30 40]
   
;--14.
;Functions
;; Clojure has many different ways to create functions.
;; (= __ ((fn add-five [x] (+ x 5)) 3))
;; (= __ ((fn [x] (+ x 5)) 3))
;; (= __ (#(+ % 5) 3))
;; (= __ ((partial + 5) 3))
8 

;--15.
;Double Down
;; Write a function which doubles a number.
;; (= (__ 2) 4)
;; (= (__ 3) 6)
;; (= (__ 11) 22)
;; (= (__ 7) 14)
#(* % 2)

;--16.
;Hello World
;; Write a function which returns a personalized greeting.
;; (= (__ "Dave") "Hello, Dave!")
;; (= (__ "Jenn") "Hello, Jenn!")
;; (= (__ "Rhea") "Hello, Rhea!")
#(str "Hello, "%"!") 

;--17.
;map
;; The map function takes two arguments: a function (f) and a sequence (s). 
;; Map returns a new sequence consisting of the result of applying f to each item of s. 
;; Do not confuse the map function with the map data structure.
;; (= __ (map #(+ % 5) '(1 2 3)))
'(6 7 8)

;--18.
;filter
;; The filter function takes two arguments: a predicate function (f) and a sequence (s). 
;; Filter returns a new sequence consisting of all the items of s for which (f item) returns true.
;; (= __ (filter #(> % 5) '(3 4 5 6 7)))
'(6 7)

;--35.
;Local bindings
;; Clojure lets you give local names to values using the special let-form.
;; (= __ (let [x 5] (+ 2 x)))
;; (= __ (let [x 3, y 10] (- y x)))
;; (= __ (let [x 21] (let [y 3] (/ x y))))
7

;--36.
;Let it Be
;; Can you bind x, y, and z so that these are all true?
;; (= 10 (let __ (+ x y)))
;; (= 4 (let __ (+ y z)))
;; (= 1 (let __ z))
[x 7, y 3, z 1] 

;--37.
;Regular Expressions
;; Regex patterns are supported with a special reader macro.
;; (= __ (apply str (re-seq #"[A-Z]+" "bA1B3Ce ")))
"ABC"

;--52.
;Intro to Destructuring
;; Let bindings and function parameter lists support destructuring.
;; (= [2 4] (let [[a b c d e f g] (range)] __))
[c e]

;--57.
;Simple Recursion
;; A recursive function is a function which calls itself. This is one of the fundamental techniques used in functional programming.
;; (= __ ((fn foo [x] (when (> x 0) (conj (foo (dec x)) x))) 5))
'(5 4 3 2 1)

;--64.
;Intro to Reduce
;; Reduce takes a 2 argument function and an optional starting value. 
;; It then applies the function to the first 2 items in the sequence 
;; (or the starting value and the first element of the sequence) . 
;; In the next iteration the function will be called on the previous 
;; return value and the next item from the sequence, thus reducing the 
;; entire collection to one value. Don't worry, it's not as complicated as it sounds.
;; (= 15 (reduce __ [1 2 3 4 5]))
;; (=  0 (reduce __ []))
;; (=  6 (reduce __ 1 [2 3]))
+

;;-68.
;Recurring Theme
;; Clojure only has one non-stack-consuming looping construct: recur. Either a function or a loop 
;; can be used as the recursion point. Either way, recur rebinds the bindings of the recursion point 
;; to the values it is passed. Recur must be called from the tail-position, and calling it elsewhere will result in an error.
;;     (= __
;;       (loop [x 5
;;              result []]
;;         (if (> x 0)
;;           (recur (dec x) (conj result (+ 2 x)))
;;           result)))
[7 6 5 4 3]

;--71.
;Rearranging Code: ->
;; The -> macro threads an expression x through a variable number of forms. 
;; First, x is inserted as the second item in the first form, making a list of it if it is not a list already. 
;; Then the first form is inserted as the second item in the second form, making a list of that form if necessary. 
;; This process continues for all the forms. Using -> can sometimes make your code more readable.
;; (= (__ (sort (rest (reverse [2 5 4 1 3 6]))))
;;    (-> [2 5 4 1 3 6] reverse rest sort __)
;;    5)
last

;--72.
;Rearranging Code: ->>
;; The ->> macro threads an expression x through a variable number of forms. 
;; First, x is inserted as the last item in the first form, making a list of it if it is not a list already. 
;; Then the first form is inserted as the last item in the second form, making a list of that form if necessary. 
;; This process continues for all the forms. Using ->> can sometimes make your code more readable.
;; (= (__ (map inc (take 3 (drop 2 [2 5 4 1 3 6]))))
;;    (->> [2 5 4 1 3 6] (drop 2) (take 3) (map inc) (__))
;;    11)
reduce +

;--134.
;A nil key
;; Write a function which, given a key and map, returns true iff the map contains an entry with that key and its value is nil.
;; (true?  (__ :a {:a nil :b 2}))
;; (false? (__ :b {:a nil :b 2}))
;; (false? (__ :c {:a nil :b 2}))
(fn [k m]
  (and
   (contains? m k)
   (nil? (k m))))

;--145.
;For the win
;; Clojure's for macro is a tremendously versatile mechanism for producing a sequence based on some other sequence (s).
;; It can take some time to understand how to use it properly, but that investment will be paid back with clear, 
;; concise sequence-wrangling later. With that in mind, read over these for expressions and try to see how each 
;; of them produces the same result.
;; (= (__) (for [x (range 40)
;;             :when (= 1 (rem x 4))]
;;         x))
;; (= __ (for [x (iterate #(+ 4 %) 0)
;;             :let [z (inc x)]
;;             :while (< z 40)]
;;         z))
;; (= __ (for [[x y] (partition 2 (range 20))]
;;         (+ x y)))
'(1 5 9 13 17 21 25 29 33 37)

;--156.
;Map Defaults
;; When retrieving values from a map, you can specify default values in case the key is not found: (= 2 (:foo {:bar 0, :baz 1} 2)) 
;; However, what if you want the map itself to contain the default values? Write a function which takes a default value 
;; and a sequence of keys and constructs a map.
;;     (= (__ 0 [:a :b :c]) {:a 0 :b 0 :c 0})
;;     (= (__ "x" [1 2 3]) {1 "x" 2 "x" 3 "x"})
;;     (= (__ [:a :b] [:foo :bar]) {:foo [:a :b] :bar [:a :b]})
(fn [x m] (apply assoc {} (interleave m (repeat (count m) x))))

;--161.
;Subset and Superset
;; Set A is a subset of set B, or equivalently B is a superset of A, if A is "contained" inside B. A and B may coincide.
;; (clojure.set/superset? __ #{2})
;; (clojure.set/subset? #{1} __)
;; (clojure.set/superset? __ #{1 2})
;; (clojure.set/subset? #{1 2} __)
#{1 2}

;--162.
;Logical falsity and truth
;; In Clojure, only nil and false represent the values of logical falsity in conditional tests - anything else is logical truth.
;; (= __ (if-not false 1 0))
;; (= __ (if-not nil 1 0))
;; (= __ (if true 1 0))
;; (= __ (if [] 1 0))
;; (= __ (if [0] 1 0))
;; (= __ (if 0 1 0))
;; (= __ (if 1 1 0))
1